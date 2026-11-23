// ifetch.cpp - see ifetch.h.
#include "ifetch.h"

#include <algorithm>
#include <cstdlib>
#include <queue>
#include <stdexcept>

namespace lpu {
namespace comp {

using namespace sim;
using sched::MemKind;
using sched::Pipeline;
using sched::POp;

namespace {

// One program entry of one IQ, with the cycle it leaves the IQ (issue; NOP: start; Repeat: first
// repeat) - the same rule the simulator uses.
struct Item {
    int way;
    std::size_t idx;
    std::int64_t t;
    std::uint32_t len;
};

std::uint32_t entry_bytes(const QEntry& e) { return e.kind == QEntry::Nop ? kNopBytes : kInsBytes; }

// Compute-way items only: the fetch way (last) is assigned separately.
std::vector<Item> items_of(const IQProgram& P) {
    std::vector<Item> out;
    const int nw = static_cast<int>(P.ways.size());
    for (int w = 0; w < nw; ++w) {
        if (w == nw - 1) continue;
        std::int64_t t = 0, last = -1;
        const auto& es = P.ways[static_cast<std::size_t>(w)];
        for (std::size_t k = 0; k < es.size(); ++k) {
            const QEntry& e = es[k];
            if (e.kind == QEntry::Nop) {
                out.push_back({w, k, t, entry_bytes(e)});
                t += e.n;
            } else if (e.kind == QEntry::Ins) {
                out.push_back({w, k, t, entry_bytes(e)});
                last = t;
                t = t + 1;
            } else {
                out.push_back({w, k, last + e.d, entry_bytes(e)});
                last += static_cast<std::int64_t>(e.n) * e.d;
                t = last + 1;
            }
        }
    }
    std::stable_sort(out.begin(), out.end(), [](const Item& a, const Item& b) {
        if (a.t != b.t) return a.t < b.t;
        if (a.way != b.way) return a.way < b.way;
        return a.idx < b.idx;
    });
    return out;
}

// 32-bit encoding [Ours]: byte 0 = opcode, bytes 1-3 = packed operand fields. NOP = opcode only.
void encode(const QEntry& e, std::vector<std::uint8_t>* out) {
    if (e.kind == QEntry::Nop) {
        out->push_back(static_cast<std::uint8_t>(Opcode::Nop));
        return;
    }
    std::uint32_t f = 0;
    std::uint8_t op;
    if (e.kind == QEntry::Rep) {
        op = static_cast<std::uint8_t>(Opcode::Repeat);
        f = (e.n & 0xFFFFu) | ((e.d & 0xFFu) << 16);
    } else {
        const Instruction& in = e.ins.in;
        op = static_cast<std::uint8_t>(in.opcode);
        f = (in.address & 0x1FFFu) | ((in.dst.base & 31u) << 13) | ((in.dst.dir == Direction::West ? 1u : 0u) << 18) |
            ((in.src[0].base & 31u) << 19);
    }
    out->push_back(op);
    out->push_back(static_cast<std::uint8_t>(f));
    out->push_back(static_cast<std::uint8_t>(f >> 8));
    out->push_back(static_cast<std::uint8_t>(f >> 16));
}

// Bytes the previous piece carries for the next Ifetch's own entries on the fetch way: up to two
// NOPs (a gap over 65535 cycles splits), the Ifetch and its Repeat, plus the fetch way's Sync NOP.
constexpr std::uint32_t kFetchReserve = 11;

struct Group {
    std::int64_t T = INT64_MIN;        // window start: items leaving at t >= T (and < next T) belong here
    std::uint32_t len = 0;             // planned bytes (incl. reserve for the next Ifetch); upper bound
    std::uint32_t cap = kIqBytes;      // bytes the piece can hold: 640 for the preload, 320 n otherwise
    std::uint32_t limit = kIqBytes;    // len may grow to this (text Reads placed on a MEM IQ later)
    std::uint32_t extra = 0;           // bytes that may land in the piece ahead of a planned item
    int n = 0;                         // text vectors (320 bytes each): 1 or 2 (Ifetch + Repeat 1)
    int slice = -1, bank = 0, base = 0;
    std::int64_t start = 0, end = 0;   // first Ifetch issue; last chunk lands at end = start + 20 n - 1
};

struct IqPlan {
    std::vector<Item> items;           // compute items, in leave order (way -1: estimate for a text Read)
    std::vector<std::uint64_t> prefix; // prefix[i] = bytes of items[0, i)
    std::vector<Group> groups;         // groups[0] = preload
    std::size_t next = 0;              // first item not yet in a group
    std::int64_t rend = 0;             // MEM IQs: end of the R-type way's content (chip cycles)
    bool r_empty = true;
    bool mem = false;                  // MEM IQ: text Reads for other IQs may land in its R way
    void reprefix() {
        prefix.assign(items.size() + 1, 0);
        for (std::size_t i = 0; i < items.size(); ++i) prefix[i + 1] = prefix[i] + items[i].len;
    }
    std::int64_t next_T() const { return next < items.size() ? items[next].t : INT64_MAX; }
};

// Bytes the text Reads of one refill add to their MEM IQ's R-type way, and when they leave: one
// Read per vector at t and t + 20 (each a separate instruction), the way's Sync NOP if it was empty,
// a NOP for the gap before a Read that lands past the way's last entry, and a NOP for the split of
// the gap a Read lands in. An upper bound on what emit() produces.
struct Delta { std::int64_t t; std::uint32_t b; };
std::vector<Delta> read_deltas(const IqPlan& P, std::int64_t t, int n) {
    std::vector<Delta> d;
    if (P.r_empty) d.push_back({0, 1});                 // the way's Sync NOP
    std::int64_t rend = P.rend;
    for (int v = 0; v < n; ++v) {
        const std::int64_t x = t + static_cast<std::int64_t>(v) * kIfetchRepeatGap;
        if (x > rend) d.push_back({rend, 1});           // gap before
        d.push_back({x, kInsBytes});
        if (x + 1 < std::max(rend, t + static_cast<std::int64_t>(n - 1) * kIfetchRepeatGap + 1))
            d.push_back({x + 1, 1});                    // gap after (split)
        rend = std::max(rend, x + 1);
    }
    return d;
}

// Can IQ P take these bytes? A byte that leaves inside a planned piece's window uses that piece's
// spare room (16 n minus its planned bytes), but only if every later Ifetch of this IQ starts after
// the byte leaves (so the extra byte never sits in the IQ at a later Ifetch's start); bytes after
// the last planned piece join the unplanned items. All bytes join the item list, so later room
// checks count them as issued when they leave.
bool take(IqPlan& P, const std::vector<Delta>& d, bool apply) {
    const std::int64_t tn = P.next_T();
    std::vector<std::uint32_t> add(P.groups.size(), 0);
    auto by_t = [](const Item& a, std::int64_t t) { return a.t < t; };
    for (const Delta& x : d) {
        if (x.t >= tn) continue;
        std::size_t j = P.groups.size() - 1;
        while (j > 0 && x.t < P.groups[j].T) --j;
        for (std::size_t h = j + 1; h < P.groups.size(); ++h)
            if (P.groups[h].start <= x.t) return false;
        if (j > 0) {
            // the byte's own 16-byte chunk must land before it leaves: it sits after the piece's
            // items that leave no later than it, plus the bytes that may come ahead of them
            const Group& G = P.groups[j];
            const std::size_t lo = static_cast<std::size_t>(std::lower_bound(P.items.begin(), P.items.end(), G.T, by_t) - P.items.begin());
            const std::size_t hi = static_cast<std::size_t>(std::lower_bound(P.items.begin(), P.items.end(), x.t + 1, by_t) - P.items.begin());
            const std::uint64_t off = P.prefix[hi] - P.prefix[lo] + G.extra + add[j];
            const std::int64_t c = std::min<std::int64_t>(static_cast<std::int64_t>((off + x.b - 1) / kFetchBytesPerCycle),
                                                          static_cast<std::int64_t>(G.n) * kChunksPerVec - 1);
            if (G.start + c >= x.t) return false;
        }
        add[j] += x.b;
    }
    for (std::size_t j = 0; j < P.groups.size(); ++j)
        if (P.groups[j].len + add[j] > std::min(P.groups[j].limit, P.groups[j].cap)) return false;
    if (!apply) return true;
    for (std::size_t j = 0; j < P.groups.size(); ++j) P.groups[j].len += add[j];
    for (const Delta& x : d) {
        Item it{-1, 0, x.t, x.b};
        auto pos = std::upper_bound(P.items.begin(), P.items.end(), it, [](const Item& a, const Item& b) { return a.t < b.t; });
        const std::size_t at = static_cast<std::size_t>(pos - P.items.begin());
        P.items.insert(pos, it);
        if (at <= P.next && x.t < tn) ++P.next;
    }
    P.reprefix();
    return true;
}

// Move a cut so that it falls between two different leave cycles (pieces are windows in time).
std::size_t cut_at_time(const std::vector<Item>& it, std::size_t lo, std::size_t k, bool allow_empty) {
    const std::size_t k0 = k;
    while (k > lo && k < it.size() && it[k].t == it[k - 1].t) --k;
    if (k > lo || (allow_empty && k == lo)) return k;
    k = k0;
    while (k < it.size() && k > 0 && it[k].t == it[k - 1].t) ++k;
    return k;
}

Pipeline fetch_pipeline(int iq, int slice, int base, int n, std::int64_t min_issue) {
    Pipeline p;
    p.name = "ifetch";
    POp r;
    r.queue = q_mem_read(slice);
    r.ins.in.opcode = Opcode::MemRead;
    r.mk = MemKind::ReadSeq;
    r.mem_g = slice;
    r.addr0 = base;
    r.stride = 1;
    r.n = n;
    r.kind = 'R';
    r.layer = -1;
    r.label = "instruction text read";
    POp f;
    f.queue = q_fetch(iq);
    f.ins.in.opcode = Opcode::Ifetch;
    f.n = n;
    f.kind = 'F';
    f.layer = -1;
    f.label = "Ifetch";
    f.min_issue = min_issue;
    const int a = p.add_op(r), b = p.add_op(f);
    const int e = p.add_edge(1);
    p.produce(a, e, 0, 1);
    p.consume(b, 0, e, 0, 1);
    return p;
}

}  // namespace

Program build_program(Compiler& cc, sched::Scheduler& sch, Simulator& sim, const Floorplan& fp, const IfetchOptions& opt,
                      IfetchStats* st) {
    *st = IfetchStats{};
    st->program_bytes.assign(kNumIQs, 0);
    Program prog = sch.emit();

    // ---- plan: preload group, then Ifetch groups in deadline order across all IQs ----
    // All times here are chip cycles (Sync preamble included), like the items' leave times.
    const std::int64_t sy = sch.lat().sync_cycles;
    std::vector<IqPlan> plan(kNumIQs);
    using PQ = std::pair<std::int64_t, int>;   // (deadline, iq)
    std::priority_queue<PQ, std::vector<PQ>, std::greater<PQ>> pq;
    for (int iq = 0; iq < kNumIQs; ++iq) {
        IqPlan& P = plan[static_cast<std::size_t>(iq)];
        const IQProgram& IP = prog.iqs[static_cast<std::size_t>(iq)];
        P.items = items_of(IP);
        P.reprefix();
        P.mem = q_is_mem(channel_of(iq, 0));
        if (P.mem) {
            // end of the R-type way: the cycle after its last instruction
            const auto& es = IP.ways[0];
            std::int64_t t = 0, last = -1;
            for (const QEntry& e : es) {
                if (e.kind == QEntry::Nop) t += e.n;
                else if (e.kind == QEntry::Ins) { last = t; t += 1; }
                else { last += static_cast<std::int64_t>(e.n) * e.d; t = last + 1; }
            }
            P.r_empty = es.empty();
            P.rend = es.empty() ? sy : std::max<std::int64_t>(t, sy);
        }
        Group g0;
        const std::uint64_t total = P.prefix.back();
        const std::uint32_t room0 = kIqBytes - kFetchReserve - (P.mem ? opt.mem_headroom : 0);
        const bool fits = total <= room0;
        std::size_t k = P.items.size();
        if (!fits) {
            k = 0;
            while (k < P.items.size() && P.prefix[k + 1] <= room0) ++k;
            k = cut_at_time(P.items, 0, k, true);
        }
        g0.T = INT64_MIN;
        g0.len = static_cast<std::uint32_t>(P.prefix[k] + (P.items.empty() ? 0 : kFetchReserve));
        g0.cap = kIqBytes;
        g0.limit = kIqBytes;
        P.groups.push_back(g0);
        P.next = k;
        if (k < P.items.size()) pq.push({P.items[k].t, iq});
    }
    std::vector<int> fetch_use(kMemSlices, 0);
    int rejected = 0, no_room = 0;
    while (!pq.empty()) {
        const int iq = pq.top().second;
        pq.pop();
        IqPlan& P = plan[static_cast<std::size_t>(iq)];
        const int gi = static_cast<int>(P.groups.size());   // this refill brings group gi
        const std::int64_t deadline = P.items[P.next].t;
        std::uint64_t fetched_before = 0;
        for (const Group& g : P.groups) fetched_before += g.len;
        // fetch-way bytes surely gone before this refill: its Sync NOP, earlier Ifetches and Repeats
        std::int64_t fw_left = 1;
        for (int h = 1; h < gi; ++h) fw_left += kInsBytes * (P.groups[static_cast<std::size_t>(h)].n > 1 ? 2 : 1);
        // the fetch way is busy until the previous refill's last chunk has landed
        const std::int64_t prev_free = gi > 1 ? P.groups.back().end + 1 : -1;
        const std::uint32_t head = P.mem ? static_cast<std::uint32_t>(opt.mem_headroom) : 0u;
        // candidate slices: nearest to the IQ first, then least used for text
        const int ipos = fp.iq_pos(iq);
        std::vector<int> slices(kMemSlices);
        for (int g = 0; g < kMemSlices; ++g) slices[static_cast<std::size_t>(g)] = g;
        std::stable_sort(slices.begin(), slices.end(), [&](int a, int b) {
            const int da = std::abs(fp.mem_pos[static_cast<std::size_t>(a)] - ipos), db = std::abs(fp.mem_pos[static_cast<std::size_t>(b)] - ipos);
            if (da != db) return da < db;
            return fetch_use[static_cast<std::size_t>(a)] < fetch_use[static_cast<std::size_t>(b)];
        });
        struct Try { std::size_t k; std::uint32_t len; int n, slice; std::int64_t lb, ub, start; };
        Try best{};
        bool have = false, ok = false;
        std::uint32_t cap = static_cast<std::uint32_t>(std::min(opt.chunk_bytes, kRefillVecs * kTextVecBytes)) - head;
        for (int shrink = 0; shrink < 4 && !ok; ++shrink, cap /= 2) {
            // items for this refill
            std::size_t k = P.next;
            while (k < P.items.size()) {
                const bool more = k + 1 < P.items.size();
                const std::uint64_t L = P.prefix[k + 1] - P.prefix[P.next] + (more ? kFetchReserve : 0);
                if (L > cap && k > P.next) break;
                ++k;
            }
            k = cut_at_time(P.items, P.next, k, false);
            const bool more = k < P.items.size();
            const std::uint32_t len = static_cast<std::uint32_t>(P.prefix[k] - P.prefix[P.next] + (more ? kFetchReserve : 0));
            const std::uint32_t room = len + head;
            const int n = static_cast<int>((room + kTextVecBytes - 1) / kTextVecBytes);
            if (n > kRefillVecs) continue;   // one item larger than a refill cannot happen (4 B), but be safe
            const int nch = static_cast<int>((room + kFetchBytesPerCycle - 1) / kFetchBytesPerCycle);
            // lower bound (room): chunk c lands at s + c; when it lands the IQ holds everything
            // fetched before plus chunks 0..c (16 B each), minus what has surely left
            std::int64_t s_space = INT64_MIN;
            for (int c = 0; c < nch; ++c) {
                const std::int64_t need = static_cast<std::int64_t>(fetched_before) + kFetchBytesPerCycle * (c + 1) - kIqBytes - fw_left;
                if (need <= 0) continue;
                const auto it = std::lower_bound(P.prefix.begin(), P.prefix.end(), static_cast<std::uint64_t>(need));
                if (it == P.prefix.end()) throw std::runtime_error("Ifetch: refill cannot fit in " + fp.iq_name(iq));
                const std::size_t j = static_cast<std::size_t>(it - P.prefix.begin());   // items [0, j) left
                if (j > 0) s_space = std::max(s_space, P.items[j - 1].t - c);
            }
            // upper bound (no underflow): each item's last byte is in chunk c, which must land before
            // the item leaves; bytes that may come ahead of it: the fetch way's own entries and the
            // MEM headroom
            const std::uint32_t extra = kFetchReserve + head;
            std::int64_t ub = INT64_MAX;
            for (std::size_t i = P.next; i < k; ++i) {
                const std::uint64_t o = P.prefix[i] - P.prefix[P.next] + extra;
                const std::int64_t c = std::min<std::int64_t>(static_cast<std::int64_t>((o + P.items[i].len - 1) / kFetchBytesPerCycle), nch - 1);
                ub = std::min(ub, P.items[i].t - 1 - c);
            }
            const std::int64_t lb = std::max({prev_free, s_space, sy});
            int tried = 0;
            for (int g : slices) {
                if (tried >= opt.candidates) break;
                if (iq_mem(g) == iq) continue;   // its own R way: the piece's indices would move
                int bank = 0, base = 0;
                if (!cc.alloc_text(g, n, &bank, &base, false)) { ++no_room; continue; }
                // first vector at the earliest start s >= lb whose second vector can go at s + 20
                std::int64_t want = lb, start = -1, rt = 0;
                for (int it = 0; it < 64; ++it) {
                    Pipeline p0 = fetch_pipeline(iq, g, base, 1, want - sy);
                    const std::int64_t S0 = sch.place(p0, false);
                    const std::int64_t s0 = S0 + p0.ops[1].phase + p0.ops[1].m0 + sy;
                    if (n == 1) { start = s0; rt = S0 + p0.ops[0].phase + p0.ops[0].m0 + sy; break; }
                    Pipeline p1 = fetch_pipeline(iq, g, base + 1, 1, s0 + kIfetchRepeatGap - sy);
                    p1.edges[0].force_dir = p0.edges[0].trial_dir;   // the Repeat reads the Ifetch's own streams
                    p1.edges[0].force_base = p0.edges[0].trial_base;
                    const std::int64_t S1 = sch.place(p1, false);
                    const std::int64_t s1 = S1 + p1.ops[1].phase + p1.ops[1].m0 + sy;
                    const std::int64_t r0 = S0 + p0.ops[0].phase + p0.ops[0].m0 + sy;
                    const std::int64_t r1 = S1 + p1.ops[0].phase + p1.ops[0].m0 + sy;
                    if (s1 == s0 + kIfetchRepeatGap && r1 == r0 + kIfetchRepeatGap) { start = s0; rt = r0; break; }
                    want = std::max(s0 + 1, s1 - kIfetchRepeatGap);
                }
                if (start < 0) continue;
                IqPlan& R = plan[static_cast<std::size_t>(iq_mem(g))];
                if (!take(R, read_deltas(R, rt, n), false)) {
                    ++rejected;
                    continue;
                }
                ++tried;
                if (!have || start < best.start) {
                    best = Try{k, len, n, g, lb, ub, start};
                    have = true;
                }
            }
            if (have && best.start <= best.ub && best.k == k) ok = true;
            else if (shrink + 1 < 4) ++st->shrunk;
        }
        if (!have)
            throw std::runtime_error("Ifetch: no MEM slice can serve instruction text of IQ " + fp.iq_name(iq) + " (deadline " +
                                     std::to_string(deadline) + ", piece " + std::to_string(P.groups.size()) + "; so far " +
                                     std::to_string(no_room) + " slice tries had no free MEM, " + std::to_string(rejected) +
                                     " could not take the Read in their own IQ text)");
        if (!ok) ++st->late;
        // commit the best try: Ifetch (vector 0) and its Repeat (vector 1, 20 cycles later)
        int bank = 0, base = 0;
        cc.alloc_text(best.slice, best.n, &bank, &base, true);
        std::int64_t rt = 0;
        int fdir = -1, fbase = -1;
        for (int v = 0; v < best.n; ++v) {
            const std::int64_t s_v = best.start + static_cast<std::int64_t>(v) * kIfetchRepeatGap;
            Pipeline p = fetch_pipeline(iq, best.slice, base + v, 1, s_v - sy);
            p.edges[0].force_dir = fdir;
            p.edges[0].force_base = fbase;
            sch.place(p, true);
            fdir = p.edges[0].dir == Direction::East ? 0 : 1;
            fbase = p.edges[0].base;
            if (p.ops[1].start + sy != s_v) throw std::runtime_error("Ifetch: placement differs from its trial");
            if (v == 0) rt = p.ops[0].start + sy;
            else if (p.ops[0].start + sy != rt + kIfetchRepeatGap) throw std::runtime_error("Ifetch: text Read spacing differs from its trial");
        }
        IqPlan& R = plan[static_cast<std::size_t>(iq_mem(best.slice))];
        if (!take(R, read_deltas(R, rt, best.n), true)) throw std::runtime_error("Ifetch: text Read no longer fits its IQ");
        R.r_empty = false;
        R.rend = std::max(R.rend, rt + static_cast<std::int64_t>(best.n - 1) * kIfetchRepeatGap + 1);
        Group g;
        g.T = P.items[P.next].t;
        g.len = best.len;
        g.cap = static_cast<std::uint32_t>(best.n * kTextVecBytes);
        g.limit = best.len + head;
        g.extra = kFetchReserve + head;
        g.n = best.n;
        g.slice = best.slice;
        g.bank = bank;
        g.base = base;
        g.start = best.start;
        g.end = best.start + static_cast<std::int64_t>(best.n) * kChunksPerVec - 1;
        st->min_slack = std::min(st->min_slack, best.ub - best.start);
        P.groups.push_back(g);
        P.next = best.k;
        fetch_use[static_cast<std::size_t>(best.slice)]++;
        ++st->ifetches;
        st->ifetch_vectors += static_cast<std::uint64_t>(best.n);
        st->read_port_cycles += static_cast<std::uint64_t>(best.n);
        st->stream_cycles += static_cast<std::uint64_t>(best.n) *
                             static_cast<std::uint64_t>(std::abs(fp.mem_pos[static_cast<std::size_t>(best.slice)] - ipos) + 1);
        if (P.next < P.items.size()) pq.push({P.items[P.next].t, iq});
    }
    st->rejected = rejected;

    // ---- final program: assign every entry to its piece by leave time, build the text ----
    prog = sch.emit();
    // The scheduler placed each refill's two vectors as two Ifetches 20 cycles apart: encode them
    // as Ifetch + Repeat 1 (d = 20), and retag the second text Read as repeat 1 of the first.
    for (int iq = 0; iq < kNumIQs; ++iq) {
        auto& es = prog.iqs[static_cast<std::size_t>(iq)].ways[static_cast<std::size_t>(iq_fetch_way(iq))];
        const IqPlan& PL = plan[static_cast<std::size_t>(iq)];
        std::vector<QEntry> out;
        std::size_t g = 1;
        for (std::size_t k = 0; k < es.size(); ++k) {
            if (es[k].kind != QEntry::Ins) { out.push_back(es[k]); continue; }
            if (g >= PL.groups.size()) throw std::runtime_error("Ifetch: fetch way has more Ifetches than planned");
            const Group& G = PL.groups[g++];
            out.push_back(es[k]);
            if (G.n == 1) continue;
            if (k + 2 >= es.size() || es[k + 1].kind != QEntry::Nop || es[k + 1].n != kIfetchRepeatGap - 1 || es[k + 2].kind != QEntry::Ins)
                throw std::runtime_error("Ifetch: refill vectors of " + fp.iq_name(iq) + " are not 20 cycles apart");
            const SimInstr& v0 = es[k].ins;
            const SimInstr& v1 = es[k + 2].ins;
            bool found = false;
            for (QEntry& e : prog.iqs[static_cast<std::size_t>(iq_mem(G.slice))].ways[0])
                if (e.kind == QEntry::Ins && e.ins.in.opcode == Opcode::MemRead && e.ins.out_edge == v1.in_edge[0] &&
                    e.ins.out_idx0 == v1.in_idx0[0]) {
                    e.ins.out_edge = v0.in_edge[0];
                    e.ins.out_idx0 = v0.in_idx0[0] + 1;
                    found = true;
                    break;
                }
            if (!found) throw std::runtime_error("Ifetch: second text Read not found");
            QEntry r;
            r.kind = QEntry::Rep;
            r.n = 1;
            r.d = kIfetchRepeatGap;
            out.push_back(r);
            k += 2;
        }
        es.swap(out);
    }
    for (int iq = 0; iq < kNumIQs; ++iq) {
        IqPlan& PL = plan[static_cast<std::size_t>(iq)];
        IQProgram& P = prog.iqs[static_cast<std::size_t>(iq)];
        const int fw = iq_fetch_way(iq);
        const std::size_t ng = PL.groups.size();
        std::vector<std::vector<Item>> members(ng);
        const auto comp = items_of(P);
        if (comp.empty() && P.ways[static_cast<std::size_t>(fw)].empty()) continue;
        for (const Item& it : comp) {
            std::size_t gi = ng - 1;
            while (gi > 0 && it.t < PL.groups[gi].T) --gi;
            members[gi].push_back(it);
        }
        // fetch way: NOPs before Ifetch g, Ifetch g and its Repeat go with group g-1
        {
            std::int64_t t = 0, last = -1;
            std::size_t seen = 0;
            const auto& es = P.ways[static_cast<std::size_t>(fw)];
            for (std::size_t k = 0; k < es.size(); ++k) {
                const QEntry& e = es[k];
                Item it{fw, k, 0, entry_bytes(e)};
                std::size_t grp;
                if (e.kind == QEntry::Nop) { it.t = t; t += e.n; grp = seen; }
                else if (e.kind == QEntry::Ins) { it.t = t; last = t; t = t + 1; grp = seen; ++seen; }
                else { it.t = last + e.d; last += static_cast<std::int64_t>(e.n) * e.d; t = last + 1; grp = seen - 1; }
                if (grp >= ng) throw std::runtime_error("Ifetch: fetch way has more Ifetches than planned");
                members[grp].push_back(it);
            }
            if (seen != ng - 1) throw std::runtime_error("Ifetch: fetch way Ifetch count mismatch");
        }
        std::vector<std::uint32_t> goff(ng + 1, 0);
        for (std::size_t gi = 0; gi < ng; ++gi) {
            auto& m = members[gi];
            std::stable_sort(m.begin(), m.end(), [](const Item& a, const Item& b) { return a.t < b.t; });
            goff[gi] = static_cast<std::uint32_t>(P.text.size());
            for (const Item& it : m) {
                QEntry& e = P.ways[static_cast<std::size_t>(it.way)][it.idx];
                e.byte_off = static_cast<std::uint32_t>(P.text.size());
                e.byte_len = it.len;
                encode(e, &P.text);
            }
            const std::uint32_t got = static_cast<std::uint32_t>(P.text.size()) - goff[gi];
            if (got > PL.groups[gi].len || got > PL.groups[gi].cap)
                throw std::runtime_error("Ifetch: piece " + std::to_string(gi) + " of " + fp.iq_name(iq) + " holds " +
                                         std::to_string(got) + " B, planned " + std::to_string(PL.groups[gi].len) + " B");
        }
        goff[ng] = static_cast<std::uint32_t>(P.text.size());
        P.preload = goff[1];
        P.capacity = kIqBytes;
        // Ifetch g: its text range, and the text words in MEM
        std::size_t g = 1;
        for (QEntry& e : P.ways[static_cast<std::size_t>(fw)]) {
            if (e.kind != QEntry::Ins) continue;
            const Group& G = PL.groups[g];
            e.ins.text_off = goff[g];
            e.ins.text_len = goff[g + 1] - goff[g];
            for (int w = 0; w < G.n; ++w) {
                Vector v{};
                for (int l = 0; l < kTextVecBytes; ++l) {
                    const std::uint32_t b = goff[g] + static_cast<std::uint32_t>(w * kTextVecBytes + l);
                    if (b < goff[g + 1]) v[static_cast<std::size_t>(l)] = P.text[b];
                }
                sim.mem(G.slice)[static_cast<std::size_t>(G.base + w)] = v;
            }
            st->text_words += static_cast<std::uint64_t>(G.n);
            ++g;
        }
        st->program_bytes[static_cast<std::size_t>(iq)] = static_cast<std::uint32_t>(P.text.size());
        st->text_bytes += P.text.size();
        st->preload_bytes += P.preload;
        st->fetched_bytes += P.text.size() - P.preload;
        ++st->iqs_used;
        if (ng > 1) ++st->iqs_fetching;
    }
    return prog;
}

}  // namespace comp
}  // namespace lpu
