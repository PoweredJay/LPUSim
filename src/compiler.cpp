// compiler.cpp - see compiler.h.
#include "compiler.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace lpu {
namespace comp {

using namespace lpu::sim;
using namespace lpu::sched;

namespace {

POp op_read(int g, int addr, int n, int m0, int stride = 1) {
    POp o;
    o.queue = q_mem_read(g);
    o.ins.in.opcode = Opcode::MemRead;
    o.mk = MemKind::ReadSeq;
    o.mem_g = g;
    o.addr0 = addr;
    o.stride = stride;
    o.n = n;
    o.m0 = m0;
    o.kind = 'R';
    return o;
}

POp op_write(int g, int addr, int n, int m0) {
    POp o;
    o.queue = q_mem_write(g);
    o.ins.in.opcode = Opcode::MemWrite;
    o.mk = MemKind::WriteSeq;
    o.mem_g = g;
    o.addr0 = addr;
    o.n = n;
    o.m0 = m0;
    o.kind = 'W';
    return o;
}

POp op_vxm(Opcode op, int n, int imm = 0) {
    POp o;
    o.queue = -1;
    o.ins.in.opcode = op;
    o.ins.in.imm = imm;
    o.n = n;
    o.kind = 'V';
    return o;
}

// VXM op consuming edges a (and b), producing a new edge of width wout. Returns the new edge.
int vxm(Pipeline& p, Opcode op, int n, int ea, int wa, int eb, int wb, int wout, int imm, int layer,
        const std::string& label) {
    POp o = op_vxm(op, n, imm);
    o.layer = layer;
    o.label = label;
    const int id = p.add_op(o);
    p.consume(id, 0, ea, 0, wa);
    if (eb >= 0) p.consume(id, 1, eb, 0, wb);
    const int e = p.add_edge(wout);
    p.produce(id, e, 0, wout);
    return e;
}

Vector map_word(const int* addr20) {
    Vector v{};
    for (int t = 0; t < kSuperlanes; ++t) {
        v[static_cast<std::size_t>(t * 16)] = static_cast<std::uint8_t>(addr20[t] & 0xff);
        v[static_cast<std::size_t>(t * 16 + 1)] = static_cast<std::uint8_t>(addr20[t] >> 8);
    }
    return v;
}

const int kPlaneOrder[4] = {0, 2, 1, 3};

}  // namespace

bool Compiler::PipeCtx::can_read(int g, int bank) const {
    if (std::find(rd.begin(), rd.end(), g) != rd.end()) return false;
    if (single && std::find(wr.begin(), wr.end(), g) != wr.end()) return false;
    for (auto& [s, b] : wr_bank)
        if (s == g && b == bank) return false;
    return true;
}
bool Compiler::PipeCtx::can_write(int g, int bank) const {
    if (std::find(wr.begin(), wr.end(), g) != wr.end()) return false;
    if (single && std::find(rd.begin(), rd.end(), g) != rd.end()) return false;
    for (auto& [s, b] : rd_bank)
        if (s == g && b == bank) return false;
    return true;
}

Compiler::Compiler(const Net& net, Simulator& loader, Scheduler& sch, const Floorplan& fp, CompileOptions opt)
    : net_(net),
      sim_(loader),
      sch_(sch),
      fp_(fp),
      opt_(opt),
      ts_(net.tensors.size()),
      free_clean_(kMemSlices),
      free_rt_(kMemSlices),
      act_use_(kMemSlices, 0),
      w_use_(kMemSlices, 0),
      map_use_(kMemSlices, 0) {
    for (auto& f : free_clean_) {
        f[0] = {{1, kBankWords}};
        f[1] = {{kBankWords + 1, kMemWords}};
    }
    plan();
}

// ---------------------------------------------------------------------------------------------
// Planning
// ---------------------------------------------------------------------------------------------
Layout Compiler::layout_for(int t) const {
    const TensorDesc& d = net_.tensors[static_cast<std::size_t>(t)];
    bool dypack = false;
    for (const LayerDesc& L : net_.layers)
        if (L.kind == LayerDesc::Conv && L.in == t && L.k > 1) dypack = true;
    Layout lay;
    for (int c0 = 0; c0 < d.C; c0 += kLanes) {
        Layout::Word w;
        w.ch0 = c0;
        w.nch = std::min(kLanes, d.C - c0);
        const int L = w.sl();
        const int copies = dypack ? std::min(3, kSuperlanes / L) : 1;
        for (int k = 0; k < copies; ++k) w.copies.push_back(k * L);
        lay.words.push_back(w);
    }
    return lay;
}

std::vector<Compiler::KVec> Compiler::kvecs(const LayerDesc& L, const Layout& in) const {
    std::vector<KVec> kv;
    if (L.k == 1) {
        for (int w = 0; w < static_cast<int>(in.words.size()); ++w) {
            KVec v;
            v.items.push_back(KItem{0, 0, w, 0});
            v.identity = L.stride == 1;
            kv.push_back(v);
        }
        return kv;
    }
    for (int dy = -L.pad; dy < L.k - L.pad; ++dy)
        for (int w = 0; w < static_cast<int>(in.words.size()); ++w) {
            const auto& copies = in.words[static_cast<std::size_t>(w)].copies;
            KVec cur;
            for (int dx = -L.pad; dx < L.k - L.pad; ++dx) {
                cur.items.push_back(KItem{dy, dx, w, copies[cur.items.size()]});
                if (cur.items.size() == copies.size()) { kv.push_back(cur); cur = KVec{}; }
            }
            if (!cur.items.empty()) kv.push_back(cur);
        }
    return kv;
}

int Compiler::alu_need(const LayerDesc& L, const std::vector<Slot>& slots, int nk) const {
    std::map<std::pair<int, int>, std::pair<int, std::pair<int, int>>> g;   // (j,chunk) -> (cnt,(minv,maxv))
    for (const Slot& s : slots) {
        auto it = g.find({s.j, s.chunk});
        if (it == g.end()) g[{s.j, s.chunk}] = {1, {s.v, s.v}};
        else {
            it->second.first++;
            it->second.second.first = std::min(it->second.second.first, s.v);
            it->second.second.second = std::max(it->second.second.second, s.v);
        }
    }
    int alu = 0;
    for (auto& [k, v] : g) {
        alu += v.first - 1 + (v.second.first > 0 ? 1 : 0);
        if (v.second.second == nk - 1) alu += L.raw32 ? 0 : (L.res >= 0 ? 5 : (L.relu ? 3 : 2));
    }
    return alu;
}

std::vector<std::vector<Compiler::Slot>> Compiler::pack_rounds(const LayerDesc& L, int nk, int nc, int nchunks) const {
    std::vector<std::vector<Slot>> rounds;
    std::vector<Slot> cur;
    for (int c = 0; c < nchunks; ++c)
        for (int j = 0; j < nc; ++j)
            for (int v = 0; v < nk; ++v) {
                std::vector<Slot> t = cur;
                t.push_back(Slot{v, j, c});
                if (cur.size() == kPlanes || alu_need(L, t, nk) > kVxmAlus) {
                    rounds.push_back(cur);
                    cur = {Slot{v, j, c}};
                } else {
                    cur = t;
                }
            }
    if (!cur.empty()) rounds.push_back(cur);
    return rounds;
}

std::vector<std::pair<int, int>> Compiler::split_rows(int H, int c) {
    std::vector<std::pair<int, int>> r;
    int y = 0;
    for (int i = 0; i < c; ++i) {
        const int n = H / c + (i < H % c ? 1 : 0);
        r.push_back({y, y + n});
        y += n;
    }
    return r;
}

void Compiler::plan() {
    for (std::size_t t = 0; t < ts_.size(); ++t) {
        ts_[t].lay = layout_for(static_cast<int>(t));
        for (const LayerDesc& L : net_.layers) {
            if (L.in == static_cast<int>(t)) ts_[t].consumers_left++;
            if (L.res == static_cast<int>(t)) ts_[t].consumers_left++;
        }
    }
    ts_[static_cast<std::size_t>(net_.input)].host = true;
    plans_.resize(net_.layers.size());
    for (std::size_t li = 0; li < net_.layers.size(); ++li) {
        const LayerDesc& L = net_.layers[li];
        Plan& P = plans_[li];
        const TensorDesc& out = net_.tensors[static_cast<std::size_t>(L.out)];
        if (L.kind == LayerDesc::Conv) {
            P.kv = kvecs(L, ts_[static_cast<std::size_t>(L.in)].lay);
            const int nk = static_cast<int>(P.kv.size());
            const int nc = static_cast<int>(ts_[static_cast<std::size_t>(L.out)].lay.words.size());
            long best = LONG_MAX;
            for (int c : {1, 2, 4}) {
                if (c > out.H) continue;
                auto chunks = split_rows(out.H, c);
                auto rounds = pack_rounds(L, nk, nc, c);
                long cost = 0;
                for (auto& r : rounds) {
                    int mx = 0;
                    for (auto& s : r) mx = std::max(mx, (chunks[static_cast<std::size_t>(s.chunk)].second - chunks[static_cast<std::size_t>(s.chunk)].first) * out.W);
                    cost += mx + opt_.round_overhead;
                }
                if (cost < best) {
                    best = cost;
                    P.chunks = chunks;
                    P.rounds = rounds;
                }
            }
            for (auto& r : P.rounds) P.slots_per_round = std::max(P.slots_per_round, static_cast<int>(r.size()));
        } else if (L.kind == LayerDesc::MaxPool) {
            const int c = std::min(4, out.H);
            P.chunks = split_rows(out.H, c);
            // two 9-input max chains per round (8 ALUs each); pair chunk i with i + c/2
            const int half = (c + 1) / 2;
            for (int i = 0; i < half; ++i) {
                std::vector<Slot> r{Slot{0, 0, i}};
                if (i + half < c) r.push_back(Slot{0, 0, i + half});
                P.rounds.push_back(r);
            }
            P.slots_per_round = 9;
        } else {
            P.chunks = {{0, out.H}};
        }
        ts_[static_cast<std::size_t>(L.out)].pieces = P.chunks;
    }
    // host input pieces = chunks of its consumer; replicas from consumers
    for (std::size_t li = 0; li < net_.layers.size(); ++li) {
        const LayerDesc& L = net_.layers[li];
        TState& in = ts_[static_cast<std::size_t>(L.in)];
        if (in.host && in.pieces.empty()) {
            // a host tensor feeding a 1x1 stride-1 conv takes the conv's row chunks, else one piece
            const TensorDesc& dt = net_.tensors[static_cast<std::size_t>(L.in)];
            if (L.kind == LayerDesc::Conv && L.k == 1 && L.stride == 1) in.pieces = plans_[li].chunks;
            else in.pieces = {{0, dt.H}};
        }
        int need = 1;
        if (L.kind == LayerDesc::Conv) need = plans_[li].slots_per_round;
        else if (L.kind == LayerDesc::MaxPool) need = 9;
        else need = 2;
        in.R = std::max(in.R, need);
        if (L.res >= 0) ts_[static_cast<std::size_t>(L.res)].R = std::max(ts_[static_cast<std::size_t>(L.res)].R, 2);
    }
    if (net_.tensors[static_cast<std::size_t>(net_.output)].int32) ts_[static_cast<std::size_t>(net_.output)].R = 4;
}

// ---------------------------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------------------------
int Compiler::alloc_common(const FreeList& free_, const std::vector<int>& slices, int size, int bank) {
    // candidate bases: starts of free intervals of every slice
    std::vector<int> cands;
    for (int g : slices)
        for (auto& iv : free_[static_cast<std::size_t>(g)][static_cast<std::size_t>(bank)]) cands.push_back(iv.first);
    std::sort(cands.begin(), cands.end());
    for (int b : cands) {
        bool ok = true;
        for (int g : slices) {
            bool in = false;
            for (auto& iv : free_[static_cast<std::size_t>(g)][static_cast<std::size_t>(bank)])
                if (iv.first <= b && b + size <= iv.second) { in = true; break; }
            if (!in) { ok = false; break; }
        }
        if (ok) return b;
    }
    return -1;
}

std::vector<int> Compiler::pick_slices(bool west, int count, int size, int bank, const std::vector<int>& exclude,
                                       int lo, int hi, std::vector<int>& use, int* base, bool in_range_only) {
    // Constants (weights, maps; use != act_use_) only take never-used space: a preloaded
    // constant must not sit where runtime data was or will be written.
    const bool constant = &use != &act_use_;
    for (int pass = 0; pass < (constant ? 2 : 4); ++pass) {
        auto& free_ = (!constant && pass < 2) ? free_rt_ : free_clean_;
        const int range_pass = pass % 2;
        if (in_range_only && range_pass == 1) continue;
        std::vector<int> cand;
        for (int i = 0; i < kMemPerHem; ++i) {
            if (range_pass == 0 && (i < lo || i > hi)) continue;
            const int g = Floorplan::mem_global(west, i);
            if (std::find(exclude.begin(), exclude.end(), g) != exclude.end()) continue;
            cand.push_back(g);
        }
        std::stable_sort(cand.begin(), cand.end(), [&](int a, int b) { return use[static_cast<std::size_t>(a)] < use[static_cast<std::size_t>(b)]; });
        std::vector<int> chosen;
        for (int g : cand) {
            std::vector<int> t = chosen;
            t.push_back(g);
            if (alloc_common(free_, t, size, bank) >= 0) chosen = t;
            if (static_cast<int>(chosen.size()) == count) break;
        }
        if (constant && pass == 1 && static_cast<int>(chosen.size()) < count) {
            // Last resort for constants: greedy by slice failed (free space is fragmented differently per slice): search by
            // base instead - every interval start / end-aligned base, take the least-used slices
            // that have [base, base + size) free.
            std::vector<int> bases;
            for (int g : cand)
                for (auto& iv : free_[static_cast<std::size_t>(g)][static_cast<std::size_t>(bank)])
                    if (iv.second - iv.first >= size) { bases.push_back(iv.first); bases.push_back(iv.second - size); }
            std::sort(bases.begin(), bases.end());
            bases.erase(std::unique(bases.begin(), bases.end()), bases.end());
            for (int b : bases) {
                std::vector<int> fit;
                for (int g : cand)
                    for (auto& iv : free_[static_cast<std::size_t>(g)][static_cast<std::size_t>(bank)])
                        if (iv.first <= b && b + size <= iv.second) { fit.push_back(g); break; }
                if (static_cast<int>(fit.size()) >= count) { fit.resize(static_cast<std::size_t>(count)); chosen = fit; break; }
            }
        }
        if (static_cast<int>(chosen.size()) == count) {
            const int b = alloc_common(free_, chosen, size, bank);
            for (int g : chosen) {
                auto& fl = free_[static_cast<std::size_t>(g)][static_cast<std::size_t>(bank)];
                for (std::size_t k = 0; k < fl.size(); ++k) {
                    if (fl[k].first <= b && b + size <= fl[k].second) {
                        const auto iv = fl[k];
                        fl.erase(fl.begin() + static_cast<long>(k));
                        if (b + size < iv.second) fl.insert(fl.begin() + static_cast<long>(k), {b + size, iv.second});
                        if (iv.first < b) fl.insert(fl.begin() + static_cast<long>(k), {iv.first, b});
                        break;
                    }
                }
                use[static_cast<std::size_t>(g)]++;
            }
            *base = b;
            alloc_log.push_back(AllocRec{alloc_what_, &use == &w_use_ ? 'W' : &use == &map_use_ ? 'M' : alloc_kind_, chosen, bank, b,
                                         size, static_cast<long>(sch_.placed())});
            live_words_ += static_cast<std::uint64_t>(size) * static_cast<std::uint64_t>(count);
            peak_words = std::max(peak_words, live_words_);
            return chosen;
        }
    }
    throw std::runtime_error("out of MEM: need " + std::to_string(count) + " slices x " + std::to_string(size) +
                             " words in bank " + std::to_string(bank));
}

bool Compiler::alloc_text(int g, int words, int* bank, int* base, bool commit) {
    for (int b : {1, 0}) {
        const int a = alloc_common(free_clean_, {g}, words, b);
        if (a < 0) continue;
        *bank = b;
        *base = a;
        if (!commit) return true;
        auto& fl = free_clean_[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
        for (std::size_t k = 0; k < fl.size(); ++k) {
            if (fl[k].first <= a && a + words <= fl[k].second) {
                const auto iv = fl[k];
                fl.erase(fl.begin() + static_cast<long>(k));
                if (a + words < iv.second) fl.insert(fl.begin() + static_cast<long>(k), {a + words, iv.second});
                if (iv.first < a) fl.insert(fl.begin() + static_cast<long>(k), {iv.first, a});
                break;
            }
        }
        live_words_ += static_cast<std::uint64_t>(words);
        peak_words = std::max(peak_words, live_words_);
        text_words += static_cast<std::uint64_t>(words);
        alloc_log.push_back(AllocRec{"instruction text", 'I', {g}, b, a, words, static_cast<long>(sch_.placed()), -1});
        return true;
    }
    return false;
}

void Compiler::free_store(const Store& s) {
    for (auto it = alloc_log.rbegin(); it != alloc_log.rend(); ++it)
        if (it->pipe_free < 0 && it->bank == s.bank && it->base == s.base && it->size == s.size && it->slices == s.slices) {
            it->pipe_free = static_cast<long>(sch_.placed());
            break;
        }
    for (int g : s.slices) {
        auto& fl = free_rt_[static_cast<std::size_t>(g)][static_cast<std::size_t>(s.bank)];
        fl.push_back({s.base, s.base + s.size});
        std::sort(fl.begin(), fl.end());
        std::vector<std::pair<int, int>> m;
        for (auto iv : fl) {
            if (!m.empty() && m.back().second >= iv.first) m.back().second = std::max(m.back().second, iv.second);
            else m.push_back(iv);
        }
        fl = m;
    }
    live_words_ -= static_cast<std::uint64_t>(s.size) * s.slices.size();
}

void Compiler::alloc_tensor(int t, const std::vector<int>& exclude, int bank) {
    TState& T = ts_[static_cast<std::size_t>(t)];
    if (T.allocated) return;
    const TensorDesc& d = net_.tensors[static_cast<std::size_t>(t)];
    std::vector<int> used = exclude;
    T.store.assign(T.pieces.size(), std::vector<Store>(T.lay.words.size()));
    for (std::size_t p = 0; p < T.pieces.size(); ++p)
        for (std::size_t w = 0; w < T.lay.words.size(); ++w) {
            Store& s = T.store[p][w];
            s.size = (T.pieces[p].second - T.pieces[p].first) * d.W;
            s.bank = bank;
            // every store of a tensor sits in one hemisphere, so all sub-range readers of one
            // operand stream in the same direction; tensors alternate hemispheres
            const bool west = (t % 2) == 0;
            alloc_what_ = d.name + (T.pieces.size() > 1 ? " p" + std::to_string(p) : "") +
                          (T.lay.words.size() > 1 ? " w" + std::to_string(w) : "");
            alloc_kind_ = 'A';
            s.slices = pick_slices(west, T.R, s.size, bank, used, opt_.act_index_lo, opt_.act_index_hi, act_use_, &s.base);
            used.insert(used.end(), s.slices.begin(), s.slices.end());
        }
    T.allocated = true;
}

// Output bank: opposite of the layer input's bank; slices whose live input/residual data is in
// the same bank are excluded, so a layer never reads and writes one bank of one slice at once.
int Compiler::out_bank(const LayerDesc& L, std::vector<int>* exclude) const {
    const TState& Ti = ts_[static_cast<std::size_t>(L.in)];
    const int bank = Ti.store.empty() ? 0 : 1 - Ti.store[0][0].bank;
    for (int t : {L.in, L.res}) {
        if (t < 0) continue;
        for (const auto& ps : ts_[static_cast<std::size_t>(t)].store)
            for (const auto& s : ps)
                if (s.bank == bank || opt_.mem_single_issue) exclude->insert(exclude->end(), s.slices.begin(), s.slices.end());
    }
    // A later layer that reads this output together with another tensor (c3: input + residual)
    // needs the two on different slices, or their reads would share one read port.
    for (const LayerDesc& L2 : net_.layers) {
        int other = -1;
        if (L2.in == L.out) other = L2.res;
        if (L2.res == L.out) other = L2.in;
        if (other < 0 || !ts_[static_cast<std::size_t>(other)].allocated) continue;
        for (const auto& ps : ts_[static_cast<std::size_t>(other)].store)
            for (const auto& s : ps) exclude->insert(exclude->end(), s.slices.begin(), s.slices.end());
    }
    return bank;
}

void Compiler::free_tensor(int t) {
    TState& T = ts_[static_cast<std::size_t>(t)];
    if (!T.allocated || T.freed || t == net_.output || !opt_.free_tensors) return;
    for (auto& ps : T.store)
        for (auto& s : ps) free_store(s);
    T.freed = true;
}

int Compiler::piece_of_row(const TState& T, int row) const {
    for (std::size_t p = 0; p < T.pieces.size(); ++p)
        if (row >= T.pieces[p].first && row < T.pieces[p].second) return static_cast<int>(p);
    return -1;
}

void Compiler::load_input(const std::vector<std::int8_t>& chw) {
    const int t = net_.input;
    TState& T = ts_[static_cast<std::size_t>(t)];
    const TensorDesc& d = net_.tensors[static_cast<std::size_t>(t)];
    alloc_tensor(t, {}, 0);
    for (std::size_t p = 0; p < T.pieces.size(); ++p)
        for (std::size_t w = 0; w < T.lay.words.size(); ++w) {
            const Store& s = T.store[p][w];
            const Layout::Word& lw = T.lay.words[w];
            for (int y = T.pieces[p].first; y < T.pieces[p].second; ++y)
                for (int x = 0; x < d.W; ++x) {
                    Vector v{};
                    for (int sl : lw.copies)
                        for (int c = 0; c < lw.nch; ++c)
                            v[static_cast<std::size_t>(sl * 16 + c)] = static_cast<std::uint8_t>(
                                chw[static_cast<std::size_t>((lw.ch0 + c) * d.H * d.W + y * d.W + x)]);
                    const int a = s.base + (y - T.pieces[p].first) * d.W + x;
                    for (int g : s.slices) sim_.mem(g)[static_cast<std::size_t>(a)] = v;
                }
        }
}

// Weight tile (v, j) of layer li as 320 columns of 320 bytes in 16 slices x 20 words.
Compiler::Store Compiler::weight_tile(int li, int v, int j, bool west) {
    const long key = ((static_cast<long>(li) * 4096 + v) * 64 + j) * 2 + (west ? 1 : 0);
    auto it = wtiles_.find(key);
    if (it != wtiles_.end()) return it->second;
    const LayerDesc& L = net_.layers[static_cast<std::size_t>(li)];
    const Layout& inl = ts_[static_cast<std::size_t>(L.in)].lay;
    const Layout::Word& ow = ts_[static_cast<std::size_t>(L.out)].lay.words[static_cast<std::size_t>(j)];
    const KVec& kv = plans_[static_cast<std::size_t>(li)].kv[static_cast<std::size_t>(v)];
    std::vector<std::int8_t> W(static_cast<std::size_t>(kLanes * kLanes), 0);   // [f][i]
    for (int a : ow.copies)
        for (int c = 0; c < ow.nch; ++c) {
            const int f = a * 16 + c, co = ow.ch0 + c;
            for (const KItem& it2 : kv.items) {
                const Layout::Word& iw = inl.words[static_cast<std::size_t>(it2.word)];
                for (int ci = 0; ci < iw.nch; ++ci) {
                    const int i = it2.sl * 16 + ci;
                    const int ky = it2.dy + L.pad, kx = it2.dx + L.pad;
                    W[static_cast<std::size_t>(f * kLanes + i)] =
                        L.w[static_cast<std::size_t>(((co * L.cin + iw.ch0 + ci) * L.k + ky) * L.k + kx)];
                }
            }
        }
    Store s;
    s.size = kSuperlanes;
    s.bank = -1;
    alloc_what_ = L.name + " W v" + std::to_string(v) + " j" + std::to_string(j);
    // both banks of the weight region first, then anywhere in the hemisphere
    for (int strict = 1; strict >= 0 && s.bank < 0; --strict)
        for (int bank : {1, 0}) {
            try {
                s.slices = pick_slices(west, 16, kSuperlanes, bank, {}, opt_.weight_index_lo, opt_.weight_index_hi, w_use_,
                                       &s.base, strict != 0);
                s.bank = bank;
                break;
            } catch (const std::exception&) {
            }
        }
    if (s.bank < 0) {
        if (!opt_.alias_weights_when_full) {
            std::uint64_t fc[2] = {0, 0};
            for (std::size_t g = 0; g < free_clean_.size(); ++g)
                for (int b = 0; b < 2; ++b)
                    for (auto& iv : free_clean_[g][static_cast<std::size_t>(b)]) fc[g < 44 ? 0 : 1] += static_cast<std::uint64_t>(iv.second - iv.first);
            throw std::runtime_error("out of MEM for weights (layer " + L.name + ", " + std::to_string(weight_words) +
                                     " weight words so far, live " + std::to_string(live_words_) + ", clean free W/E " +
                                     std::to_string(fc[0]) + "/" + std::to_string(fc[1]) +
                                     "); --alias-weights gives a timing-only projection");
        }
        // Timing-only projection: MEM is full, so reuse the slices of an earlier tile on this
        // hemisphere (same port / bank pattern a real allocation would have). The tile's data
        // is NOT written, so results are not bit-exact; the run is flagged.
        std::vector<const Store*> same;
        for (const auto& kv2 : wtiles_)
            if (((kv2.first & 1) != 0) == west && !kv2.second.slices.empty()) same.push_back(&kv2.second);
        if (same.empty()) throw std::runtime_error("out of MEM for weights");
        s = *same[static_cast<std::size_t>(weights_aliased) % same.size()];
        weights_aliased++;
        tiles_total++;
        weight_words += 16 * kSuperlanes;
        wtiles_[key] = s;
        return s;
    }
    for (int r = 0; r < kSuperlanes; ++r)
        for (int st = 0; st < 16; ++st) {
            const int f = 16 * r + st;
            Vector col;
            for (int i = 0; i < kLanes; ++i) col[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(W[static_cast<std::size_t>(f * kLanes + i)]);
            sim_.mem(s.slices[static_cast<std::size_t>(st)])[static_cast<std::size_t>(s.base + r)] = col;
        }
    weight_words += 16 * kSuperlanes;
    tiles_total++;
    wtiles_[key] = s;
    return s;
}

int Compiler::choose_replica(const TState& T, int piece, int word, int pref, const PipeCtx& ctx) const {
    const Store& s = T.store[static_cast<std::size_t>(piece)][static_cast<std::size_t>(word)];
    for (int k = 0; k < T.R; ++k) {
        const int r = (pref + k) % T.R;
        if (ctx.can_read(s.slices[static_cast<std::size_t>(r)], s.bank)) return r;
    }
    return pref % T.R;   // let the scheduler report it
}

void Compiler::add_act_source(Pipeline& p, PipeCtx& ctx, int act_edge, int tin, int word, int rep, int n,
                              const std::vector<int>& yy, const std::vector<int>& xx, bool identity, int layer,
                              const std::string& tag) {
    const TState& T = ts_[static_cast<std::size_t>(tin)];
    const TensorDesc& d = net_.tensors[static_cast<std::size_t>(tin)];
    int m = 0;
    int last_piece = 0;
    while (m < n) {
        int piece = (yy[static_cast<std::size_t>(m)] >= 0 && yy[static_cast<std::size_t>(m)] < d.H) ? piece_of_row(T, yy[static_cast<std::size_t>(m)]) : last_piece;
        int mb = m + 1;
        while (mb < n) {
            const int y2 = yy[static_cast<std::size_t>(mb)];
            const int p2 = (y2 >= 0 && y2 < d.H) ? piece_of_row(T, y2) : piece;
            if (p2 != piece) break;
            ++mb;
        }
        last_piece = piece;
        const Store& s = T.store[static_cast<std::size_t>(piece)][static_cast<std::size_t>(word)];
        const int r = choose_replica(T, piece, word, rep, ctx);
        const int g = s.slices[static_cast<std::size_t>(r)];
        const int row0 = T.pieces[static_cast<std::size_t>(piece)].first;
        ctx.read(g, s.bank);
        if (identity) {
            POp o = op_read(g, s.base + (yy[static_cast<std::size_t>(m)] - row0) * d.W + xx[static_cast<std::size_t>(m * kSuperlanes)], mb - m, m);
            o.layer = layer;
            o.label = tag + " read";
            const int id = p.add_op(o);
            p.produce(id, act_edge, 0, 1);
        } else {
            const int cnt = mb - m;
            std::vector<std::uint16_t> gl(static_cast<std::size_t>(cnt * kSuperlanes));
            const int zero = s.bank * kBankWords;
            for (int i = 0; i < cnt; ++i) {
                const int y = yy[static_cast<std::size_t>(m + i)];
                for (int t = 0; t < kSuperlanes; ++t) {
                    const int x = xx[static_cast<std::size_t>((m + i) * kSuperlanes + t)];
                    int a = zero;
                    if (y >= 0 && y < d.H && x >= 0 && x < d.W) {
                        if (y < row0 || y >= T.pieces[static_cast<std::size_t>(piece)].second)
                            throw std::runtime_error("gather crosses a piece boundary");
                        a = s.base + (y - row0) * d.W + x;
                    }
                    gl[static_cast<std::size_t>(i * kSuperlanes + t)] = static_cast<std::uint16_t>(a);
                }
            }
            // map slice: same hemisphere as the gather slice, not read elsewhere in this pipeline
            std::vector<int> excl = ctx.rd;
            if (ctx.single) excl.insert(excl.end(), ctx.wr.begin(), ctx.wr.end());
            for (const auto& ps : T.store)
                for (const auto& st : ps) excl.insert(excl.end(), st.slices.begin(), st.slices.end());
            int mbase = 0;
            int mbank = 1;
            std::vector<int> ms;
            // least-used map slice on either side (a map stream may cross the VXM)
            {
                std::vector<int> order(kMemSlices);
                std::iota(order.begin(), order.end(), 0);
                auto key = [&](int g2) {
                    const int ix = Floorplan::mem_index(g2);
                    return (ix > opt_.act_index_hi || ix < opt_.act_index_lo ? 1000000 : 0) + map_use_[static_cast<std::size_t>(g2)];
                };
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key(a) < key(b); });
                for (int cand : order) {
                    if (std::find(excl.begin(), excl.end(), cand) != excl.end()) continue;
                    for (int bank : {1, 0}) {
                        bool clash = false;
                        for (auto& [sg, sb] : ctx.wr_bank) if (sg == cand && sb == bank) clash = true;
                        if (clash) continue;
                        std::vector<int> ex2;
                        for (int i = 0; i < kMemSlices; ++i) if (i != cand) ex2.push_back(i);
                        try {
                            alloc_what_ = net_.layers[static_cast<std::size_t>(layer)].name + " gather map";
                            ms = pick_slices(Floorplan::mem_west(cand), 1, cnt, bank, ex2, 0, kMemPerHem - 1, map_use_, &mbase);
                            mbank = bank;
                        } catch (const std::exception&) {
                        }
                        if (!ms.empty()) break;
                    }
                    if (!ms.empty()) break;
                }
            }
            if (ms.empty()) throw std::runtime_error("out of MEM for gather maps");
            map_words += static_cast<std::uint64_t>(cnt);
            for (int i = 0; i < cnt; ++i) {
                int a20[kSuperlanes];
                for (int t = 0; t < kSuperlanes; ++t) a20[t] = gl[static_cast<std::size_t>(i * kSuperlanes + t)];
                sim_.mem(ms[0])[static_cast<std::size_t>(mbase + i)] = map_word(a20);
            }
            ctx.read(ms[0], mbank);
            POp mo = op_read(ms[0], mbase, cnt, m);
            mo.kind = 'M';
            mo.layer = layer;
            mo.label = tag + " map";
            const int mid = p.add_op(mo);
            const int em = p.add_edge(1);
            p.produce(mid, em, 0, 1);
            POp go;
            go.queue = q_mem_read(g);
            go.ins.in.opcode = Opcode::MemGather;
            go.mk = MemKind::Gather;
            go.mem_g = g;
            go.gather = std::move(gl);
            go.n = cnt;
            go.m0 = m;
            go.kind = 'G';
            go.layer = layer;
            go.label = tag + " gather";
            const int gid = p.add_op(go);
            p.consume(gid, 0, em, 0, 1);
            p.produce(gid, act_edge, 0, 1);
        }
        m = mb;
    }
}

int Compiler::pick_planes(int count, std::vector<int>* planes) {
    std::vector<int> order(kPlanes);
    for (int i = 0; i < kPlanes; ++i) order[static_cast<std::size_t>(i)] = kPlaneOrder[i];
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return planes_[static_cast<std::size_t>(a)].abc_end < planes_[static_cast<std::size_t>(b)].abc_end;
    });
    planes->assign(order.begin(), order.begin() + count);
    return count;
}

void Compiler::note(const Pipeline& p, int li) {
    LayerStats& st = stats[static_cast<std::size_t>(li)];
    for (const POp& o : p.ops) {
        st.first = std::min(st.first, o.start);
        st.last = std::max(st.last, o.start + o.n + o.ins.in.d_func);
        if (o.kind == 'A') st.abc_vectors += static_cast<std::uint64_t>(o.n);
    }
}

// ---------------------------------------------------------------------------------------------
// Conv / FC
// ---------------------------------------------------------------------------------------------
void Compiler::lower_conv(int li) {
    const LayerDesc& L = net_.layers[static_cast<std::size_t>(li)];
    const Plan& P = plans_[static_cast<std::size_t>(li)];
    TState& Tin = ts_[static_cast<std::size_t>(L.in)];
    TState& Tout = ts_[static_cast<std::size_t>(L.out)];
    const TensorDesc& din = net_.tensors[static_cast<std::size_t>(L.in)];
    const TensorDesc& dout = net_.tensors[static_cast<std::size_t>(L.out)];
    const int nk = static_cast<int>(P.kv.size());
    {
        std::vector<int> ex;
        const int bank = out_bank(L, &ex);
        alloc_tensor(L.out, ex, bank);
    }
    LayerStats& st = stats[static_cast<std::size_t>(li)];
    st.rounds = static_cast<int>(P.rounds.size());
    st.kvecs = nk;
    st.nwords = static_cast<int>(Tout.lay.words.size());
    st.chunks = static_cast<int>(P.chunks.size());
    st.tiles = nk * st.nwords;

    // partial-sum buffers (int32 as 4 byte planes), ping-pong banks
    struct PB { std::array<Store, 2> s; int done = 0; };
    std::map<std::pair<int, int>, PB> pbuf;
    std::vector<int> avoid;
    for (auto& ps : Tin.store) for (auto& s : ps) avoid.insert(avoid.end(), s.slices.begin(), s.slices.end());
    for (auto& ps : Tout.store) for (auto& s : ps) avoid.insert(avoid.end(), s.slices.begin(), s.slices.end());
    if (L.res >= 0)
        for (auto& ps : ts_[static_cast<std::size_t>(L.res)].store) for (auto& s : ps) avoid.insert(avoid.end(), s.slices.begin(), s.slices.end());

    for (std::size_t r = 0; r < P.rounds.size(); ++r) {
        const auto& slots = P.rounds[r];
        std::vector<int> planes;
        pick_planes(static_cast<int>(slots.size()), &planes);
        std::vector<std::int64_t> lw_end(slots.size(), -1);
        std::vector<char> need_iw(slots.size(), 0);
        // 1. LW pipelines (weights for this round go into the LW buffers)
        for (std::size_t k = 0; k < slots.size(); ++k) {
            const int pl = planes[k];
            const long tile = (static_cast<long>(li) * 4096 + slots[k].v) * 64 + slots[k].j;
            if (planes_[static_cast<std::size_t>(pl)].tile == tile) continue;
            const Store ws = weight_tile(li, slots[k].v, slots[k].j, Floorplan::plane_west(pl));
            Pipeline lw;
            lw.name = L.name + " LW";
            lw.layer = li;
            const int e16 = lw.add_edge(16);
            for (int s = 0; s < 16; ++s) {
                POp o = op_read(ws.slices[static_cast<std::size_t>(s)], ws.base, kSuperlanes, 0);
                o.layer = li;
                o.kind = 'R';
                o.label = L.name + " weights";
                const int id = lw.add_op(o);
                lw.produce(id, e16, s, 1);
            }
            POp o;
            o.queue = q_mxm_lw(pl);
            o.ins.in.opcode = Opcode::MxmLoadWeights;
            o.ins.in.unit = static_cast<std::uint8_t>(pl);
            o.n = kSuperlanes;
            o.kind = 'L';
            o.layer = li;
            o.label = L.name + " LW";
            o.min_issue = planes_[static_cast<std::size_t>(pl)].last_iw + 1;
            const int lid = lw.add_op(o);
            lw.consume(lid, 0, e16, 0, 16);
            sch_.place(lw);
            note(lw, li);
            lw_end[k] = lw.ops[static_cast<std::size_t>(lid)].start + kSuperlanes;
            planes_[static_cast<std::size_t>(pl)].tile = tile;
            need_iw[k] = 1;
        }
        // 2. the round pipeline
        Pipeline p;
        p.name = L.name + " round " + std::to_string(r);
        p.layer = li;
        PipeCtx ctx;
        ctx.single = opt_.mem_single_issue;
        // output writes first so reads can avoid their banks
        std::map<std::pair<int, int>, std::vector<int>> out_slots;   // (j,chunk) -> slot idx
        for (std::size_t k = 0; k < slots.size(); ++k) out_slots[{slots[k].j, slots[k].chunk}].push_back(static_cast<int>(k));
        // Pre-pass: register every partial-sum, output and residual access of this round first,
        // so gather maps and replicas picked below never land on a port this pipeline uses.
        std::map<std::pair<int, int>, int> res_src;
        for (auto& [key, ks] : out_slots) {
            const int j = key.first, chunk = key.second;
            const auto ch = P.chunks[static_cast<std::size_t>(chunk)];
            const int n = (ch.second - ch.first) * dout.W;
            int minv = INT32_MAX, maxv = -1;
            for (int k : ks) { minv = std::min(minv, slots[static_cast<std::size_t>(k)].v); maxv = std::max(maxv, slots[static_cast<std::size_t>(k)].v); }
            const bool partial_in = minv > 0, final = maxv == nk - 1;
            if (partial_in || !final) {
                auto it = pbuf.find(key);
                if (it == pbuf.end()) {
                    PB nb;
                    const bool west = ((j + chunk) % 2) == 0;
                    for (int b = 0; b < 2; ++b) {
                        nb.s[static_cast<std::size_t>(b)].size = n;
                        nb.s[static_cast<std::size_t>(b)].bank = b;
                        // near the VXM: the partial round trip (VXM -> MEM -> VXM) sets how soon
                        // the next round of this output can start
                        alloc_what_ = L.name + " psum j" + std::to_string(j) + " c" + std::to_string(chunk) + (b ? " pong" : " ping");
                        alloc_kind_ = 'P';
                        nb.s[static_cast<std::size_t>(b)].slices = pick_slices(west, 4, n, b, avoid, opt_.act_index_lo, opt_.act_index_lo + 11, act_use_, &nb.s[static_cast<std::size_t>(b)].base);
                        // single-issue MEM IQs: the ping and pong buffers need different slices
                        if (opt_.mem_single_issue) avoid.insert(avoid.end(), nb.s[static_cast<std::size_t>(b)].slices.begin(), nb.s[static_cast<std::size_t>(b)].slices.end());
                    }
                    for (auto& s2 : nb.s) avoid.insert(avoid.end(), s2.slices.begin(), s2.slices.end());
                    it = pbuf.emplace(key, nb).first;
                }
                PB& pb = it->second;
                if (partial_in) for (int g : pb.s[static_cast<std::size_t>((pb.done - 1) % 2)].slices) ctx.read(g, pb.s[static_cast<std::size_t>((pb.done - 1) % 2)].bank);
                if (!final) for (int g : pb.s[static_cast<std::size_t>(pb.done % 2)].slices) ctx.write(g, pb.s[static_cast<std::size_t>(pb.done % 2)].bank);
            }
            if (final) {
                const Store& os = Tout.store[static_cast<std::size_t>(chunk)][static_cast<std::size_t>(j)];
                for (int g : os.slices) ctx.write(g, os.bank);
                if (L.res >= 0 && !L.raw32) {
                    const TState& Tr = ts_[static_cast<std::size_t>(L.res)];
                    std::vector<int> yy(static_cast<std::size_t>(n)), xx(static_cast<std::size_t>(n * kSuperlanes), -1);
                    for (int m = 0; m < n; ++m) {
                        yy[static_cast<std::size_t>(m)] = ch.first + m / dout.W;
                        xx[static_cast<std::size_t>(m * kSuperlanes)] = m % dout.W;
                    }
                    const int er = p.add_edge(1);
                    add_act_source(p, ctx, er, L.res, j, Tr.R - 1, n, yy, xx, true, li, L.name + " residual");
                    res_src[key] = er;
                }
            }
        }
        std::vector<int> res_edge(slots.size(), -1), abc_op(slots.size(), -1), iw_op(slots.size(), -1);
        for (std::size_t k = 0; k < slots.size(); ++k) {
            const Slot& s = slots[k];
            const int pl = planes[k];
            const auto ch = P.chunks[static_cast<std::size_t>(s.chunk)];
            const int n = (ch.second - ch.first) * dout.W;
            const KVec& kv = P.kv[static_cast<std::size_t>(s.v)];
            const int word = kv.items[0].word;
            const int Lw = Tin.lay.words[static_cast<std::size_t>(word)].sl();
            std::vector<int> yy(static_cast<std::size_t>(n)), xx(static_cast<std::size_t>(n * kSuperlanes), -1);
            for (int m = 0; m < n; ++m) {
                const int y = ch.first + m / dout.W, x = m % dout.W;
                yy[static_cast<std::size_t>(m)] = y * L.stride + kv.items[0].dy;
                for (const KItem& it : kv.items)
                    for (int t = it.sl; t < it.sl + Lw && t < kSuperlanes; ++t)
                        xx[static_cast<std::size_t>(m * kSuperlanes + t)] = x * L.stride + it.dx;
                if (kv.identity) xx[static_cast<std::size_t>(m * kSuperlanes)] = x;
            }
            const int ea = p.add_edge(1);
            add_act_source(p, ctx, ea, L.in, word, static_cast<int>(k), n, yy, xx, kv.identity, li,
                           L.name + " v" + std::to_string(s.v));
            POp a;
            a.queue = q_mxm_abc(pl);
            a.ins.in.opcode = Opcode::MxmActivate;
            a.ins.in.unit = static_cast<std::uint8_t>(pl);
            a.n = n;
            a.kind = 'A';
            a.layer = li;
            a.label = L.name + " ABC v" + std::to_string(s.v) + " j" + std::to_string(s.j);
            a.min_issue = planes_[static_cast<std::size_t>(pl)].abc_end;
            abc_op[k] = p.add_op(a);
            p.consume(abc_op[k], 0, ea, 0, 1);
            res_edge[k] = p.add_edge(4);
            p.produce(abc_op[k], res_edge[k], 0, 4);
            if (need_iw[k]) {
                POp iw;
                iw.queue = q_mxm_iw(pl);   // own IQ [Ours]
                iw.ins.in.opcode = Opcode::MxmInstallWeights;
                iw.ins.in.from_buffer = true;
                iw.ins.in.unit = static_cast<std::uint8_t>(pl);
                iw.n = 1;
                iw.kind = 'I';
                iw.layer = li;
                iw.label = L.name + " IW";
                iw.min_issue = std::max(lw_end[k], planes_[static_cast<std::size_t>(pl)].abc_end);
                iw_op[k] = p.add_op(iw);
                p.links.push_back({abc_op[k], iw_op[k], -1});
            }
        }
        for (auto& [key, ks] : out_slots) {
            const int j = key.first, chunk = key.second;
            const auto ch = P.chunks[static_cast<std::size_t>(chunk)];
            const int n = (ch.second - ch.first) * dout.W;
            int minv = INT32_MAX, maxv = -1;
            for (int k : ks) { minv = std::min(minv, slots[static_cast<std::size_t>(k)].v); maxv = std::max(maxv, slots[static_cast<std::size_t>(k)].v); }
            const bool partial_in = minv > 0, final = maxv == nk - 1;
            PB* pb = (partial_in || !final) ? &pbuf.at(key) : nullptr;
            // sum the results of this output's slots
            int sum = res_edge[static_cast<std::size_t>(ks[0])];
            for (std::size_t q = 1; q < ks.size(); ++q)
                sum = vxm(p, Opcode::VxmAdd, n, sum, 4, res_edge[static_cast<std::size_t>(ks[q])], 4, 4, 0, li, L.name + " sum");
            if (partial_in) {
                const Store& in = pb->s[static_cast<std::size_t>((pb->done - 1) % 2)];
                const int ep = p.add_edge(4);
                for (int b = 0; b < 4; ++b) {
                    POp o = op_read(in.slices[static_cast<std::size_t>(b)], in.base, n, 0);
                    o.layer = li;
                    o.label = L.name + " partial read";
                    const int id = p.add_op(o);
                    p.produce(id, ep, b, 1);
                    ctx.read(in.slices[static_cast<std::size_t>(b)], in.bank);
                }
                sum = vxm(p, Opcode::VxmAdd, n, sum, 4, ep, 4, 4, 0, li, L.name + " +partial");
            }
            if (!final) {
                const Store& out = pb->s[static_cast<std::size_t>(pb->done % 2)];
                for (int b = 0; b < 4; ++b) {
                    POp o = op_write(out.slices[static_cast<std::size_t>(b)], out.base, n, 0);
                    o.layer = li;
                    o.label = L.name + " partial write";
                    const int id = p.add_op(o);
                    p.consume(id, 0, sum, b, 1);
                    ctx.write(out.slices[static_cast<std::size_t>(b)], out.bank);
                }
                pb->done++;
                continue;
            }
            if (pb) pb->done++;
            const Store& os = Tout.store[static_cast<std::size_t>(chunk)][static_cast<std::size_t>(j)];
            if (L.raw32) {
                for (int b = 0; b < 4; ++b) {
                    POp o = op_write(os.slices[static_cast<std::size_t>(b)], os.base, n, 0);
                    o.layer = li;
                    o.label = L.name + " write int32";
                    const int id = p.add_op(o);
                    p.consume(id, 0, sum, b, 1);
                    ctx.write(os.slices[static_cast<std::size_t>(b)], os.bank);
                }
                continue;
            }
            int acc = vxm(p, Opcode::VxmMul, n, sum, 4, -1, 0, 4, L.Ma, li, L.name + " requant mul");
            if (L.res >= 0) {
                const int er = res_src.at(key);
                const int rm = vxm(p, Opcode::VxmMul, n, er, 1, -1, 0, 4, L.Mr, li, L.name + " residual mul");
                acc = vxm(p, Opcode::VxmAdd, n, acc, 4, rm, 4, 4, 0, li, L.name + " residual add");
            }
            int e8 = vxm(p, Opcode::VxmConvert, n, acc, 4, -1, 0, 1, L.S, li, L.name + " requant shift");
            if (L.relu) e8 = vxm(p, Opcode::VxmRelu, n, e8, 1, -1, 0, 1, 0, li, L.name + " relu");
            for (int g : os.slices) {
                POp o = op_write(g, os.base, n, 0);
                o.layer = li;
                o.label = L.name + " write";
                const int id = p.add_op(o);
                p.consume(id, 0, e8, 0, 1);
                ctx.write(g, os.bank);
            }
        }
        sch_.place(p);
        note(p, li);
        for (std::size_t k = 0; k < slots.size(); ++k) {
            PlaneState& ps = planes_[static_cast<std::size_t>(planes[k])];
            const POp& a = p.ops[static_cast<std::size_t>(abc_op[k])];
            ps.abc_end = a.start + a.n;
            if (iw_op[k] >= 0) ps.last_iw = p.ops[static_cast<std::size_t>(iw_op[k])].start;
        }
    }
    for (auto& [k, pb] : pbuf)
        for (auto& s : pb.s) free_store(s);
    st.macs = static_cast<std::uint64_t>(dout.H) * dout.W * L.cout * L.cin * L.k * L.k;
    (void)din;
}

// ---------------------------------------------------------------------------------------------
// Max pool 3x3 stride 2 (VXM max over 9 gathered pixels)
// ---------------------------------------------------------------------------------------------
void Compiler::lower_maxpool(int li) {
    const LayerDesc& L = net_.layers[static_cast<std::size_t>(li)];
    const Plan& P = plans_[static_cast<std::size_t>(li)];
    TState& Tout = ts_[static_cast<std::size_t>(L.out)];
    const TensorDesc& dout = net_.tensors[static_cast<std::size_t>(L.out)];
    {
        std::vector<int> ex;
        const int bank = out_bank(L, &ex);
        alloc_tensor(L.out, ex, bank);
    }
    for (std::size_t r = 0; r < P.rounds.size(); ++r) {
        // each 9-input max chain is its own pipeline; the scheduler overlaps them
        for (const Slot& s : P.rounds[r]) {
            Pipeline p;
            p.name = L.name + " chunk " + std::to_string(s.chunk);
            p.layer = li;
            PipeCtx ctx;
            ctx.single = opt_.mem_single_issue;
            const auto ch = P.chunks[static_cast<std::size_t>(s.chunk)];
            const int n = (ch.second - ch.first) * dout.W;
            for (int g : Tout.store[static_cast<std::size_t>(s.chunk)][0].slices) ctx.write(g, Tout.store[static_cast<std::size_t>(s.chunk)][0].bank);
            std::vector<int> taps;
            int o = 0;
            for (int dy = -L.pad; dy < L.k - L.pad; ++dy)
                for (int dx = -L.pad; dx < L.k - L.pad; ++dx, ++o) {
                    std::vector<int> yy(static_cast<std::size_t>(n)), xx(static_cast<std::size_t>(n * kSuperlanes));
                    for (int m = 0; m < n; ++m) {
                        const int y = ch.first + m / dout.W, x = m % dout.W;
                        yy[static_cast<std::size_t>(m)] = y * L.stride + dy;
                        for (int t = 0; t < kSuperlanes; ++t) xx[static_cast<std::size_t>(m * kSuperlanes + t)] = x * L.stride + dx;
                    }
                    const int e = p.add_edge(1);
                    add_act_source(p, ctx, e, L.in, 0, o, n, yy, xx, false, li, L.name + " tap" + std::to_string(o));
                    taps.push_back(e);
                }
            while (taps.size() > 1) {
                std::vector<int> nx;
                for (std::size_t i = 0; i + 1 < taps.size(); i += 2)
                    nx.push_back(vxm(p, Opcode::VxmMax, n, taps[i], 1, taps[i + 1], 1, 1, 0, li, L.name + " max"));
                if (taps.size() % 2) nx.push_back(taps.back());
                taps = nx;
            }
            const Store& os = Tout.store[static_cast<std::size_t>(s.chunk)][0];
            for (int g : os.slices) {
                POp w = op_write(g, os.base, n, 0);
                w.layer = li;
                w.label = L.name + " write";
                const int id = p.add_op(w);
                p.consume(id, 0, taps[0], 0, 1);
                ctx.write(g, os.bank);
            }
            sch_.place(p);
            note(p, li);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Global average pool: pairwise int32 add tree in the VXM, then requantize (divide folded in Ma)
// ---------------------------------------------------------------------------------------------
void Compiler::lower_avgpool(int li) {
    const LayerDesc& L = net_.layers[static_cast<std::size_t>(li)];
    const TState& Tin = ts_[static_cast<std::size_t>(L.in)];
    TState& Tout = ts_[static_cast<std::size_t>(L.out)];
    const TensorDesc& din = net_.tensors[static_cast<std::size_t>(L.in)];
    {
        std::vector<int> ex;
        const int bank = out_bank(L, &ex);
        alloc_tensor(L.out, ex, bank);
    }
    const int npix = din.H * din.W;
    for (std::size_t w = 0; w < Tin.lay.words.size(); ++w) {
        const Store& src = Tin.store[0][w];
        // level state: entries [0,h) in X, [h,n) in Y (int32 as 4 byte planes), or int8 source
        int n = npix;
        bool first = true;
        Store X, Y;
        std::vector<Store> to_free;
        while (n > 1) {
            const int h = (n + 1) / 2;
            const int nb = n - h;
            const int nout = h;
            Pipeline p;
            p.name = L.name + " w" + std::to_string(w) + " n" + std::to_string(n);
            p.layer = li;
            PipeCtx ctx;
            ctx.single = opt_.mem_single_issue;
            const int ea = p.add_edge(first ? 1 : 4), eb = p.add_edge(first ? 1 : 4);
            auto add_reads = [&](int e, const Store& s, int off, int cnt, int total, int rep) {
                const int width = first ? 1 : 4;
                for (int b = 0; b < width; ++b) {
                    const int g = first ? s.slices[static_cast<std::size_t>(rep)] : s.slices[static_cast<std::size_t>(b)];
                    if (cnt > 0) {
                        POp o = op_read(g, s.base + off, cnt, 0);
                        o.layer = li;
                        o.label = L.name + " read";
                        const int id = p.add_op(o);
                        p.produce(id, e, b, 1);
                    }
                    if (cnt < total) {
                        POp z = op_read(g, s.bank * kBankWords, total - cnt, cnt, 0);
                        z.layer = li;
                        z.label = L.name + " zero pad";
                        const int id = p.add_op(z);
                        p.produce(id, e, b, 1);
                    }
                    ctx.read(g, s.bank);
                }
            };
            if (first) {
                add_reads(ea, src, 0, h, h, 0);
                add_reads(eb, src, h, nb, h, 1);
            } else {
                add_reads(ea, X, 0, h, h, 0);
                add_reads(eb, Y, 0, nb, h, 0);
            }
            const int sum = vxm(p, Opcode::VxmAdd, h, ea, first ? 1 : 4, eb, first ? 1 : 4, 4, 0, li, L.name + " add");
            if (nout == 1) {
                const int a = vxm(p, Opcode::VxmMul, 1, sum, 4, -1, 0, 4, L.Ma, li, L.name + " scale");
                const int e8 = vxm(p, Opcode::VxmConvert, 1, a, 4, -1, 0, 1, L.S, li, L.name + " shift");
                const Store& os = Tout.store[0][w];
                for (int g : os.slices) {
                    POp o = op_write(g, os.base, 1, 0);
                    o.layer = li;
                    o.label = L.name + " write";
                    const int id = p.add_op(o);
                    p.consume(id, 0, e8, 0, 1);
                }
            } else {
                const int h2 = (nout + 1) / 2;
                Store NX, NY;
                std::vector<int> avoid = ctx.rd;
                for (auto* S2 : {&NX, &NY}) {
                    const int sz = S2 == &NX ? h2 : nout - h2;
                    S2->size = std::max(1, sz);
                    S2->bank = (static_cast<int>(w) + (first ? 0 : 1) + n) % 2;
                    alloc_what_ = L.name + " partial w" + std::to_string(w) + " n" + std::to_string(nout);
                    alloc_kind_ = 'P';
                    S2->slices = pick_slices((w % 2) == 0, 4, S2->size, S2->bank, avoid, opt_.act_index_lo, opt_.act_index_hi, act_use_, &S2->base);
                    avoid.insert(avoid.end(), S2->slices.begin(), S2->slices.end());
                }
                for (int b = 0; b < 4; ++b) {
                    POp o = op_write(NX.slices[static_cast<std::size_t>(b)], NX.base, h2, 0);
                    o.layer = li;
                    o.label = L.name + " write partial";
                    int id = p.add_op(o);
                    p.consume(id, 0, sum, b, 1);
                    if (nout - h2 > 0) {
                        POp o2 = op_write(NY.slices[static_cast<std::size_t>(b)], NY.base - h2, nout - h2, h2);
                        o2.addr0 = NY.base;   // vector m (>= h2) -> NY.base + (m - h2)
                        o2.layer = li;
                        o2.label = L.name + " write partial";
                        id = p.add_op(o2);
                        p.consume(id, 0, sum, b, 1);
                    }
                }
                if (!first) { to_free.push_back(X); to_free.push_back(Y); }
                X = NX;
                Y = NY;
            }
            sch_.place(p);
            note(p, li);
            first = false;
            n = nout;
        }
        for (auto& s : to_free) free_store(s);
        if (!first && npix > 2) { free_store(X); free_store(Y); }
    }
    stats[static_cast<std::size_t>(li)].macs = 0;
}

// ---------------------------------------------------------------------------------------------
void Compiler::compile() {
    stats.assign(net_.layers.size(), LayerStats{});
    for (std::size_t li = 0; li < net_.layers.size(); ++li) {
        const LayerDesc& L = net_.layers[li];
        if (opt_.no_layer_overlap) sch_.floor = sch_.makespan();
        stats[li].name = L.name;
        stats[li].index = static_cast<int>(li);
        if (!ts_[static_cast<std::size_t>(L.in)].allocated)
            throw std::runtime_error("layer " + L.name + ": input tensor not produced yet");
        if (L.kind == LayerDesc::Conv) lower_conv(static_cast<int>(li));
        else if (L.kind == LayerDesc::MaxPool) lower_maxpool(static_cast<int>(li));
        else lower_avgpool(static_cast<int>(li));
        if (--ts_[static_cast<std::size_t>(L.in)].consumers_left == 0) free_tensor(L.in);
        if (L.res >= 0 && --ts_[static_cast<std::size_t>(L.res)].consumers_left == 0) free_tensor(L.res);
        if (opt_.verbose) {
            const LayerStats& s = stats[li];
            std::fprintf(stderr, "  %-14s rounds %3d tiles %3d chunks %d  cycles %6lld..%6lld\n", L.name.c_str(), s.rounds,
                         s.tiles, s.chunks, static_cast<long long>(s.first), static_cast<long long>(s.last));
        }
    }
    sch_.floor = 0;   // Ifetch pipelines are planned afterwards, in the gaps
}

std::vector<std::int8_t> Compiler::read_tensor(int t, Simulator& s) const {
    const TState& T = ts_[static_cast<std::size_t>(t)];
    const TensorDesc& d = net_.tensors[static_cast<std::size_t>(t)];
    std::vector<std::int8_t> out(static_cast<std::size_t>(d.C * d.H * d.W));
    for (int c = 0; c < d.C; ++c) {
        const int w = c / kLanes;
        const Layout::Word& lw = T.lay.words[static_cast<std::size_t>(w)];
        const int lane = lw.copies[0] * 16 + (c - lw.ch0);
        for (int y = 0; y < d.H; ++y) {
            const int p = piece_of_row(T, y);
            const Store& st = T.store[static_cast<std::size_t>(p)][static_cast<std::size_t>(w)];
            for (int x = 0; x < d.W; ++x) {
                const int a = st.base + (y - T.pieces[static_cast<std::size_t>(p)].first) * d.W + x;
                out[static_cast<std::size_t>((c * d.H + y) * d.W + x)] =
                    static_cast<std::int8_t>(s.mem(st.slices[0])[static_cast<std::size_t>(a)][static_cast<std::size_t>(lane)]);
            }
        }
    }
    return out;
}

std::vector<std::int32_t> Compiler::read_logits(Simulator& s) const {
    const TState& T = ts_[static_cast<std::size_t>(net_.output)];
    const TensorDesc& d = net_.tensors[static_cast<std::size_t>(net_.output)];
    std::vector<std::int32_t> out(static_cast<std::size_t>(d.C));
    for (int f = 0; f < d.C; ++f) {
        const Store& st = T.store[0][static_cast<std::size_t>(f / kLanes)];
        std::uint32_t u = 0;
        for (int b = 0; b < 4; ++b)
            u |= static_cast<std::uint32_t>(s.mem(st.slices[static_cast<std::size_t>(b)])[static_cast<std::size_t>(st.base)][static_cast<std::size_t>(f % kLanes)]) << (8 * b);
        out[static_cast<std::size_t>(f)] = static_cast<std::int32_t>(u);
    }
    return out;
}

}  // namespace comp
}  // namespace lpu
