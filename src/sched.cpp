// sched.cpp - see sched.h.
#include "sched.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>

namespace lpu {
namespace sched {

void Pipeline::produce(int op, int edge, int sub, int width) {
    POp& o = ops[static_cast<std::size_t>(op)];
    o.out_edge = edge;
    o.out_sub = sub;
    o.out_width = width;
    edges[static_cast<std::size_t>(edge)].producers.push_back(op);
}

void Pipeline::consume(int op, int k, int edge, int sub, int width) {
    POp& o = ops[static_cast<std::size_t>(op)];
    const std::size_t kk = static_cast<std::size_t>(k);
    if (o.in_edge.size() <= kk) {
        o.in_edge.resize(kk + 1, -1);
        o.in_sub.resize(kk + 1, 0);
        o.in_width.resize(kk + 1, 0);
    }
    o.in_edge[kk] = edge;
    o.in_sub[kk] = sub;
    o.in_width[kk] = width;
    edges[static_cast<std::size_t>(edge)].consumers.push_back(op);
}

Scheduler::Scheduler(const Floorplan& fp, const LatencyTable& lat)
    : fp_(fp),
      lat_(lat),
      queue_busy_(sim::kNumChannels),
      bank_read_(2 * sim::kMemSlices),
      bank_write_(2 * sim::kMemSlices),
      ready_(static_cast<std::size_t>(sim::kMemSlices) * sim::kMemWords, 0),
      last_read_(static_cast<std::size_t>(sim::kMemSlices) * sim::kMemWords, -1) {}

namespace {
int op_bank(const POp& o) {
    if (o.mk == MemKind::Gather) {
        for (std::uint16_t a : o.gather)
            if (a % sim::kBankWords != 0) return a / sim::kBankWords;
        return o.gather.empty() ? 0 : o.gather[0] / sim::kBankWords;
    }
    return o.addr0 / sim::kBankWords;
}
}  // namespace

bool Scheduler::overlaps(const Intervals& iv, std::int64_t a, std::int64_t b, std::int64_t* end) {
    auto it = iv.lower_bound(a);
    if (it != iv.begin()) {
        auto pr = std::prev(it);
        if (pr->second > a) { *end = pr->second; return true; }
    }
    if (it != iv.end() && it->first < b) {
        // find the latest end among the overlapping run for a bigger safe jump
        std::int64_t e = it->second;
        *end = e;
        return true;
    }
    return false;
}

void Scheduler::insert(Intervals& iv, std::int64_t a, std::int64_t b) { iv[a] = b; }

static std::int64_t bucket_key(int dir, int stream, std::int64_t b) {
    return ((static_cast<std::int64_t>(dir) * 32 + stream) << 40) ^ (b + (1LL << 39));
}

bool Scheduler::rect_free(int dir, int stream, const Rect& r, std::int64_t* need_d) const {
    bool free = true;
    for (std::int64_t b = r.d0 >> 6; b <= (r.d1 - 1) >> 6; ++b) {
        auto it = rect_buckets_.find(bucket_key(dir, stream, b));
        if (it == rect_buckets_.end()) continue;
        for (int id : it->second) {
            const Rect& o = rects_[static_cast<std::size_t>(id)];
            if (o.d0 < r.d1 && r.d0 < o.d1 && o.p0 <= r.p1 && r.p0 <= o.p1) {
                free = false;
                *need_d = std::max(*need_d, o.d1 - r.d0);
            }
        }
    }
    return free;
}

void Scheduler::rect_add(int dir, int stream, const Rect& r) {
    const int id = static_cast<int>(rects_.size());
    rects_.push_back(r);
    for (std::int64_t b = r.d0 >> 6; b <= (r.d1 - 1) >> 6; ++b)
        rect_buckets_[bucket_key(dir, stream, b)].push_back(id);
}

void Scheduler::solve(Pipeline& p) const {
    const int n = static_cast<int>(p.ops.size());
    for (POp& o : p.ops) o.pos = o.queue >= 0 ? fp_.queue_pos(o.queue) : fp_.vxm_pos;
    std::vector<std::vector<std::pair<int, int>>> adj(static_cast<std::size_t>(n));
    for (const Edge& e : p.edges) {
        for (int a : e.producers)
            for (int c : e.consumers) {
                const POp& A = p.ops[static_cast<std::size_t>(a)];
                const POp& C = p.ops[static_cast<std::size_t>(c)];
                const int delta = dfunc(A) + std::abs(C.pos - A.pos) - dskew(C);
                adj[static_cast<std::size_t>(a)].push_back({c, delta});
                adj[static_cast<std::size_t>(c)].push_back({a, -delta});
            }
    }
    for (const Pipeline::Link& l : p.links) {
        const POp& A = p.ops[static_cast<std::size_t>(l.a)];
        const POp& B = p.ops[static_cast<std::size_t>(l.b)];
        const int delta = A.m0 + l.delta - B.m0;   // phase_b = phase_a + delta
        adj[static_cast<std::size_t>(l.a)].push_back({l.b, delta});
        adj[static_cast<std::size_t>(l.b)].push_back({l.a, -delta});
    }
    std::vector<char> seen(static_cast<std::size_t>(n), 0);
    for (int s = 0; s < n; ++s) {
        if (seen[static_cast<std::size_t>(s)]) continue;
        seen[static_cast<std::size_t>(s)] = 1;
        p.ops[static_cast<std::size_t>(s)].phase = 0;
        std::deque<int> dq{s};
        while (!dq.empty()) {
            const int x = dq.front();
            dq.pop_front();
            for (auto [y, d] : adj[static_cast<std::size_t>(x)]) {
                const int ph = p.ops[static_cast<std::size_t>(x)].phase + d;
                if (!seen[static_cast<std::size_t>(y)]) {
                    seen[static_cast<std::size_t>(y)] = 1;
                    p.ops[static_cast<std::size_t>(y)].phase = ph;
                    dq.push_back(y);
                } else if (p.ops[static_cast<std::size_t>(y)].phase != ph) {
                    throw std::runtime_error("pipeline " + p.name + ": operands cannot be aligned (" +
                                             p.ops[static_cast<std::size_t>(x)].label + " / " +
                                             p.ops[static_cast<std::size_t>(y)].label + ")");
                }
            }
        }
    }
    int mn = INT32_MAX;
    for (const POp& o : p.ops) mn = std::min(mn, o.phase + o.m0);
    for (POp& o : p.ops) o.phase -= mn;
    // direction sanity
    for (const Edge& e : p.edges) {
        int sgn = 0;
        for (int a : e.producers)
            for (int c : e.consumers) {
                const int d = p.ops[static_cast<std::size_t>(c)].pos - p.ops[static_cast<std::size_t>(a)].pos;
                const int s2 = d > 0 ? 1 : (d < 0 ? -1 : 0);
                if (s2 != 0 && sgn != 0 && s2 != sgn)
                    throw std::runtime_error("pipeline " + p.name + ": edge consumers on both sides of a producer");
                if (s2 != 0) sgn = s2;
            }
    }
}

void Scheduler::edge_rects(const Pipeline& p, int e, Direction dir, std::int64_t S,
                           std::vector<std::pair<int, Rect>>* out) const {
    const Edge& E = p.edges[static_cast<std::size_t>(e)];
    int cmin = INT32_MAX, cmax = INT32_MIN;
    for (int c : E.consumers) {
        cmin = std::min(cmin, p.ops[static_cast<std::size_t>(c)].pos);
        cmax = std::max(cmax, p.ops[static_cast<std::size_t>(c)].pos);
    }
    for (int a : E.producers) {
        const POp& A = p.ops[static_cast<std::size_t>(a)];
        const std::int64_t t0 = S + A.phase + A.m0 + dfunc(A);   // = write cycle + 1 of vector m0
        Rect r;
        if (dir == Direction::East) {
            r.d0 = t0 - A.pos;
            r.p0 = A.pos;
            r.p1 = std::max(A.pos, cmax);
        } else {
            r.d0 = t0 + A.pos;
            r.p0 = std::min(A.pos, cmin);
            r.p1 = A.pos;
        }
        r.d1 = r.d0 + A.n;
        for (int s = 0; s < A.out_width; ++s) out->push_back({A.out_sub + s, r});
    }
}

std::int64_t Scheduler::mem_lower_bound(const Pipeline& p) const {
    std::int64_t lb = INT64_MIN;
    for (const POp& o : p.ops) {
        if (o.mk == MemKind::None) continue;
        const std::int64_t base = o.phase + o.m0 + dskew(o);
        for (int i = 0; i < o.n; ++i) {
            if (o.mk == MemKind::Gather) {
                int last = -1;
                for (int t = 0; t < kSuperlanes; ++t) {
                    const int a = o.gather[static_cast<std::size_t>(i * kSuperlanes + t)];
                    if (a == last || a % sim::kBankWords == 0) continue;
                    last = a;
                    lb = std::max(lb, ready_[idx(o.mem_g, a)] - base - i);
                }
            } else {
                const int a = o.addr0 + i * o.stride;
                if (a % sim::kBankWords == 0) continue;   // zero words are never written
                const std::size_t w = idx(o.mem_g, a);
                if (o.mk == MemKind::ReadSeq) {
                    lb = std::max(lb, ready_[w] - base - i);
                } else {
                    lb = std::max(lb, std::max(ready_[w], last_read_[w]) - base - i);
                }
            }
        }
    }
    return lb;
}

void Scheduler::commit_mem(const Pipeline& p, std::int64_t S) {
    for (const POp& o : p.ops) {
        if (o.mk == MemKind::None) continue;
        const std::int64_t t0 = S + o.phase + o.m0 + dskew(o);
        for (int i = 0; i < o.n; ++i) {
            const std::int64_t t = t0 + i;
            if (o.mk == MemKind::Gather) {
                for (int k = 0; k < kSuperlanes; ++k) {
                    const int a = o.gather[static_cast<std::size_t>(i * kSuperlanes + k)];
                    if (a % sim::kBankWords == 0) continue;
                    std::int64_t& lr = last_read_[idx(o.mem_g, a)];
                    lr = std::max(lr, t);
                }
            } else {
                const int a = o.addr0 + i * o.stride;
                if (a % sim::kBankWords == 0) continue;
                const std::size_t w = idx(o.mem_g, a);
                if (o.mk == MemKind::ReadSeq) last_read_[w] = std::max(last_read_[w], t);
                else ready_[w] = std::max(ready_[w], t + 1);
            }
        }
    }
}

std::int64_t Scheduler::place(Pipeline& p, bool commit) {
    solve(p);
    std::int64_t S = p.min_start;
    S = std::max(S, mem_lower_bound(p));
    for (const POp& o : p.ops)
        if (o.min_issue != INT64_MIN) S = std::max(S, o.min_issue - o.phase - o.m0);
    if (floor > 0)
        for (const POp& o : p.ops) S = std::max(S, floor - o.phase - o.m0);
    S = std::max<std::int64_t>(S, 0);

    // Intra-pipeline sanity: two fixed-queue ops overlapping on one queue can never be placed.
    for (std::size_t i = 0; i < p.ops.size(); ++i)
        for (std::size_t j = i + 1; j < p.ops.size(); ++j) {
            const POp& a = p.ops[i];
            const POp& b = p.ops[j];
            if (a.queue < 0 || b.queue < 0 || res(a.queue) != res(b.queue)) continue;
            const int a0 = a.phase + a.m0, b0 = b.phase + b.m0;
            if (a0 < b0 + b.n && b0 < a0 + a.n)
                throw std::runtime_error("pipeline " + p.name + ": two ops overlap on queue " +
                                         fp_.queue_name(a.queue) + " (" + a.label + ", " + b.label + ")");
        }

    // Intra-pipeline bank rule: a read and a write of one bank of one slice may not overlap.
    for (std::size_t i = 0; i < p.ops.size(); ++i)
        for (std::size_t j = 0; j < p.ops.size(); ++j) {
            const POp& a = p.ops[i];
            const POp& b = p.ops[j];
            if (a.mk == MemKind::None || b.mk != MemKind::WriteSeq || a.mk == MemKind::WriteSeq) continue;
            if (a.mem_g != b.mem_g || op_bank(a) != op_bank(b)) continue;
            const int a0 = a.phase + a.m0, b0 = b.phase + b.m0;
            if (a0 < b0 + b.n && b0 < a0 + a.n)
                throw std::runtime_error("pipeline " + p.name + ": read and write of one bank overlap on " +
                                         fp_.queue_name(a.queue) + " (" + a.label + ", " + b.label + ")");
        }

    // Static per-edge rectangles at S = 0 (shifted by S later), per direction choice.
    const int ne = static_cast<int>(p.edges.size());
    std::vector<std::vector<Direction>> dir_opts(static_cast<std::size_t>(ne));
    std::vector<std::array<std::vector<std::pair<int, Rect>>, 2>> rects0(static_cast<std::size_t>(ne));
    for (int e = 0; e < ne; ++e) {
        const Edge& E = p.edges[static_cast<std::size_t>(e)];
        int sgn = 0;
        for (int a : E.producers)
            for (int c : E.consumers) {
                const int d = p.ops[static_cast<std::size_t>(c)].pos - p.ops[static_cast<std::size_t>(a)].pos;
                if (d != 0) sgn = d > 0 ? 1 : -1;
            }
        if (sgn >= 0) dir_opts[static_cast<std::size_t>(e)].push_back(Direction::East);
        if (sgn <= 0) dir_opts[static_cast<std::size_t>(e)].push_back(Direction::West);
        for (Direction d : dir_opts[static_cast<std::size_t>(e)])
            edge_rects(p, e, d, 0, &rects0[static_cast<std::size_t>(e)][d == Direction::East ? 0 : 1]);
    }
    auto bases_for = [](int width) {
        std::vector<int> b;
        if (width >= 16) { b = {0, 16}; }
        else if (width == 4) { for (int s = 28; s >= 0; s -= 4) b.push_back(s); }
        else if (width == 2) { for (int s = 30; s >= 0; s -= 2) b.push_back(s); }
        else { for (int s = 31; s >= 0; --s) b.push_back(s); }
        return b;
    };

    struct Tent { int dir; int stream; Rect r; };
    std::vector<int> alu(p.ops.size(), -1);
    std::vector<int> edir(static_cast<std::size_t>(ne)), ebase(static_cast<std::size_t>(ne));
    const char* dbg = std::getenv("LPU_SCHED_DEBUG");   // print why a pipeline is delayed
    const bool debug = dbg && std::strncmp(p.name.c_str(), dbg, std::strlen(dbg)) == 0;
    if (debug) std::fprintf(stderr, "[sched] %s: lower bound %lld\n", p.name.c_str(), static_cast<long long>(S));
    while (true) {
        ++steps_;
        bool ok = true;
        std::int64_t need = S + 1;
        const char* why = "";
        auto want = [&](std::int64_t s2) {
            ok = false;
            need = std::max(need, s2);
        };

        // 1. fixed issue channels (IQ ways)
        for (const POp& o : p.ops) {
            if (o.queue < 0) continue;
            const std::int64_t a = S + o.phase + o.m0;
            std::int64_t end;
            if (overlaps(queue_busy_[static_cast<std::size_t>(res(o.queue))], a, a + o.n, &end)) {
                want(end - o.phase - o.m0);
                if (debug) why = "queue";
                if (debug) std::fprintf(stderr, "   S=%lld queue %s busy until %lld\n", static_cast<long long>(S), fp_.queue_name(o.queue).c_str(), static_cast<long long>(end));
            }
        }
        // 2. MEM banks (pseudo-dual-port: read and write of one bank may not share a cycle)
        for (const POp& o : p.ops) {
            if (o.mk == MemKind::None) continue;
            const std::size_t bi = static_cast<std::size_t>(o.mem_g * 2 + op_bank(o));
            const std::int64_t a = S + o.phase + o.m0;
            std::int64_t end;
            const Intervals& other = o.mk == MemKind::WriteSeq ? bank_read_[bi] : bank_write_[bi];
            if (overlaps(other, a, a + o.n, &end)) {
                want(end - o.phase - o.m0);
                if (debug) std::fprintf(stderr, "   S=%lld bank conflict on %s\n", static_cast<long long>(S), fp_.queue_name(o.queue).c_str());
            }
        }
        // intra-pipeline bank conflicts are rigid: detect once
        // 3. VXM ALUs
        if (ok) {
            for (std::size_t i = 0; i < p.ops.size(); ++i) {
                const POp& o = p.ops[i];
                if (o.queue >= 0) continue;
                const std::int64_t a = S + o.phase + o.m0;
                std::int64_t best_end = INT64_MAX;
                alu[i] = -1;
                for (int u = 0; u < sim::kVxmAlus && alu[i] < 0; ++u) {
                    bool clash = false;
                    for (std::size_t j = 0; j < i; ++j) {
                        if (alu[j] != u) continue;
                        const POp& b = p.ops[j];
                        const std::int64_t b0 = S + b.phase + b.m0;
                        if (a < b0 + b.n && b0 < a + o.n) { clash = true; break; }
                    }
                    if (clash) continue;
                    std::int64_t end;
                    if (overlaps(queue_busy_[static_cast<std::size_t>(sim::q_vxm(u))], a, a + o.n, &end)) {
                        best_end = std::min(best_end, end);
                        continue;
                    }
                    alu[i] = u;
                }
                if (alu[i] < 0) {
                    if (best_end == INT64_MAX)
                        throw std::runtime_error("pipeline " + p.name + ": needs more than 16 VXM ALUs at once");
                    want(best_end - o.phase - o.m0);
                    if (debug) std::fprintf(stderr, "   S=%lld no free VXM ALU until %lld\n", static_cast<long long>(S), static_cast<long long>(best_end));
                    break;
                }
            }
        }
        // 4. streams
        for (int attempt = 0; ok && attempt < 2; ++attempt) {
            std::vector<Tent> tent;
            // edges with a forced direction first, then the free ones (producer and consumers at
            // one position, e.g. VXM chaining) on whichever direction is less used. If the
            // pipeline's own edges do not fit that way (fragmented stream ids), retry once with
            // the widest edges first inside each group.
            std::vector<int> order;
            for (int e = 0; e < ne; ++e) if (dir_opts[static_cast<std::size_t>(e)].size() == 1) order.push_back(e);
            const std::size_t nforced = order.size();
            for (int e = 0; e < ne; ++e) if (dir_opts[static_cast<std::size_t>(e)].size() != 1) order.push_back(e);
            if (attempt == 1) {
                auto wider = [&](int a, int b) { return p.edges[static_cast<std::size_t>(a)].width > p.edges[static_cast<std::size_t>(b)].width; };
                std::stable_sort(order.begin(), order.begin() + static_cast<long>(nforced), wider);
                std::stable_sort(order.begin() + static_cast<long>(nforced), order.end(), wider);
            }
            bool retry = false;
            for (int e : order) {
                if (!ok) break;
                const Edge& E = p.edges[static_cast<std::size_t>(e)];
                bool found = false;
                std::int64_t best_shift = INT64_MAX;
                std::vector<Direction> dirs = dir_opts[static_cast<std::size_t>(e)];
                if (dirs.size() == 2) {
                    int used[2] = {0, 0};
                    for (const Tent& t : tent) used[t.dir]++;
                    if (used[1] < used[0]) std::swap(dirs[0], dirs[1]);
                }
                for (Direction d : dirs) {
                    const int di = d == Direction::East ? 0 : 1;
                    for (int base : bases_for(E.width)) {
                        bool intra = false;
                        bool free = true;
                        std::int64_t need_d = 0;
                        for (const auto& [off, r0] : rects0[static_cast<std::size_t>(e)][static_cast<std::size_t>(di)]) {
                            Rect r = r0;
                            r.d0 += S;
                            r.d1 += S;
                            const int stream = base + off;
                            for (const Tent& t : tent)
                                if (t.dir == di && t.stream == stream && t.r.d0 < r.d1 && r.d0 < t.r.d1 &&
                                    t.r.p0 <= r.p1 && r.p0 <= t.r.p1) intra = true;
                            if (intra) break;
                            if (!rect_free(di, stream, r, &need_d)) free = false;
                        }
                        if (intra) continue;
                        if (free) {
                            found = true;
                            edir[static_cast<std::size_t>(e)] = di;
                            ebase[static_cast<std::size_t>(e)] = base;
                            for (const auto& [off, r0] : rects0[static_cast<std::size_t>(e)][static_cast<std::size_t>(di)]) {
                                Rect r = r0;
                                r.d0 += S;
                                r.d1 += S;
                                tent.push_back(Tent{di, base + off, r});
                            }
                            break;
                        }
                        best_shift = std::min(best_shift, need_d);
                    }
                    if (found) break;
                }
                if (!found) {
                    if (best_shift == INT64_MAX && attempt == 0) { retry = true; break; }
                    if (best_shift == INT64_MAX)
                    {
                        std::string why = "pipeline " + p.name + ": not enough streams for its own edges (edge " +
                                          std::to_string(e) + " width " + std::to_string(E.width) + ", " +
                                          std::to_string(tent.size()) + " streams already taken by this pipeline; producers";
                        for (int a : E.producers) why += " " + p.ops[static_cast<std::size_t>(a)].label + "@" + std::to_string(p.ops[static_cast<std::size_t>(a)].pos);
                        why += "; consumers";
                        for (int c : E.consumers) why += " " + p.ops[static_cast<std::size_t>(c)].label + "@" + std::to_string(p.ops[static_cast<std::size_t>(c)].pos);
                        throw std::runtime_error(why + ")");
                    }
                    want(S + std::max<std::int64_t>(1, best_shift));
                    if (debug) std::fprintf(stderr, "   S=%lld no free stream for edge %d (%s), shift %lld\n", static_cast<long long>(S), e,
                                            p.ops[static_cast<std::size_t>(E.producers[0])].label.c_str(), static_cast<long long>(best_shift));
                }
            }
            if (!retry) break;
        }
        (void)why;
        if (ok) break;
        S = need;
    }

    if (!commit) return S;
    // Commit
    const int ev_pipe = static_cast<int>(placed_);
    const std::size_t ev_base = ev_ops.size();
    for (int e = 0; e < ne; ++e) {
        Edge& E = p.edges[static_cast<std::size_t>(e)];
        E.dir = edir[static_cast<std::size_t>(e)] == 0 ? Direction::East : Direction::West;
        E.base = ebase[static_cast<std::size_t>(e)];
        E.gid = next_gid_++;
        for (const auto& [off, r0] : rects0[static_cast<std::size_t>(e)][static_cast<std::size_t>(edir[static_cast<std::size_t>(e)])]) {
            Rect r = r0;
            r.d0 += S;
            r.d1 += S;
            rect_add(edir[static_cast<std::size_t>(e)], E.base + off, r);
        }
        if (record_events) {
            // same rectangles again, but tagged with their producer (edge_rects lists them per producer)
            std::vector<std::pair<int, Rect>> rr;
            edge_rects(p, e, E.dir, S, &rr);
            std::size_t k = 0;
            for (int a : E.producers) {
                const POp& A = p.ops[static_cast<std::size_t>(a)];
                for (int s2 = 0; s2 < A.out_width; ++s2, ++k) {
                    const Rect& r = rr[k].second;
                    ev_streams.push_back(EventStream{ev_pipe, E.gid, E.dir, E.base + rr[k].first, r.d0, r.d1, r.p0, r.p1,
                                                     static_cast<int>(ev_base) + a});
                }
            }
        }
    }
    for (std::size_t i = 0; i < p.ops.size(); ++i) {
        POp& o = p.ops[i];
        o.placed_queue = o.queue >= 0 ? o.queue : sim::q_vxm(alu[i]);
        o.start = S + o.phase + o.m0;
        insert(queue_busy_[static_cast<std::size_t>(res(o.placed_queue))], o.start, o.start + o.n);
        if (o.mk != MemKind::None) {
            const std::size_t bi = static_cast<std::size_t>(o.mem_g * 2 + op_bank(o));
            insert(o.mk == MemKind::WriteSeq ? bank_write_[bi] : bank_read_[bi], o.start, o.start + o.n);
        }
        SimInstr si = o.ins;
        for (std::size_t k = 0; k < o.in_edge.size(); ++k) {
            if (o.in_edge[k] < 0) continue;
            const Edge& E = p.edges[static_cast<std::size_t>(o.in_edge[k])];
            si.in.src[k] = StreamGroup{static_cast<std::uint8_t>(E.base + o.in_sub[k]),
                                       static_cast<std::uint8_t>(o.in_width[k]), E.dir};
            si.in_edge[k] = E.gid;
            si.in_idx0[k] = static_cast<std::uint32_t>(o.m0);
        }
        if (o.out_edge >= 0) {
            const Edge& E = p.edges[static_cast<std::size_t>(o.out_edge)];
            si.in.dst = StreamGroup{static_cast<std::uint8_t>(E.base + o.out_sub),
                                    static_cast<std::uint8_t>(o.out_width), E.dir};
            si.out_edge = E.gid;
            si.out_idx0 = static_cast<std::uint32_t>(o.m0);
        }
        if (o.mk == MemKind::ReadSeq || o.mk == MemKind::WriteSeq) {
            si.in.address = static_cast<std::uint16_t>(o.addr0);
            si.addr_stride = o.stride;
        }
        si.in.d_func = static_cast<std::int16_t>(dfunc(o));
        si.in.d_skew = static_cast<std::int16_t>(dskew(o));
        si.op_id = static_cast<std::int32_t>(trace_.size());
        emitted_.push_back({o.placed_queue, si});
        emitted_time_n_.push_back({o.start, o.n});
        trace_.push_back(TraceOp{o.placed_queue, o.pos, o.start, o.n, o.kind, o.layer, o.label});
        if (record_events) {
            EventOp ev{o.start, o.placed_queue, o.pos, o.n, ev_pipe, o.layer, o.kind, si.in.opcode, si.in.src, si.in.dst,
                       -1, -1, -1, -1, 0, dfunc(o), o.label};
            if (o.mk != MemKind::None) {
                ev.mem_g = o.mem_g;
                ev.bank = op_bank(o);
                if (o.mk == MemKind::Gather) {
                    int lo = INT32_MAX, hi = -1;
                    for (std::uint16_t a : o.gather)
                        if (a % sim::kBankWords != 0) { lo = std::min(lo, int(a)); hi = std::max(hi, int(a)); }
                    if (hi >= 0) { ev.addr_lo = lo; ev.addr_hi = hi; }
                } else {
                    ev.addr_lo = std::min(o.addr0, o.addr0 + o.stride * (o.n - 1));
                    ev.addr_hi = std::max(o.addr0, o.addr0 + o.stride * (o.n - 1));
                    ev.stride = o.stride;
                }
            }
            ev_ops.push_back(std::move(ev));
        }
        makespan_ = std::max(makespan_, o.start + o.n + dfunc(o));
    }
    commit_mem(p, S);
    if (record_events) ev_pipes.push_back(EventPipe{p.name, p.layer, S});
    ++placed_;
    return S;
}

Program Scheduler::emit() const {
    Program prog;
    std::vector<std::vector<std::size_t>> per_q(sim::kNumChannels);
    for (std::size_t i = 0; i < emitted_.size(); ++i) per_q[static_cast<std::size_t>(emitted_[i].first)].push_back(i);
    for (int q = 0; q < sim::kNumChannels; ++q) {
        auto& ids = per_q[static_cast<std::size_t>(q)];
        if (ids.empty()) continue;
        std::sort(ids.begin(), ids.end(), [&](std::size_t a, std::size_t b) {
            return emitted_time_n_[a].first < emitted_time_n_[b].first;
        });
        auto& out = prog.channel(q);   // the (IQ, way) this channel maps to
        auto nop = [&](std::int64_t n) {
            while (n > 0) {
                const std::int64_t c = std::min<std::int64_t>(n, 65535);   // 16-bit N field [P]
                sim::QEntry e;
                e.kind = sim::QEntry::Nop;
                e.n = static_cast<std::uint32_t>(c);
                out.push_back(e);
                n -= c;
            }
        };
        nop(lat_.sync_cycles);   // Sync/Notify preamble
        std::int64_t t = 0;
        for (std::size_t i : ids) {
            const auto [start, n] = emitted_time_n_[i];
            if (start < t) throw std::runtime_error("emit: overlapping instructions on " + fp_.queue_name(q));
            nop(start - t);
            sim::QEntry e;
            e.kind = sim::QEntry::Ins;
            e.ins = emitted_[i].second;
            out.push_back(e);
            if (n > 1) {
                sim::QEntry r;
                r.kind = sim::QEntry::Rep;
                r.n = static_cast<std::uint32_t>(n - 1);
                r.d = 1;
                out.push_back(r);
            }
            t = start + n;
        }
    }
    return prog;
}

}  // namespace sched
}  // namespace lpu
