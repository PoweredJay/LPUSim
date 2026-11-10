// tests.cpp - the incremental steps from the scheduler plan, each one scheduled by the
// scheduler, run on the cycle simulator, and checked against a reference.
//   0  DiagStreams == lpu::StreamRegisterFile on random traffic
//   1  one Read, one Write three hops away (main.cpp, now scheduled)
//   2  streaming add (Listing 1, p.151): 1 vector and 1,024 vectors
//   3  three adds competing for the same slices (reservation table)
//   3b a hand-broken schedule is caught by the simulator
//   5+ matmul / maxpool / conv / bottleneck live in compiler tests (tests_layers.cpp)
#include <cstdio>
#include <random>
#include <string>

#include "lpusim.h"
#include "sched.h"
#include "testutil.h"

using namespace lpu;
using namespace lpu::sim;
using namespace lpu::sched;

static bool test_streams_equivalence() {
    const int P = 20;
    StreamRegisterFile ref(P);
    DiagStreams fast(P);
    std::mt19937 rng(7);
    int mismatches = 0;
    for (int cycle = 0; cycle < 400; ++cycle) {
        // reads at start of cycle
        for (int k = 0; k < 50; ++k) {
            const int p = static_cast<int>(rng() % P), s = static_cast<int>(rng() % 4);
            const Direction d = (rng() & 1) ? Direction::East : Direction::West;
            const auto& e = ref.read(p, d, s);
            const Vector* v = fast.read(p, d, s, cycle, nullptr);
            const bool same = (!e.valid && !v) || (e.valid && v && e.data == *v);
            if (!same) ++mismatches;
        }
        for (int k = 0; k < 6; ++k) {
            const int p = static_cast<int>(rng() % P), s = static_cast<int>(rng() % 4);
            const Direction d = (rng() & 1) ? Direction::East : Direction::West;
            Vector v;
            v.fill(static_cast<std::uint8_t>(rng()));
            const bool a = ref.write(p, d, s, v);
            const bool b = fast.write(p, d, s, cycle, v, Tag{});
            if (a != b) ++mismatches;
        }
        ref.end_cycle();
    }
    return check("0  DiagStreams matches StreamRegisterFile (20000 random reads)", mismatches == 0,
                 std::to_string(mismatches) + " mismatches");
}

static bool test_read_write() {
    Floorplan fp;
    LatencyTable lat;
    Scheduler sch(fp, lat);
    const int A = Floorplan::mem_global(true, 0), B = Floorplan::mem_global(true, 3);   // 3 hops
    Pipeline p;
    p.name = "read-write";
    const int r = p.add_op(make_read(A, 10, 1));
    const int w = p.add_op(make_write(B, 3, 1));
    const int e = p.add_edge(1);
    p.produce(r, e, 0, 1);
    p.consume(w, 0, e, 0, 1);
    sch.place(p);
    Simulator s(fp, lat);
    s.mem(A)[10][0] = 42;
    s.run(sch.emit());
    const std::int64_t gap = p.ops[1].start - p.ops[0].start;
    return check("1  Read -> Write 3 hops: write issues 4 cycles after read, B[3][0] = 42",
                 s.violations == 0 && s.mem(B)[3][0] == 42 && gap == 4,
                 "gap " + std::to_string(gap) + ", value " + std::to_string(s.mem(B)[3][0]) + ", violations " +
                     std::to_string(s.violations));
}

// a + b -> c for n vectors: Read a (West), Read b (East), VXM add int8 sat, Write c.
static Pipeline add_pipeline(int ga, int gb, int gc, int n, const std::string& name) {
    Pipeline p;
    p.name = name;
    const int ra = p.add_op(make_read(ga, 100, n));
    const int rb = p.add_op(make_read(gb, 100, n));
    POp add = make_vxm(Opcode::VxmAdd, n);
    add.ins.in.saturate = true;
    const int ad = p.add_op(add);
    const int wc = p.add_op(make_write(gc, 2000, n));
    const int e1 = p.add_edge(1), e2 = p.add_edge(1), e3 = p.add_edge(1);
    p.produce(ra, e1, 0, 1);
    p.produce(rb, e2, 0, 1);
    p.consume(ad, 0, e1, 0, 1);
    p.consume(ad, 1, e2, 0, 1);
    p.produce(ad, e3, 0, 1);
    p.consume(wc, 0, e3, 0, 1);
    return p;
}

static bool run_adds(int count, int n, const std::string& title) {
    Floorplan fp;
    LatencyTable lat;
    Scheduler sch(fp, lat);
    const int ga = Floorplan::mem_global(true, 2), gb = Floorplan::mem_global(false, 2);
    const int gc = Floorplan::mem_global(true, 10);
    std::vector<Pipeline> ps;
    for (int k = 0; k < count; ++k) {
        ps.push_back(add_pipeline(ga, gb, gc, n, "add" + std::to_string(k)));
        // different addresses per add so they are independent
        for (POp& o : ps.back().ops) o.addr0 += k * n;
        sch.place(ps.back());
    }
    Simulator s(fp, lat);
    std::mt19937 rng(1);
    for (int i = 0; i < count * n; ++i)
        for (int l = 0; l < kLanes; ++l) {
            s.mem(ga)[100 + i][l] = static_cast<std::uint8_t>(rng());
            s.mem(gb)[100 + i][l] = static_cast<std::uint8_t>(rng());
        }
    s.run(sch.emit());
    int bad = 0;
    for (int i = 0; i < count * n; ++i)
        for (int l = 0; l < kLanes; ++l) {
            const int a = static_cast<std::int8_t>(s.mem(ga)[100 + i][l]);
            const int b = static_cast<std::int8_t>(s.mem(gb)[100 + i][l]);
            const int c = std::max(-128, std::min(127, a + b));
            if (static_cast<std::int8_t>(s.mem(gc)[2000 + i][l]) != c) ++bad;
        }
    std::string starts;
    for (auto& p : ps) starts += std::to_string(p.ops[0].start) + " ";
    return check(title, bad == 0 && s.violations == 0,
                 "pipeline starts " + starts + "| makespan " + std::to_string(sch.makespan()) +
                     " | bad lanes " + std::to_string(bad) + " | violations " + std::to_string(s.violations));
}

static bool test_broken_schedule_caught() {
    Floorplan fp;
    LatencyTable lat;
    Scheduler sch(fp, lat);
    Pipeline p = add_pipeline(Floorplan::mem_global(true, 2), Floorplan::mem_global(false, 2),
                              Floorplan::mem_global(true, 10), 8, "broken");
    sch.place(p);
    Program prog = sch.emit();
    // delay the Write by one cycle: it now reads vector m-1's slot -> tag mismatch / empty
    auto& wq = prog.channel(q_mem_write(Floorplan::mem_global(true, 10)));
    for (auto& e : wq)
        if (e.kind == QEntry::Nop) { e.n += 1; break; }
    Simulator s(fp, lat);
    s.run(prog);
    return check("3b a Write moved by 1 cycle is reported as a timing error", s.violations > 0,
                 s.violation_log.empty() ? "" : s.violation_log[0]);
}

// 4: one MEM IQ issues a Read (R-type way) and a Write (W-type way) in the same cycle, on
// different banks [Ours, patent col. 16]; the single-issue variant serializes them; a Read put
// on the W-type way is rejected.
static bool test_mem_iq_dual_issue() {
    Floorplan fp;
    LatencyTable lat;
    const int A = Floorplan::mem_global(true, 11), B = Floorplan::mem_global(true, 10), C = Floorplan::mem_global(true, 5);
    const int n = 8;
    auto build = [&](Scheduler& sch, Pipeline& p1, Pipeline& p2) {
        p1.name = "A->B";   // writes B bank 1
        int r = p1.add_op(make_read(A, 0, n));
        int w = p1.add_op(make_write(B, 5000, n));
        int e = p1.add_edge(1);
        p1.produce(r, e, 0, 1);
        p1.consume(w, 0, e, 0, 1);
        sch.place(p1);
        p2.name = "B->C";   // reads B bank 0
        r = p2.add_op(make_read(B, 10, n));
        w = p2.add_op(make_write(C, 0, n));
        e = p2.add_edge(1);
        p2.produce(r, e, 0, 1);
        p2.consume(w, 0, e, 0, 1);
        sch.place(p2);
    };
    bool ok = true;
    std::string info;
    for (int single = 0; single < 2; ++single) {
        Scheduler sch(fp, lat);
        sch.mem_single_issue = single != 0;
        Pipeline p1, p2;
        build(sch, p1, p2);
        Simulator s(fp, lat);
        s.mem_single_issue = single != 0;
        for (int i = 0; i < n; ++i) { s.mem(A)[static_cast<std::size_t>(i)][0] = static_cast<std::uint8_t>(i + 1); s.mem(B)[static_cast<std::size_t>(10 + i)][0] = static_cast<std::uint8_t>(100 + i); }
        s.run(sch.emit());
        const std::int64_t w0 = p1.ops[1].start, r0 = p2.ops[0].start;
        const bool overlap = r0 < w0 + n && w0 < r0 + n;
        bool data = true;
        for (int i = 0; i < n; ++i) data = data && s.mem(B)[static_cast<std::size_t>(5000 + i)][0] == i + 1 && s.mem(C)[static_cast<std::size_t>(i)][0] == 100 + i;
        ok = ok && s.violations == 0 && data && (single ? !overlap : overlap);
        info += std::string(single ? "single-issue" : "dual-issue") + ": write B @" + std::to_string(w0) + ", read B @" +
                std::to_string(r0) + (overlap ? " (same cycles)" : " (serialized)") + ", violations " + std::to_string(s.violations) + "; ";
    }
    // a Read encoded on the W-type way of its IQ must be rejected
    {
        Scheduler sch(fp, lat);
        Pipeline p1, p2;
        build(sch, p1, p2);
        Program prog = sch.emit();
        auto& rway = prog.channel(q_mem_read(B));
        auto& wway = prog.channel(q_mem_write(B));
        std::swap(rway, wway);
        Simulator s(fp, lat);
        s.run(prog);
        ok = ok && s.violations > 0;
        info += "Read on the W-type way: " + (s.violation_log.empty() ? std::string("not caught") : s.violation_log[0]);
    }
    return check("4  one MEM IQ: Read + Write in the same cycle (different banks)", ok, info);
}

// 5: the program has exactly the 144 IQs, with the per-type split and column order of lpusim.h
static bool test_iq_map() {
    Floorplan fp;
    Program prog;
    int mem = 0, vxm = 0, mxm = 0, sxm = 0, ways = 0;
    bool order = true;
    int last = -1;
    for (int iq = 0; iq < kNumIQs; ++iq) {
        switch (iq_kind(iq)) {
            case IqKind::Mem: ++mem; break;
            case IqKind::Vxm: ++vxm; break;
            case IqKind::Mxm: ++mxm; break;
            case IqKind::Sxm: ++sxm; break;
        }
        ways += static_cast<int>(prog.iqs[static_cast<std::size_t>(iq)].ways.size());
        order = order && fp.iq_pos(iq) >= last;   // column order, west to east
        last = fp.iq_pos(iq);
    }
    bool roundtrip = true;
    for (int q = 0; q < kNumChannels; ++q) roundtrip = roundtrip && channel_of(channel_iq(q), channel_way(q)) == q;
    const bool ok = kNumIQs == 144 && mem == 88 && vxm == 16 && mxm == 12 && sxm == 28 && ways == kNumChannels && order && roundtrip;
    return check("5  144 IQs: 88 MEM + 28 SXM + 12 MXM + 16 VXM, column order, channel map", ok,
                 "MEM " + std::to_string(mem) + ", SXM " + std::to_string(sxm) + ", MXM " + std::to_string(mxm) + ", VXM " +
                     std::to_string(vxm) + ", ways " + std::to_string(ways) + ", MEM_W0 at " +
                     std::to_string(fp.queue_pos(q_mem_read(0))) + " = " +
                     std::to_string(fp.queue_pos(q_mem_read(0)) - fp.mxm_west_pos) + " hops from MXM_W");
}

// 6: bounded IQs [P col. 12/13]: an IQ whose last instruction byte never arrives is an underflow;
// preloading more than the IQ holds is an overflow; the exact preload passes.
static bool test_iq_bounds() {
    Floorplan fp;
    LatencyTable lat;
    Scheduler sch(fp, lat);
    Pipeline p = add_pipeline(Floorplan::mem_global(true, 2), Floorplan::mem_global(false, 2),
                              Floorplan::mem_global(true, 10), 8, "bounds");
    sch.place(p);
    auto run = [&](int short_by, int cap_short) {
        Program prog = sch.emit();
        for (auto& P : prog.iqs) {
            std::uint32_t off = 0;
            for (auto& w : P.ways)
                for (auto& e : w) {
                    e.byte_off = off;
                    e.byte_len = e.kind == QEntry::Nop ? kNopBytes : kInsBytes;
                    off += e.byte_len;
                }
            P.text.assign(off, 0);
            if (off == 0) continue;
            P.preload = off - static_cast<std::uint32_t>(short_by);
            P.capacity = off - static_cast<std::uint32_t>(cap_short);
        }
        Simulator s(fp, lat);
        s.run(prog);
        return std::make_pair(s.underflows, s.overflows);
    };
    const auto ok = run(0, 0), under = run(1, 0), over = run(0, 1);
    return check("6  bounded IQs: missing text byte -> underflow, preload > capacity -> overflow",
                 ok.first == 0 && ok.second == 0 && under.first > 0 && over.second > 0,
                 "exact: " + std::to_string(ok.first) + "/" + std::to_string(ok.second) + ", 1 B short: " +
                     std::to_string(under.first) + " underflows, capacity 1 B short: " + std::to_string(over.second) +
                     " overflows");
}

int main() {
    int fails = 0;
    fails += !test_streams_equivalence();
    fails += !test_read_write();
    fails += !run_adds(1, 1, "2  streaming add, 1 vector");
    fails += !run_adds(1, 1024, "2  streaming add, 1,024 vectors (Repeat)");
    fails += !run_adds(3, 256, "3  three adds share the same slices -> serialized by the reservation table");
    fails += !test_broken_schedule_caught();
    fails += !test_mem_iq_dual_issue();
    fails += !test_iq_map();
    fails += !test_iq_bounds();
    std::printf("\n%s (%d failed)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}
