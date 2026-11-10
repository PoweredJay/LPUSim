// sched.h - static scheduler for the LPU simulator.
//
// The unit of scheduling is a *pipeline*: a set of streaming instructions (each one instruction
// plus Repeat n) connected by stream edges, e.g. Read -> Gather -> ABC -> VXM chain -> Write.
// Inside a pipeline every timing relation is fixed by the hardware, because streams cannot be
// buffered: for each edge producer a -> consumer c,
//
//     phase(c) + d_skew(c) = phase(a) + d_func(a) + hops(a, c)        ([P] Eq. 4, p.151)
//
// where the issue cycle of vector m of an op is  S + phase(op) + m. The scheduler solves these
// equations once (rigid template), then list-schedules pipelines in program order: each one is
// placed at the earliest start S that satisfies
//   * data dependencies through MEM, tracked per word (RAW, WAR, WAW), so a consumer can start
//     reading a tensor before its producer has finished writing it (paper p.154, ~5,500 cycles)
//   * per-op lower bounds the compiler adds (MXM plane state: LW before IW before ABC)
//   * a reservation table: issue-channel busy intervals (one channel per way of one of the 144
//     instruction queues, see lpusim.h), MEM bank read/write ports, and the
//     stream registers. A value on stream s flowing East written at position q in cycle t sits
//     on "diagonal" D = t + 1 - q; an edge occupies the rectangle [D0, D0+n) x [q, consumer] of
//     (diagonal, position) on each of its streams, and two rectangles may not overlap.
// Stream ids and VXM ALUs are picked during placement (first free), so the compiler never names
// them. The result is converted to per-IQ programs (one NOP / instruction / Repeat list per way)
// with NOP gaps and Repeats.
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "lpusim.h"

namespace lpu {
namespace sched {

using sim::Floorplan;
using sim::LatencyTable;
using sim::Program;
using sim::SimInstr;

enum class MemKind : std::uint8_t { None, ReadSeq, WriteSeq, Gather };

struct POp {
    int queue = -1;             // fixed issue channel (sim::q_*), or -1 = any free VXM ALU
    SimInstr ins;               // opcode, address, imm, widths; stream ids are filled in by place()
    int m0 = 0;                 // covers pipeline vectors [m0, m0 + n)
    int n = 1;
    std::vector<int> in_edge;   // operand k -> edge index in the pipeline
    std::vector<int> in_sub;    // operand k reads streams [edge.base + sub, + width)
    std::vector<int> in_width;
    int out_edge = -1;
    int out_sub = 0;            // writes streams [edge.base + out_sub, + out_width)
    int out_width = 0;
    std::int64_t min_issue = INT64_MIN;   // lower bound on the first issue cycle

    // Memory effects (for dependency tracking)
    MemKind mk = MemKind::None;
    int mem_g = -1;
    int addr0 = 0;
    int stride = 1;
    std::vector<std::uint16_t> gather;    // Gather: n * 20 word addresses (per superlane)

    // Trace info
    std::string label;
    int layer = -1;
    char kind = '?';            // R read, G gather, W write, M map read, L LW, I IW, A ABC, V vxm

    // Filled in by the scheduler
    int phase = 0;
    int pos = 0;
    std::int64_t start = -1;    // first issue cycle (program time)
    int placed_queue = -1;
};

struct Edge {
    int width = 1;              // streams in the group: 1, 4 or 16
    std::vector<int> producers;
    std::vector<int> consumers;
    // chosen at placement
    Direction dir = Direction::East;
    int base = -1;
    std::uint32_t gid = 0;
};

struct Pipeline {
    std::string name;
    int layer = -1;
    std::vector<POp> ops;
    std::vector<Edge> edges;
    std::int64_t min_start = 0;
    // extra rigid timing links: first issue of op b = first issue of op a + delta
    struct Link { int a, b, delta; };
    std::vector<Link> links;

    int add_op(const POp& op) { ops.push_back(op); return static_cast<int>(ops.size()) - 1; }
    int add_edge(int width) { Edge e; e.width = width; edges.push_back(e); return static_cast<int>(edges.size()) - 1; }
    // op writes `width` streams of edge starting at `sub`
    void produce(int op, int edge, int sub, int width);
    // op reads edge streams [sub, sub+width) as operand k
    void consume(int op, int k, int edge, int sub, int width);
};

struct TraceOp {
    int queue;
    int pos;
    std::int64_t start;
    int n;
    char kind;
    int layer;
    std::string label;
};

// Event export (visualization). Recorded only when Scheduler::record_events is set; recording
// never changes a placement decision, so results are identical with it on or off.
struct EventOp {
    std::int64_t start;         // first issue cycle (program time, no Sync preamble)
    int queue;                  // issue channel
    int pos;
    int n;
    int pipe;                   // index of the pipeline (placement order)
    int layer;
    char kind;
    Opcode opcode;
    std::array<StreamGroup, 4> src;
    StreamGroup dst;
    int mem_g = -1, bank = -1;  // MEM ops only
    int addr_lo = -1, addr_hi = -1, stride = 0;   // word range touched (Gather: min/max of its map)
    int d_func = 0;
    std::string label;
};
struct EventStream {            // one stream of one edge: rectangle in (diagonal x position)
    int pipe;
    std::uint32_t edge;         // edge gid (matches SimInstr tags)
    Direction dir;
    int stream;
    std::int64_t d0, d1;        // diagonals [d0, d1): East D = t - p, West D = t + p (program time)
    int p0, p1;                 // inclusive positions
    int producer;               // index into events (EventOp) of the producing op
};
struct EventPipe { std::string name; int layer; std::int64_t S; };

class Scheduler {
public:
    Scheduler(const Floorplan& fp, const LatencyTable& lat);

    // Solve the template, find the earliest legal start, reserve resources, record the ops.
    // Returns the start cycle S (vector 0 of phase-0 op).
    // commit = false: only find S (trial placement; nothing is reserved).
    std::int64_t place(Pipeline& p, bool commit = true);

    // Per-IQ programs with NOP gaps; every used way starts with the Sync preamble.
    Program emit() const;

    // Sensitivity switch: a MEM IQ issues ONE instruction per cycle (its R-type and W-type ways
    // share one busy table). Default false = the patent's "one R-type and one W-type" per cycle.
    bool mem_single_issue = false;
    int res(int q) const { return mem_single_issue && sim::q_is_mem(q) ? (q & ~1) : q; }

    // Mark words as already holding data at time 0 (weights, maps, input) - nothing to do,
    // all words start ready; kept for clarity.
    std::int64_t makespan() const { return makespan_; }
    const sim::LatencyTable& lat() const { return lat_; }
    std::int64_t floor = 0;   // no pipeline starts an op before this cycle (no-layer-overlap baseline)
    const std::vector<TraceOp>& trace() const { return trace_; }
    std::uint64_t search_steps() const { return steps_; }
    std::uint64_t placed() const { return placed_; }
    std::int64_t word_ready(int g, int addr) const { return ready_[idx(g, addr)]; }

    bool record_events = false;
    std::vector<EventOp> ev_ops;
    std::vector<EventStream> ev_streams;
    std::vector<EventPipe> ev_pipes;

private:
    struct Rect {
        std::int64_t d0, d1;    // [d0, d1)
        int p0, p1;             // inclusive
    };
    using Intervals = std::map<std::int64_t, std::int64_t>;

    static std::size_t idx(int g, int addr) { return static_cast<std::size_t>(g) * sim::kMemWords + static_cast<std::size_t>(addr); }
    void solve(Pipeline& p) const;
    std::int64_t mem_lower_bound(const Pipeline& p) const;
    void commit_mem(const Pipeline& p, std::int64_t S);
    static bool overlaps(const Intervals& iv, std::int64_t a, std::int64_t b, std::int64_t* end);
    static void insert(Intervals& iv, std::int64_t a, std::int64_t b);
    bool rect_free(int dir, int stream, const Rect& r, std::int64_t* need_d) const;
    void rect_add(int dir, int stream, const Rect& r);
    int dfunc(const POp& op) const { return lat_.of(op.ins.in.opcode).d_func; }
    int dskew(const POp& op) const { return lat_.of(op.ins.in.opcode).d_skew; }
    void edge_rects(const Pipeline& p, int e, Direction dir, std::int64_t S,
                    std::vector<std::pair<int, Rect>>* out) const;   // (stream offset, rect)

    const Floorplan& fp_;
    const LatencyTable& lat_;
    std::vector<Intervals> queue_busy_;
    std::vector<Intervals> bank_read_, bank_write_;
    std::vector<Rect> rects_;
    std::unordered_map<std::int64_t, std::vector<int>> rect_buckets_;
    std::vector<std::int64_t> ready_;      // first cycle a read sees the latest write
    std::vector<std::int64_t> last_read_;
    std::vector<TraceOp> trace_;
    std::vector<std::pair<int, SimInstr>> emitted_;   // (queue, instr with start in op_id slot)
    std::vector<std::pair<std::int64_t, int>> emitted_time_n_;
    std::uint32_t next_gid_ = 1;
    std::int64_t makespan_ = 0;
    std::uint64_t steps_ = 0;
    std::uint64_t placed_ = 0;
};

}  // namespace sched
}  // namespace lpu
