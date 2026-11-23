// modules.h - shared types and the Slice base class for the LPU (TSP) cycle-level simulator.
//
// MEM, VXM, MXM and SXM slices all inherit from Slice. One cycle loop drives every slice
// through two virtual hooks:
//
//   dispatch(inst)  hand the slice an instruction that its queue issues this cycle
//   tick(cycle)     advance the slice by one clock: consume operands, advance pipelines,
//                   produce results onto stream registers
//
// Instructions are deliberately NOT a class hierarchy: one flat Instruction struct holds the
// opcode plus every operand and timing field any module needs. Unused fields stay at defaults.
//
// Numbers tagged [P] are published (Abts et al., ISCA 2020), [A] are our assumptions; see the
// planning outlines in planning/ for the reasoning.
//
// Driving loop (sketch):
//
//   StreamRegisterFile srf(kNumSrPositions);
//   std::vector<std::unique_ptr<Slice>> slices = build_chip(srf);
//   for (int cycle = 0; cycle < max_cycles; ++cycle) {
//       for (auto& s : slices)
//           if (auto inst = s->queue_issue(cycle)) s->dispatch(*inst);   // your IQ model
//       for (auto& s : slices) s->tick(cycle);   // reads see last cycle's SRs; order-free
//       srf.end_cycle();                         // commit writes, move streams one hop
//   }
//
// Because reads see the committed state and writes land in a separate buffer, the order in
// which slices are ticked inside one cycle does not change the result (determinism rule).

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace lpu {

// ---------------------------------------------------------------------------------------------
// Chip geometry [P] p.146-147
// ---------------------------------------------------------------------------------------------
inline constexpr int kSuperlanes = 20;                         // = tiles per slice
inline constexpr int kTilesPerSlice = kSuperlanes;
inline constexpr int kLanesPerSuperlane = 16;
inline constexpr int kLanes = kSuperlanes * kLanesPerSuperlane;  // 320
inline constexpr int kStreamsPerDirection = 32;
inline constexpr int kBytesPerTilePerStream = kLanesPerSuperlane;   // 16 B per tile per cycle
inline constexpr int kBytesPerStream = kLanes;                      // 320 B per stream per cycle

// One stream's worth of data in one cycle: 1 byte per lane, 320 lanes (16 B x 20 superlanes).
// Byte index = superlane * 16 + lane_in_superlane. Multi-byte types span a stream pair/quad.
using Vector = std::array<std::uint8_t, kBytesPerStream>;

// Lane-index helper: superlane t, lane l within it.
constexpr int lane_index(int superlane, int lane) { return superlane * kLanesPerSuperlane + lane; }

// ---------------------------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------------------------
enum class Direction : std::uint8_t { East, West };
enum class Hemisphere : std::uint8_t { East, West };
enum class SliceKind : std::uint8_t { Mem, Vxm, Mxm, Sxm };

enum class DataType : std::uint8_t { Int8, Int16, Int32, Fp16, Fp32 };

// Bytes per element; also the number of consecutive streams an operand occupies
// (int16 = stream pair, int32/fp32 = SG4) [P] p.146.
constexpr int streams_for(DataType t) {
    switch (t) {
        case DataType::Int8:  return 1;
        case DataType::Int16: return 2;
        case DataType::Fp16:  return 2;
        case DataType::Int32: return 4;
        case DataType::Fp32:  return 4;
    }
    return 1;
}

// Every opcode from Table I (p.149), grouped by the module that executes it.
enum class Opcode : std::uint8_t {
    // ICU instructions: legal on every slice's queue [P] p.151-152
    Nop,        // count = N cycles
    Repeat,     // repeat previous instruction: count = n times, delay = d cycles apart
    Sync,       // chip-wide barrier, ~35 cycles [P] p.152
    Notify,
    Ifetch,     // 640 B of instructions into the queue
    Config,     // low-power mode per superlane (mask in imm)

    // MEM [P] Table I
    MemRead,    // address -> dst stream
    MemWrite,   // src[0] stream -> address
    MemGather,  // src[0] = map stream (addresses) -> dst stream
    MemScatter, // src[0] = data stream, src[1] = map stream

    // VXM [P] Table I, p.152. Binary ops use src[0], src[1]; unary ops src[0].
    VxmAdd, VxmSub, VxmMul, VxmMax, VxmMin,   // saturate flag picks _sat vs _mod
    VxmMask, VxmNegate,
    VxmRelu, VxmTanh, VxmExp, VxmRsqrt,
    VxmConvert,                               // dtype -> dst_dtype

    // MXM [P] Table I, p.152-153
    MxmLoadWeights,      // LW: src[0] (16 streams) -> weight buffer of `unit` (plane)
    MxmInstallWeights,   // IW: from streams (src[0]) or from LW buffer (from_buffer = true)
    MxmActivate,         // ABC: activations src[0] -> results dst (SG4, int32/fp32)
    MxmAccumulate,       // ACC: src[0] + src[1] -> dst

    // SXM [P] Table I, p.153, Fig. 8
    SxmShiftNorth,  // imm = N lanes
    SxmShiftSouth,  // imm = N lanes
    SxmSelect,      // src[0] unshifted, src[1] North, src[2] South, src[3] selection stream
    SxmPermute,     // bijective 320-lane map, map_id selects the stored map
    SxmDistribute,  // 16-lane remap per superlane with replication/zero-fill, map_id
    SxmRotate,      // imm = n (3 or 4), produces n*n output streams
    SxmTranspose,   // sg16: src[0] width 16 -> dst width 16, per superlane
};

constexpr bool is_icu_op(Opcode op) { return op <= Opcode::Config; }

// Which slice type executes an opcode. ICU ops return the caller's kind (any slice runs them).
constexpr SliceKind slice_kind_of(Opcode op, SliceKind self) {
    if (is_icu_op(op)) return self;
    if (op <= Opcode::MemScatter) return SliceKind::Mem;
    if (op <= Opcode::VxmConvert) return SliceKind::Vxm;
    if (op <= Opcode::MxmAccumulate) return SliceKind::Mxm;
    return SliceKind::Sxm;
}

// Printable opcode names, in enum order (used by the visualization export).
inline const char* opcode_name(Opcode op) {
    static constexpr const char* kNames[] = {
        "Nop", "Repeat", "Sync", "Notify", "Ifetch", "Config",
        "Read", "Write", "Gather", "Scatter",
        "add", "sub", "mul", "max", "min", "mask", "negate", "relu", "tanh", "exp", "rsqrt", "convert",
        "LW", "IW", "ABC", "ACC",
        "shift_n", "shift_s", "select", "permute", "distribute", "rotate", "transpose"};
    static_assert(sizeof(kNames) / sizeof(kNames[0]) == static_cast<std::size_t>(Opcode::SxmTranspose) + 1,
                  "opcode_name: one name per Opcode, in enum order");
    const auto i = static_cast<std::size_t>(op);
    return i < sizeof(kNames) / sizeof(kNames[0]) ? kNames[i] : "?";
}

// ---------------------------------------------------------------------------------------------
// Instruction: one flat struct for every module
// ---------------------------------------------------------------------------------------------

// A group of consecutive streams in one direction: one stream (int8), a pair, an SG4 (base must
// be a multiple of 4), or SG16 for transposes and weight installs.
struct StreamGroup {
    std::uint8_t base = 0;   // first stream id, 0..31
    std::uint8_t width = 0;  // 0 = unused operand, else 1, 2, 4 or 16
    Direction dir = Direction::East;

    constexpr bool used() const { return width != 0; }
    constexpr bool valid() const {
        return width != 0 && base + width <= kStreamsPerDirection &&
               (width == 1 || base % width == 0);   // pairs/SG4/SG16 must be aligned [P] p.146
    }
};

struct Instruction {
    Opcode opcode = Opcode::Nop;
    DataType dtype = DataType::Int8;      // operand type
    DataType dst_dtype = DataType::Int8;  // result type (VxmConvert, MXM int8 -> int32, ...)

    // Stream operands. Up to 4 inputs covers SxmSelect (3 data + selection stream).
    std::array<StreamGroup, 4> src{};
    StreamGroup dst{};                    // dst.dir is the result direction (any slice may turn it)

    // Module-specific operands
    std::uint16_t address = 0;   // MEM: 13-bit word address [P] p.148
    std::uint8_t unit = 0;       // MXM plane, VXM ALU index (0..15), SXM unit index, MEM bank
    std::uint8_t queue = 0;      // which instruction queue of the block issued it
    std::uint8_t map_id = 0;     // SXM permute/distribute map slot
    std::int32_t imm = 0;        // shift amount, rotate n, Config mask, ...
    bool saturate = false;       // VXM _sat (true) vs _mod (false)
    bool from_buffer = false;    // MXM IW: install from the LW buffer instead of streams

    // ICU fields
    std::uint16_t count = 0;     // NOP N, Repeat n
    std::uint16_t delay = 0;     // Repeat d

    // Timing [P] Eq. 4 p.151: T = N + d_func + delta(j,i). Values come from the config table;
    // the assembler fills them in so slices do not look them up per cycle.
    std::int16_t d_func = 1;     // functional delay: tile reached -> result on its SR
    std::int16_t d_skew = 0;     // cycles between instruction arrival and operand read
};

// ---------------------------------------------------------------------------------------------
// Stream registers
// ---------------------------------------------------------------------------------------------
// SR positions are numbered along the chip from West edge (0) to East edge (num_positions-1).
// East-flowing streams move to position p+1 each cycle, West-flowing to p-1 [P] p.149, p.155.
// A slice reads at its own position and may overwrite a stream there; anything not
// overwritten keeps flowing. Values falling off the chip edge are dropped [A].
//
// Granularity: one Vector holds all 20 superlanes of a stream for one cycle. The 1-cycle-per-
// superlane stagger [P] Fig. 6 is bookkeeping inside the slices (per-tile timestamps) [A].
class StreamRegisterFile {
public:
    struct Entry {
        Vector data{};
        bool valid = false;   // false = nothing produced onto this stream yet
    };

    explicit StreamRegisterFile(int num_positions)
        : cur_(static_cast<std::size_t>(num_positions)),
          next_(static_cast<std::size_t>(num_positions)) {}

    int num_positions() const { return static_cast<int>(cur_.size()); }

    // Committed value at (position, direction, stream) as of the start of this cycle.
    const Entry& read(int pos, Direction dir, int stream) const {
        return cur_.at(idx(pos))[dir_idx(dir)].at(static_cast<std::size_t>(stream));
    }

    // Produce a value; it becomes visible at `pos` next cycle (and then flows on).
    // Returns false if another slice already wrote this register this cycle: that is a
    // scheduling conflict the caller should report (the hardware has no arbiter).
    bool write(int pos, Direction dir, int stream, const Vector& v) {
        Slot& s = next_.at(idx(pos))[dir_idx(dir)].at(static_cast<std::size_t>(stream));
        if (s.written) return false;
        s.entry.data = v;
        s.entry.valid = true;
        s.written = true;
        return true;
    }

    // Commit writes and move every unwritten stream one hop in its direction of flow.
    void end_cycle() {
        const int n = num_positions();
        for (int p = 0; p < n; ++p) {
            for (int d = 0; d < 2; ++d) {
                const int from = (d == dir_idx(Direction::East)) ? p - 1 : p + 1;
                for (int s = 0; s < kStreamsPerDirection; ++s) {
                    Slot& slot = next_[idx(p)][d][static_cast<std::size_t>(s)];
                    if (!slot.written) {
                        slot.entry = (from >= 0 && from < n)
                                         ? cur_[idx(from)][d][static_cast<std::size_t>(s)]
                                         : Entry{};
                    }
                }
            }
        }
        for (int p = 0; p < n; ++p) {
            for (int d = 0; d < 2; ++d) {
                for (int s = 0; s < kStreamsPerDirection; ++s) {
                    Slot& slot = next_[idx(p)][d][static_cast<std::size_t>(s)];
                    cur_[idx(p)][d][static_cast<std::size_t>(s)] = slot.entry;
                    slot.written = false;
                }
            }
        }
    }

private:
    struct Slot {
        Entry entry{};
        bool written = false;
    };
    using Streams = std::array<Entry, kStreamsPerDirection>;
    using Slots = std::array<Slot, kStreamsPerDirection>;

    static std::size_t idx(int pos) { return static_cast<std::size_t>(pos); }
    static constexpr int dir_idx(Direction d) { return d == Direction::East ? 0 : 1; }

    std::vector<std::array<Streams, 2>> cur_;
    std::vector<std::array<Slots, 2>> next_;
};

// ---------------------------------------------------------------------------------------------
// Slice: base class for MEM, VXM, MXM and SXM
// ---------------------------------------------------------------------------------------------
struct SliceStats {
    std::uint64_t instructions = 0;   // instructions dispatched
    std::uint64_t busy_cycles = 0;    // cycles with work in flight
    std::uint64_t violations = 0;     // scheduling conflicts detected
};

class Slice {
public:
    Slice(SliceKind kind, Hemisphere hemisphere, int index, int sr_position,
          StreamRegisterFile& streams)
        : kind_(kind), hemisphere_(hemisphere), index_(index),
          sr_position_(sr_position), streams_(streams) {}

    virtual ~Slice() = default;

    // Slices are owned through std::unique_ptr<Slice> and never copied.
    Slice(const Slice&) = delete;
    Slice& operator=(const Slice&) = delete;
    Slice(Slice&&) = delete;
    Slice& operator=(Slice&&) = delete;

    // Hand the slice an instruction issued this cycle. Called before tick() of the same cycle.
    // The slice queues it internally and applies d_skew / d_func / tile stagger itself.
    virtual void dispatch(const Instruction& inst) = 0;

    // Advance one clock. Read operands with streams().read(), produce results with
    // produce(); never modify another slice.
    virtual void tick(int cycle) = 0;

    // True if this slice type can execute `op` (ICU ops are always accepted).
    virtual bool accepts(Opcode op) const {
        return slice_kind_of(op, kind_) == kind_;
    }

    // True while any instruction is still in flight (lets the driver detect the end of a run).
    virtual bool busy() const { return false; }

    // Clear internal state between runs.
    virtual void reset() { stats_ = SliceStats{}; }

    // e.g. "MEM_W12", "MXM_E", used in traces and violation reports.
    virtual std::string name() const {
        static constexpr const char* kNames[] = {"MEM", "VXM", "MXM", "SXM"};
        return std::string(kNames[static_cast<int>(kind_)]) +
               (hemisphere_ == Hemisphere::East ? "_E" : "_W") + std::to_string(index_);
    }

    SliceKind kind() const { return kind_; }
    Hemisphere hemisphere() const { return hemisphere_; }
    int index() const { return index_; }              // e.g. 0..43 for MEM within a hemisphere
    int sr_position() const { return sr_position_; }  // where this slice taps the SR file
    const SliceStats& stats() const { return stats_; }

protected:
    StreamRegisterFile& streams() { return streams_; }
    const StreamRegisterFile& streams() const { return streams_; }

    // Write a result at this slice's SR position; counts a violation on a double write.
    bool produce(Direction dir, int stream, const Vector& v) {
        const bool ok = streams_.write(sr_position_, dir, stream, v);
        if (!ok) ++stats_.violations;
        return ok;
    }

    SliceStats stats_;

private:
    SliceKind kind_;
    Hemisphere hemisphere_;
    int index_;
    int sr_position_;
    StreamRegisterFile& streams_;
};

}  // namespace lpu
