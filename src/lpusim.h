// lpusim.h - cycle-level LPU (TSP) simulator that executes scheduled instruction-queue programs.
//
// Builds on ../modules_full.h (lpu::Instruction, lpu::Vector, Opcode, StreamGroup). What is new:
//   * Floorplan: SR position of every slice (one SR hop per slice) [A].
//   * LatencyTable: d_func / d_skew / occupancy per opcode, read from
//     planning/instruction-timing-table.csv (exported from the .xlsx by planning/xlsx_to_csv.py),
//     with tagged defaults for blank cells.
//   * DiagStreams: an event-driven stream-register file with the same semantics as
//     lpu::StreamRegisterFile (values move one hop per cycle, a write is visible at the
//     writer's position next cycle, later writers overwrite) but O(1) per access instead of
//     copying every SR every cycle. tests.cpp checks the two agree on random traffic.
//   * Simulator: 144 instruction queues (IQs). MEM slices (one IQ each, R-type + W-type ways,
//     2 banks, Read/Write/Gather), the VXM (16 ALU IQs), 4 MXM planes (LW buffer, IW, ABC; 3 IQs
//     per plane), 28 SXM IQs (no SXM op is modeled yet). Each IQ way runs a program of NOP N /
//     instruction / Repeat n,d entries, exactly what the scheduler emits.
//
// Checks done every cycle (all are hard errors the scheduler must never trigger):
//   - two writers on one stream register in one cycle
//   - an instruction reading a stream that does not hold the value it expects (every value
//     carries a (edge, index) tag; every operand carries the tag it should see)
//   - MEM read and write of the same bank of one slice in the same cycle (pseudo-dual-port)
//   - an opcode issued on an IQ (or MEM way) that cannot execute it
//   - more than one R-type or more than one W-type instruction from one MEM IQ in one cycle,
//     or more than one instruction from any other IQ in one cycle
//
// Superlane stagger: every op used here is lane-local except the MXM reduction, whose 20-cycle
// d_func already covers it (decision T2), so superlane t is an exact copy of superlane 0 shifted
// by t cycles. We simulate superlane-0 time and add LatencyTable::stagger (19) to the end time.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "../modules_full.h"

namespace lpu {
namespace sim {

inline constexpr int kMemSlices = 88;      // 44 per hemisphere [P] p.148
inline constexpr int kMemPerHem = 44;
inline constexpr int kMemWords = 8192;     // 13-bit address [P] p.148
inline constexpr int kBankWords = 4096;    // bank = address bit 12 [A]
inline constexpr int kPlanes = 4;          // 2 per MXM, W0 W1 E0 E1 [P] Fig. 5
inline constexpr int kVxmAlus = 16;        // per lane [P] p.152

// ---------------------------------------------------------------------------------------------
// Instruction queues (IQs) and issue channels
//
// The chip has 144 independent IQs in the ICU [P] p.148, patent US 11,360,934 col. 9
// ("Each of the 144 independent on-chip ICUs can issue one or more instructions per clock
// cycle"). Each IQ dispatches up one column (one functional slice).
// Per-type split [Ours, planning doc]: per hemisphere 44 MEM + 14 SXM + 6 MXM = 64, plus 16 VXM
// shared: 2 x 64 + 16 = 144. IQ ids follow the column order, west to east:
//     0-5 MXM_W | 6-19 SXM_W | 20-63 MEM_W43..W0 | 64-79 VXM | 80-123 MEM_E0..E43 |
//     124-137 SXM_E | 138-143 MXM_E
//
// An IQ can issue more than one instruction per cycle (patent col. 19: "multi-way instruction queue
// dispatch"). We model that as "ways": independent NOP / instruction / Repeat threads inside one
// IQ. The scheduler reserves *issue channels*, one per way:
//   * MEM IQ [Ours]: 2 ways. Way 0 issues R-type ops (Read, Gather), way 1 W-type ops (Write,
//     Scatter), so a Read and a Write can issue from the same IQ in one cycle. The patent's rule
//     "a memory tile can handle at most one R-type and one W-type instruction" per cycle (col. 16), and the
//     bank rule (no read + write on one bank in one cycle, pseudo-dual-port), are both checked.
//   * MXM [Ours]: the 6 IQs of a side are 3 per plane, one per instruction class that can be in
//     flight at the same time: LW (stream the next tile into the plane's LW buffer), IW (buffer
//     -> weight array), ABC (stream activations through the array; ACC would also issue here).
//     LW of tile k+1 overlaps ABC of tile k; IW sits between them in time (enforced by the
//     compiler's per-plane ordering, not by sharing an IQ).
//   * VXM: 16 IQs, one per ALU, 1 way each.
//   * SXM: 14 IQs per side, 1 way each. ResNet-50 leaves them empty.
// Every IQ other than MEM is single-issue [Ours].
// ---------------------------------------------------------------------------------------------
inline constexpr int kMxmIqsPerPlane = 3;                    // LW, IW, ABC [Ours]
inline constexpr int kMxmIqsPerHem = 2 * kMxmIqsPerPlane;    // 6 [Ours]
inline constexpr int kSxmIqsPerHem = 14;                     // [Ours]
inline constexpr int kIqsPerHem = kMemPerHem + kSxmIqsPerHem + kMxmIqsPerHem;   // 64
inline constexpr int kNumIQs = 2 * kIqsPerHem + kVxmAlus;    // 144
static_assert(kNumIQs == 144, "144 IQs");
enum class IqKind : std::uint8_t { Mem, Sxm, Mxm, Vxm };
enum MxmRole : int { kMxmLW = 0, kMxmIW = 1, kMxmABC = 2 };

inline constexpr int kIqMxmW = 0, kIqSxmW = kIqMxmW + kMxmIqsPerHem, kIqMemW = kIqSxmW + kSxmIqsPerHem;
inline constexpr int kIqVxm = kIqMemW + kMemPerHem, kIqMemE = kIqVxm + kVxmAlus;
inline constexpr int kIqSxmE = kIqMemE + kMemPerHem, kIqMxmE = kIqSxmE + kSxmIqsPerHem;
// global MEM slice g: West 0..43 (index), East 44..87; West IQs run W43..W0 west to east
inline int iq_mem(int g) { return g < kMemPerHem ? kIqMemW + (kMemPerHem - 1 - g) : kIqMemE + (g - kMemPerHem); }
inline int iq_vxm(int alu) { return kIqVxm + alu; }
inline int iq_mxm(int plane, int role) {   // plane 0,1 = West (W0, W1), 2,3 = East
    return plane < 2 ? kIqMxmW + plane * kMxmIqsPerPlane + role : kIqMxmE + (plane - 2) * kMxmIqsPerPlane + role;
}
inline int iq_sxm(bool west, int k) { return (west ? kIqSxmW : kIqSxmE) + k; }
IqKind iq_kind(int iq);
// Ways per IQ: MEM 2 (R-type, W-type), others 1, plus one fetch way on every IQ (the last way)
// that runs Ifetch alongside normal issue (patent col. 12: "All functional slices can fetch
// instructions simultaneously with normal instruction execution") [Ours: modeled as a way].
inline int iq_ways(int iq) { return (iq_kind(iq) == IqKind::Mem ? 2 : 1) + 1; }
inline int iq_fetch_way(int iq) { return iq_ways(iq) - 1; }

// Issue channels = (IQ, way): what the scheduler reserves. Ids:
//   MEM 0..175 (2g = R-type way of slice g, 2g+1 = W-type way), VXM 176..191,
//   MXM 192..203 (192 + 3 * plane + role), SXM 204..231.
inline constexpr int kQVxmBase = 2 * kMemSlices;                      // 176
inline constexpr int kQMxmBase = kQVxmBase + kVxmAlus;                // 192
inline constexpr int kQSxmBase = kQMxmBase + kMxmIqsPerPlane * kPlanes;   // 204
inline constexpr int kQFetchBase = kQSxmBase + 2 * kSxmIqsPerHem;    // 232: fetch way of IQ i = 232 + i
inline constexpr int kNumChannels = kQFetchBase + kNumIQs;            // 376
inline int q_mem_read(int g) { return 2 * g; }            // R-type way of MEM slice g's IQ
inline int q_mem_write(int g) { return 2 * g + 1; }       // W-type way of the same IQ
inline int q_vxm(int alu) { return kQVxmBase + alu; }
inline int q_mxm_lw(int plane) { return kQMxmBase + kMxmIqsPerPlane * plane + kMxmLW; }
inline int q_mxm_iw(int plane) { return kQMxmBase + kMxmIqsPerPlane * plane + kMxmIW; }
inline int q_mxm_abc(int plane) { return kQMxmBase + kMxmIqsPerPlane * plane + kMxmABC; }
inline int q_sxm(bool west, int k) { return kQSxmBase + (west ? 0 : kSxmIqsPerHem) + k; }
inline bool q_is_mem(int q) { return q < kQVxmBase; }
inline bool q_is_vxm(int q) { return q >= kQVxmBase && q < kQMxmBase; }
inline bool q_is_mxm(int q) { return q >= kQMxmBase && q < kQSxmBase; }
inline bool q_is_sxm(int q) { return q >= kQSxmBase && q < kQFetchBase; }
inline bool q_is_fetch(int q) { return q >= kQFetchBase; }
inline int q_fetch(int iq) { return kQFetchBase + iq; }
inline int q_mem_slice(int q) { return q / 2; }
inline int q_plane(int q) { return (q - kQMxmBase) / kMxmIqsPerPlane; }
inline int q_mxm_role(int q) { return (q - kQMxmBase) % kMxmIqsPerPlane; }
inline int q_sxm_index(int q) { return (q - kQSxmBase) % kSxmIqsPerHem; }
inline bool q_sxm_west(int q) { return q - kQSxmBase < kSxmIqsPerHem; }
// Instruction text [patent col. 10: 8-bit opcode for NOP, 32-bit (or 8-bit) discrete
// instructions; exact encoding Ours]: NOP = 1 byte, every other instruction (incl. Repeat and
// Ifetch) = 4 bytes. Each IQ holds 640 bytes = two 320-byte vectors [patent col. 12].
inline constexpr int kNopBytes = 1;
inline constexpr int kInsBytes = 4;
inline constexpr int kIqBytes = 640;
inline constexpr int kFetchBytesPerCycle = 16;   // [ISCA 2020, per the planning doc]: one superlane's 16 B
// channel -> (IQ, way) and back
int channel_iq(int q);
int channel_way(int q);
int channel_of(int iq, int way);

// ---------------------------------------------------------------------------------------------
// Floorplan: one row of the chip, West edge = position 0.
//   MXM_W | SXM_W (4 positions) | MEM_W43 .. MEM_W0 | VXM | MEM_E0 .. MEM_E43 | SXM_E | MXM_E
// [P] order from Fig. 2/4; [A] one SR hop per slice and 4 hops across the SXM.
// Global MEM slice id g: West slices 0..43 (g = index), East slices 44..87 (g = 44 + index).
// Index 0 is next to the VXM, 43 next to the SXM/MXM.
// ---------------------------------------------------------------------------------------------
struct Floorplan {
    int sxm_hops = 4;
    int num_positions = 0;
    int vxm_pos = 0;
    int mxm_west_pos = 0;
    int mxm_east_pos = 0;
    std::array<int, kMemSlices> mem_pos{};

    explicit Floorplan(int sxm_hops_ = 4);
    static bool mem_west(int g) { return g < kMemPerHem; }
    static int mem_index(int g) { return g % kMemPerHem; }
    static int mem_global(bool west, int index) { return west ? index : kMemPerHem + index; }
    static bool plane_west(int p) { return p < 2; }
    int plane_pos(int p) const { return plane_west(p) ? mxm_west_pos : mxm_east_pos; }
    std::array<int, 2 * kSxmIqsPerHem> sxm_pos{};   // SXM IQ k of a side -> position
    int queue_pos(int q) const;                // position of an issue channel's column
    std::string queue_name(int q) const;       // channel name: IQ name (+ ".R" / ".W" on MEM)
    int iq_pos(int iq) const;
    std::string iq_name(int iq) const;
};

// ---------------------------------------------------------------------------------------------
// Latency table. Defaults are the values in planning/ (tag "A" assumed, "I" inferred);
// load_csv() overrides any cell that holds a number in instruction-timing-table.csv.
// ---------------------------------------------------------------------------------------------
struct Lat {
    int d_func = 1;       // issue (tile 0) -> result visible on the slice's SR
    int d_skew = 0;       // issue -> operands read from the SR
    int occupancy = 1;    // cycles the queue is busy per issue
    std::string tag = "A";
    std::string source = "default";
};

struct LatencyTable {
    Lat mem_read{1, 0, 1, "A", "instruction-timing-table.csv MEM Read"};
    Lat mem_write{0, 0, 1, "A", "instruction-timing-table.csv MEM Write (d_skew 0)"};
    Lat mem_gather{2, 0, 1, "A", "instruction-timing-table.csv MEM Gather"};
    Lat vxm_add{1, 0, 1, "A", "default: 1-cycle ALU"};
    Lat vxm_mul{2, 0, 1, "A", "default: 2-cycle multiplier"};
    Lat vxm_convert{1, 0, 1, "A", "default: shift+round+saturate"};
    Lat vxm_relu{1, 0, 1, "A", "default"};
    Lat vxm_max{1, 0, 1, "A", "default"};
    Lat mxm_lw{1, 0, 1, "A", "LW: 1 issue per cycle, Repeat 20 = one plane"};
    Lat mxm_iw{1, 0, 1, "A", "IW from LW buffer (T5)"};
    Lat mxm_abc{20, 0, 1, "I", "ABC d_func 20 (HC34, T1)"};
    Lat icu_ifetch{1, 0, 1, "Ours", "Ifetch: consumes one 16-byte piece of its stream per cycle"};
    int sync_cycles = 35;   // [P] p.152 Notify->Sync, program preamble
    int stagger = 19;       // [P] Fig. 6: superlane 19 finishes 19 cycles after superlane 0
    double clock_ghz = 1.0; // the paper's math uses 1 GHz; 0.9 GHz nominal

    bool load_csv(const std::string& path, std::string* log);
    const Lat& of(Opcode op) const;
    void dump_csv(const std::string& path) const;
};

// ---------------------------------------------------------------------------------------------
// Stream registers
// ---------------------------------------------------------------------------------------------
struct Tag {
    std::uint32_t edge = 0;   // 0 = untagged
    std::uint32_t idx = 0;
    bool operator==(const Tag& o) const { return edge == o.edge && idx == o.idx; }
};

// Value on (dir, stream) at position p, cycle T comes from the nearest upstream writer q on the
// same "diagonal": East D = T - p (writer at q, cycle t gives D = t + 1 - q), West D = T + p.
class DiagStreams {
public:
    explicit DiagStreams(int num_positions);
    // Returns false on a double write of the same SR in the same cycle.
    bool write(int pos, Direction dir, int stream, std::int64_t cycle, const Vector& v, Tag tag);
    // Committed value at the start of `cycle`; nullptr if nothing has been produced.
    const Vector* read(int pos, Direction dir, int stream, std::int64_t cycle, Tag* tag) const;

private:
    struct Entry {
        std::int16_t q;
        std::uint32_t slot;
        std::int64_t written;
        Tag tag;
    };
    struct Cell {
        std::int64_t diag = INT64_MIN;
        std::vector<Entry> entries;
    };
    static constexpr int kRing = 1024;
    static constexpr std::uint32_t kPool = 1u << 17;
    Cell& cell(Direction dir, int stream, std::int64_t diag);
    const Cell& cell(Direction dir, int stream, std::int64_t diag) const;

    int num_positions_;
    std::vector<Cell> cells_;              // [dir][stream][ring]
    std::vector<Vector> pool_;
    std::vector<std::int64_t> pool_cycle_;
    std::uint32_t next_slot_ = 0;
};

// ---------------------------------------------------------------------------------------------
// Programs
// ---------------------------------------------------------------------------------------------
// One instruction as the scheduler emits it: the lpu::Instruction plus simulator-only fields.
struct SimInstr {
    Instruction in;
    std::int32_t addr_stride = 1;            // MEM: address step per Repeat [A]
    std::uint32_t out_edge = 0;              // tag written on results: (out_edge, out_idx0 + r)
    std::uint32_t out_idx0 = 0;
    std::array<std::uint32_t, 4> in_edge{};  // tag each operand must carry: (in_edge, in_idx0 + r)
    std::array<std::uint32_t, 4> in_idx0{};
    std::int32_t op_id = -1;                 // scheduler op id (for traces)
    std::uint32_t text_off = 0, text_len = 0;  // Ifetch: byte range of the IQ's text it brings in
};

struct QEntry {
    enum Kind : std::uint8_t { Nop, Ins, Rep };
    Kind kind = Nop;
    std::uint32_t n = 0;   // Nop: cycles; Rep: repeat count
    std::uint32_t d = 1;   // Rep: cycles between repeats
    SimInstr ins;          // Ins only
    std::uint32_t byte_off = 0, byte_len = 0;   // where this entry sits in its IQ's text
};

// One program per IQ; each way of the IQ is its own NOP / instruction / Repeat thread.
struct IQProgram {
    std::vector<std::vector<QEntry>> ways;   // last way = fetch way (Ifetch only)
    std::vector<std::uint8_t> text;          // encoded instructions, in fetch order
    std::uint32_t preload = 0;               // bytes of `text` in the IQ before the Sync (boot load)
    std::uint32_t capacity = kIqBytes;       // IQ size in bytes [P col. 12]
};
struct Program {
    Program();
    std::vector<IQProgram> iqs;                       // kNumIQs
    std::vector<QEntry>& channel(int q);              // the way list that channel q maps to
    std::size_t num_entries() const;
    int iqs_used() const;
};

// Per-cycle activity counters for the power-style plot.
struct Activity {
    std::vector<std::uint8_t> abc, lw, vxm, mem_read, mem_write, sr_write;
    void bump(std::vector<std::uint8_t>& v, std::int64_t cycle, int n = 1);
};

class Simulator {
public:
    Simulator(const Floorplan& fp, const LatencyTable& lat);

    std::vector<Vector>& mem(int g) { return mem_[static_cast<std::size_t>(g)]; }
    void run(const Program& prog);

    std::int64_t last_cycle = 0;           // last cycle anything happened (superlane 0)
    std::uint64_t violations = 0;
    std::vector<std::string> violation_log;
    std::uint64_t issued = 0;
    std::uint64_t abc_issued = 0;
    Activity act;
    bool mem_single_issue = false;   // check MEM IQs as single-issue (sensitivity runs)
    // IQ text accounting (filled by run): fill level after every change, per IQ
    std::vector<std::vector<std::pair<std::int64_t, int>>> iq_fill;
    std::vector<int> iq_max_fill;
    std::uint64_t underflows = 0, overflows = 0, ifetch_vectors = 0;

private:
    struct Emit {
        int pos;
        Direction dir;
        int stream;
        Vector v;
        Tag tag;
    };
    struct PendingWrite {
        int g;
        int addr;
        Vector v;
    };
    struct Plane {
        std::vector<std::int8_t> w = std::vector<std::int8_t>(kLanes * kLanes, 0);    // [f][i]
        std::vector<std::int8_t> buf = std::vector<std::int8_t>(kLanes * kLanes, 0);
    };

    void violation(std::int64_t cycle, int q, const std::string& what);
    bool legal(int q, Opcode op) const;
    const Vector* operand(std::int64_t cycle, int q, const SimInstr& si, int k, int sub, int r);
    void emit(std::int64_t cycle, int pos, Direction dir, int stream, const Vector& v, Tag tag);
    void execute(std::int64_t cycle, int q, const SimInstr& si, int r);
    void exec_vxm(std::int64_t cycle, int q, const SimInstr& si, int r);

    const Floorplan& fp_;
    const LatencyTable& lat_;
    DiagStreams sr_;
    std::vector<std::vector<Vector>> mem_;
    std::array<Plane, kPlanes> planes_;
    static constexpr int kEmitRing = 64;
    std::array<std::vector<Emit>, kEmitRing> emits_;
    std::size_t pending_emits_ = 0;
    std::vector<PendingWrite> writes_;
    std::vector<std::int64_t> bank_read_cycle_, bank_write_cycle_;   // [g*2 + bank]
    std::vector<std::int64_t> iq_cycle_;          // last cycle an IQ issued
    std::vector<int> iq_r_, iq_w_, iq_n_;         // R-type / W-type / total issues in that cycle
    struct ByteEv { std::int64_t t; std::uint32_t end, len; };
    std::vector<std::vector<ByteEv>> consumed_, delivered_;   // per IQ
    std::vector<std::uint32_t> fetched_;                      // per IQ: bytes of text in order so far
    const Program* prog_ = nullptr;
    void check_text(const Program& prog);
};

}  // namespace sim
}  // namespace lpu
