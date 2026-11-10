// compiler.h - lowers a quantized CNN graph (conv / maxpool / residual / avgpool / FC) onto the
// LPU and drives the scheduler. Output: per-queue programs + the initial MEM image (weights,
// gather maps, input) written straight into the simulator's memory.
//
// Lowering choices (all [A] unless noted; see src/scheduler/README.md):
//  * Activations are int8, one 320-byte MEM word per pixel per 320-channel group. Layout per
//    tensor is picked from its consumers: "standard" (channels at lane 0) or "dy-packed" for
//    3x3 consumers: the channel block is stored 2-3 times at different superlane offsets so a
//    single Gather can assemble an im2col vector holding the (dy, dx = -1, 0, +1) taps of one
//    row (the MXM writes the copies for free by duplicating weight columns).
//  * Gather = per-superlane address from a map stream (Table I "Gather s map"); maps are
//    compile-time constants stored in MEM.
//  * Conv = sum over K-vectors v of [M x 320] x [320 x 320] tiles (v, j); j = output word.
//    Tiles run on the 4 MXM planes in "rounds": up to 4 (v, j, M-chunk) slots per round, all
//    streaming at once in one rigid pipeline. Results of one output are summed in the VXM;
//    an output whose K-vectors span several rounds keeps an int32 partial sum in MEM.
//  * Requantize in the VXM: y = relu(sat8((acc*Ma + res*Mr + 2^(S-1)) >> S)).
//  * Weights: LW (16 streams x 20 cycles) into the plane's buffer while the previous tile
//    computes, then a 1-cycle IW (T5). Tensors that are read by several planes at once are
//    written to several MEM slices (replicas), because each slice has one read port.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "lpusim.h"
#include "sched.h"

namespace lpu {
namespace comp {

struct TensorDesc {
    std::string name;
    int H = 1, W = 1, C = 1;
    bool int32 = false;          // logits
};

struct LayerDesc {
    enum Kind { Conv, MaxPool, AvgPool } kind = Conv;
    std::string name;
    int in = -1, out = -1, res = -1;
    int k = 1, stride = 1, pad = 0;
    int cin = 0, cout = 0;
    bool relu = true;
    bool raw32 = false;          // FC: write int32 sums, no requantize
    std::int32_t Ma = 1, Mr = 0;
    int S = 0;
    std::vector<std::int8_t> w;  // [cout][cin][k][k]
    // conv1 only: original 7x7 weights for the golden model ([cout][3][7][7])
    std::vector<std::int8_t> w_orig;
    int k_orig = 0, stride_orig = 0, pad_orig = 0;
};

struct Net {
    std::vector<TensorDesc> tensors;
    std::vector<LayerDesc> layers;
    int input = -1;              // host-provided tensor (im2col of the image for conv1)
    int output = -1;
    int image_h = 224, image_w = 224, image_c = 3;
};

struct CompileOptions {
    // MEM slice index ranges per hemisphere (0 = next to the VXM, 43 = next to the MXM).
    // Weights get their own slices so the next tile's LW never waits for a read port that a
    // long activation stream holds.
    int weight_index_lo = 28;
    int weight_index_hi = 43;
    int act_index_lo = 0;        // activations, maps, partial sums: act_index_lo..act_index_hi
    int act_index_hi = 27;
    bool free_tensors = true;    // reuse MEM of dead activations
    bool alias_weights_when_full = false;   // timing-only projection when weights do not fit
    bool mem_single_issue = false;   // sensitivity: one instruction per MEM IQ per cycle
    bool no_layer_overlap = false;   // baseline: a layer starts only after the previous layer's last result
    int round_overhead = 40;     // cost model used to choose M-split
    bool verbose = false;
};

struct LayerStats {
    std::string name;
    int index = 0;
    std::int64_t first = INT64_MAX, last = 0;   // first issue, last completion (program cycles)
    std::uint64_t macs = 0;                     // useful MACs
    std::uint64_t abc_vectors = 0;              // MXM plane-cycles
    int rounds = 0, tiles = 0, chunks = 1, kvecs = 0, nwords = 0;
};

struct Layout {
    struct Word {
        int ch0 = 0, nch = 0;
        std::vector<int> copies;   // superlane offsets
        int sl() const { return (nch + 15) / 16; }
    };
    std::vector<Word> words;
};

class Compiler {
public:
    Compiler(const Net& net, sim::Simulator& loader, sched::Scheduler& sch, const sim::Floorplan& fp,
             CompileOptions opt = {});

    // host input: CHW int8 tensor matching net.tensors[net.input] (for conv1: im2col, C=147)
    void load_input(const std::vector<std::int8_t>& chw);
    void compile();

    // Read a tensor back from the simulator after the run (copy 0 of replica 0).
    std::vector<std::int8_t> read_tensor(int t, sim::Simulator& s) const;
    std::vector<std::int32_t> read_logits(sim::Simulator& s) const;
    bool tensor_alive(int t) const { return ts_[static_cast<std::size_t>(t)].allocated && !ts_[static_cast<std::size_t>(t)].freed; }

    std::vector<LayerStats> stats;
    std::uint64_t weight_words = 0, map_words = 0, peak_words = 0, tiles_total = 0;
    std::uint64_t weights_aliased = 0;   // tiles that did not fit (alias_weights_when_full)
    std::string log;

    // Allocation map (visualization export). One record per store: kind W weight tile,
    // M gather map, A activation tensor, P partial sum / pooling partial. pipe_alloc / pipe_free
    // are scheduler pipeline counts (placement order) at allocation and at free; -1 = never freed.
    struct AllocRec {
        std::string what;
        char kind;
        std::vector<int> slices;
        int bank, base, size;
        long pipe_alloc, pipe_free = -1;
    };
    std::vector<AllocRec> alloc_log;

    // Instruction text for Ifetch: `words` never-used MEM words on slice g (a constant, so only
    // clean space - never space runtime data used). commit = false only checks for room.
    bool alloc_text(int g, int words, int* bank, int* base, bool commit);
    std::uint64_t text_words = 0;

private:
    std::string alloc_what_;
    char alloc_kind_ = 'A';
    struct KItem { int dy, dx, word, sl; };
    struct KVec { std::vector<KItem> items; bool identity = false; };
    struct Slot { int v, j, chunk; };
    struct Plan {
        std::vector<KVec> kv;
        std::vector<std::pair<int, int>> chunks;   // output row ranges
        std::vector<std::vector<Slot>> rounds;
        int slots_per_round = 1;
    };
    struct Store { std::vector<int> slices; int base = 0; int bank = 0; int size = 0; };
    struct TState {
        Layout lay;
        int R = 1;
        std::vector<std::pair<int, int>> pieces;   // row ranges
        std::vector<std::vector<Store>> store;     // [piece][word]
        bool allocated = false, freed = false;
        int consumers_left = 0;
        bool host = false;
    };
    struct PlaneState { std::int64_t last_iw = -1, abc_end = 0; long tile = -1; };

    // planning
    Layout layout_for(int t) const;
    std::vector<KVec> kvecs(const LayerDesc& L, const Layout& in) const;
    int alu_need(const LayerDesc& L, const std::vector<Slot>& slots, int nk) const;
    std::vector<std::vector<Slot>> pack_rounds(const LayerDesc& L, int nk, int nc, int nchunks) const;
    static std::vector<std::pair<int, int>> split_rows(int H, int c);
    void plan();

    // memory
    using FreeList = std::vector<std::array<std::vector<std::pair<int, int>>, 2>>;   // [slice][bank]
    static int alloc_common(const FreeList& fl, const std::vector<int>& slices, int size, int bank);
    std::vector<int> pick_slices(bool west, int count, int size, int bank, const std::vector<int>& exclude,
                                 int lo, int hi, std::vector<int>& use, int* base, bool in_range_only = false);
    void free_store(const Store& s);
    void alloc_tensor(int t, const std::vector<int>& exclude, int bank);
    int out_bank(const LayerDesc& L, std::vector<int>* exclude) const;
    void free_tensor(int t);
    Store weight_tile(int layer, int v, int j, bool west);

    // lowering
    void lower_conv(int li);
    void lower_maxpool(int li);
    void lower_avgpool(int li);
    int piece_of_row(const TState& T, int row) const;
    // Per-pipeline bookkeeping so two ops of one rigid pipeline never share a MEM port or
    // read+write one bank of one slice.
    struct PipeCtx {
        std::vector<int> rd, wr;                        // slices with a read / write op
        std::vector<std::pair<int, int>> rd_bank, wr_bank;
        bool single = false;                            // MEM IQ single-issue (sensitivity)
        bool can_read(int g, int bank) const;
        bool can_write(int g, int bank) const;
        void read(int g, int bank) { rd.push_back(g); rd_bank.push_back({g, bank}); }
        void write(int g, int bank) { wr.push_back(g); wr_bank.push_back({g, bank}); }
    };
    // Streams activation vectors [0, n) of one slot onto act_edge. yy[m] = input row of
    // vector m (all taps of one vector share a row); xx[m*20 + t] = input column for
    // superlane t, or -1 for a zero superlane. identity = 1x1 stride-1 -> plain Read.
    void add_act_source(sched::Pipeline& p, PipeCtx& ctx, int act_edge, int tin, int word, int rep,
                        int n, const std::vector<int>& yy, const std::vector<int>& xx, bool identity,
                        int layer, const std::string& tag);
    int choose_replica(const TState& T, int piece, int word, int pref, const PipeCtx& ctx) const;
    int pick_planes(int count, std::vector<int>* planes);
    void note(const sched::Pipeline& p, int li);

    const Net& net_;
    sim::Simulator& sim_;
    sched::Scheduler& sch_;
    const sim::Floorplan& fp_;
    CompileOptions opt_;
    std::vector<TState> ts_;
    std::vector<Plan> plans_;
    FreeList free_clean_;   // never used: constants and runtime data
    FreeList free_rt_;      // freed runtime space: runtime data only
    std::vector<int> act_use_, w_use_, map_use_;
    int map_use_side_[2] = {0, 0};
    std::map<long, Store> wtiles_;
    std::array<PlaneState, sim::kPlanes> planes_;
    std::uint64_t live_words_ = 0;
};

}  // namespace comp
}  // namespace lpu
