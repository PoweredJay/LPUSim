// main_resnet.cpp - compile ResNet-50/101/152 for the LPU, run it on the cycle simulator,
// check the logits bit-exactly against the golden int8 model, and write results/ CSVs.
//
//   resnet [--depth 50] [--seed 1] [--out ../../results] [--latency ../../planning/instruction-timing-table.csv]
//          [--no-golden]      skip the reference check (timing only)
//          [--mem-single-issue]  sensitivity: a MEM IQ issues one instruction per cycle
//          [--sxm-hops N]     sensitivity: SXM width in hops per side (default 4)
//          [--tag S]          suffix for the output file names
//          [--events DIR]     also write the visualization event export (DIR/<tag>_events.csv,
//                             _streams.csv, _alloc.csv, _pipes.csv); does not change any result
//          [--no-layer-overlap]  baseline: each layer waits for the previous layer's last result
//          [--chunk B]        max bytes per refill (default 640 = Ifetch + Repeat 1)
//          [--lat name=d]     override one latency's d_func (mem_read, mem_write, mem_gather, vxm_add,
//                             vxm_mul, vxm_convert, vxm_relu, vxm_max, mxm_lw, mxm_iw, mxm_abc)
//          [--alias-weights]  if weights overflow MEM, reuse earlier tiles' slices: a timing-only
//                             projection (used for ResNet-101/152, which do not fit on one chip)
//
// The run has six steps, one function each: parse_args, build_reference, compile + build_program,
// simulate, check_logits, and the report (print_performance + report::write_*). The real work
// lives in model.h (network + golden), compiler.h / sched.h (lowering + scheduling), ifetch.h
// (IQ programs + refills) and lpusim.h (the cycle simulator); report.h writes the files.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "compiler.h"
#include "ifetch.h"
#include "model.h"
#include "report.h"

using namespace lpu;
using namespace lpu::sim;
using namespace lpu::sched;
using namespace lpu::comp;
using namespace lpu::model;

namespace {

using ull = unsigned long long;   // printf %llu
template <class T> ull U(T v) { return static_cast<ull>(v); }

// ---- step 1: options ----

struct RunOptions {
    int depth = 50;
    unsigned seed = 1;
    std::string out = "../../results";
    std::string latency_csv = "../../planning/instruction-timing-table.csv";
    std::string tag_suffix, events_dir;
    bool check_golden = true;
    bool alias_weights = false;
    bool mem_single_issue = false;
    bool no_layer_overlap = false;
    int sxm_hops = -1;                                         // -1 = Floorplan default
    IfetchOptions ifetch;
    std::vector<std::pair<std::string, int>> lat_overrides;   // --lat name=d_func

    std::string tag() const { return "resnet" + std::to_string(depth) + tag_suffix; }
};

bool parse_args(int argc, char** argv, RunOptions* o) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto is = [&](const char* flag) { return !std::strcmp(a, flag); };
        auto has_value = [&](const char* flag) { return is(flag) && i + 1 < argc; };
        if (has_value("--depth")) o->depth = std::atoi(argv[++i]);
        else if (has_value("--seed")) o->seed = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (has_value("--out")) o->out = argv[++i];
        else if (has_value("--latency")) o->latency_csv = argv[++i];
        else if (is("--no-golden")) o->check_golden = false;
        else if (is("--alias-weights")) o->alias_weights = true;
        else if (is("--mem-single-issue")) o->mem_single_issue = true;
        else if (has_value("--sxm-hops")) o->sxm_hops = std::atoi(argv[++i]);
        else if (has_value("--tag")) o->tag_suffix = argv[++i];
        else if (has_value("--events")) o->events_dir = argv[++i];
        else if (is("--no-layer-overlap")) o->no_layer_overlap = true;
        else if (has_value("--chunk")) o->ifetch.chunk_bytes = std::atoi(argv[++i]);
        else if (has_value("--lat")) {
            const std::string v = argv[++i];
            const auto eq = v.find('=');
            if (eq == std::string::npos) { std::fprintf(stderr, "--lat wants name=value\n"); return false; }
            o->lat_overrides.emplace_back(v.substr(0, eq), std::atoi(v.c_str() + eq + 1));
        } else {
            std::fprintf(stderr, "unknown option %s\n", a);
            return false;
        }
    }
    return true;
}

// Latency table from the CSV, then the --lat overrides.
bool load_latencies(const RunOptions& o, LatencyTable* lat) {
    std::string log;
    lat->load_csv(o.latency_csv, &log);
    for (const auto& [name, d] : o.lat_overrides) {
        Lat* L = lat->find(name);
        if (!L) { std::fprintf(stderr, "--lat: unknown latency %s\n", name.c_str()); return false; }
        std::printf("latency override: %s d_func %d -> %d\n", name.c_str(), L->d_func, d);
        L->d_func = d;
        L->source = "override --lat";
    }
    return true;
}

// ---- step 2: network + golden reference ----

struct Reference {
    Net net;
    std::vector<std::int8_t> image;   // 3 x 224 x 224, CHW
    GoldenResult gold;
};

Reference build_reference(const RunOptions& o) {
    Reference r;
    r.net = build_resnet(o.depth, o.seed);
    r.image = random_tensor(static_cast<std::size_t>(3 * 224 * 224), o.seed + 1000);
    r.gold = golden(r.net, r.image, true);   // calibrates Ma / S / Mr per layer (stored in r.net)
    return r;
}

// ---- step 3: compile + schedule + Ifetch planning ----

CompileOptions compile_options(const RunOptions& o) {
    CompileOptions c;
    c.alias_weights_when_full = o.alias_weights;
    c.mem_single_issue = o.mem_single_issue;
    c.no_layer_overlap = o.no_layer_overlap;
    return c;
}

void print_compile_report(const Compiler& cc, const Scheduler& sch, const Program& prog, const IfetchStats& ist,
                          double secs) {
    std::printf("compiled + scheduled: %llu pipelines, %zu IQ entries on %d of %d IQs, %llu search steps, %.1f s\n",
                U(sch.placed()), prog.num_entries(), prog.iqs_used(), kNumIQs, U(sch.search_steps()), secs);
    if (cc.weights_aliased)
        std::printf("PROJECTION: %llu of %llu weight tiles did not fit in MEM and alias earlier tiles (timing only)\n",
                    U(cc.weights_aliased), U(cc.tiles_total));
    std::printf("Ifetch: %d refills (%llu text vectors) on %d of %d IQs with code; text %llu B (%llu B loaded at boot, "
                "%llu B fetched) in %llu MEM words; %llu read-port cycles, %llu stream-register cycles; "
                "min slack %lld; %d late, %d shrunk\n",
                ist.ifetches, U(ist.ifetch_vectors), ist.iqs_fetching, ist.iqs_used, U(ist.text_bytes),
                U(ist.preload_bytes), U(ist.fetched_bytes), U(ist.text_words), U(ist.read_port_cycles),
                U(ist.stream_cycles), static_cast<long long>(ist.ifetches ? ist.min_slack : 0), ist.late, ist.shrunk);
    std::printf("Ifetch planner: %d candidate slices skipped (their own IQ text could not take the Read)\n", ist.rejected);
    std::printf("MEM: weights %llu words (%llu tiles), maps %llu words, peak runtime %llu words of %d\n",
                U(cc.weight_words), U(cc.tiles_total), U(cc.map_words), U(cc.peak_words), kMemSlices * kMemWords);
}

// ---- step 4: simulate ----

// Total chip cycles. The program already starts with the Sync preamble; only superlane 0 is
// simulated, and superlane 19 finishes `stagger` cycles after it.
std::int64_t simulate(Simulator& sim, const Program& prog, const LatencyTable& lat) {
    sim.run(prog);
    return sim.last_cycle + 1 + lat.stagger;
}

void print_sim_report(const Simulator& sim, std::int64_t cycles, double secs) {
    std::printf("simulated %lld cycles in %.1f s: %llu instructions issued, %llu MXM vector ops, %llu violations\n",
                static_cast<long long>(cycles), secs, U(sim.issued), U(sim.abc_issued), U(sim.violations));
    for (const auto& v : sim.violation_log) std::printf("  violation: %s\n", v.c_str());
    int max_fill = 0;
    for (int f : sim.iq_max_fill) max_fill = std::max(max_fill, f);
    std::printf("IQ text: %llu underflows, %llu overflows, max fill %d B, %llu Ifetch vectors\n", U(sim.underflows),
                U(sim.overflows), max_fill, U(sim.ifetch_vectors));
}

// ---- step 5: correctness ----

// Number of logits that differ from the golden reference.
std::size_t check_logits(const Compiler& cc, Simulator& sim, const GoldenResult& gold) {
    const auto lg = cc.read_logits(sim);
    std::size_t wrong = 0;
    for (std::size_t i = 0; i < lg.size(); ++i) wrong += lg[i] != gold.logits[i];
    if (cc.weights_aliased) std::printf("(weights aliased: a mismatch is expected)\n");
    std::printf("logits vs golden int8 reference: %zu of %zu differ -> %s\n", wrong, lg.size(),
                wrong == 0 ? "BIT-EXACT" : "MISMATCH");
    return wrong;
}

// ---- step 6: report ----

void print_performance(const report::Run& r) {
    const double peak_macs = static_cast<double>(r.cycles) * kPlanes * kLanes * kLanes;
    std::printf("latency %.1f us @ 1 GHz (%.0f IPS), %.1f us @ 0.9 GHz (%.0f IPS); %.2f GMAC, %.1f%% of 4-plane peak\n",
                r.latency_us(1000.0), r.ips(1000.0), r.latency_us(900.0), r.ips(900.0), r.macs / 1e9,
                100.0 * static_cast<double>(r.macs) / peak_macs);
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    RunOptions o;
    if (!parse_args(argc, argv, &o)) return 2;
    const std::string tag = o.tag();
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

    // 2. network + golden reference
    auto t0 = now();
    const Reference ref = build_reference(o);
    auto t1 = now();
    std::printf("%s: %zu layers, golden int8 reference + calibration %.1f s\n", tag.c_str(), ref.net.layers.size(),
                secs(t0, t1));

    // 3. machine model, compile, schedule, Ifetch
    const Floorplan fp = o.sxm_hops >= 0 ? Floorplan(o.sxm_hops) : Floorplan();
    LatencyTable lat;
    if (!load_latencies(o, &lat)) return 2;
    Scheduler sch(fp, lat);
    Simulator sim(fp, lat);   // also receives the initial MEM image (weights, maps, input) from the compiler
    sch.record_events = !o.events_dir.empty();
    sch.mem_single_issue = sim.mem_single_issue = o.mem_single_issue;
    Compiler cc(ref.net, sim, sch, fp, compile_options(o));
    cc.load_input(im2col_input(ref.net, ref.image));
    cc.compile();
    const std::int64_t compute_end = sch.makespan();
    IfetchStats ist;
    const Program prog = build_program(cc, sch, sim, fp, o.ifetch, &ist);
    auto t2 = now();
    print_compile_report(cc, sch, prog, ist, secs(t1, t2));

    // 4. simulate
    const std::int64_t cycles = simulate(sim, prog, lat);
    auto t3 = now();
    print_sim_report(sim, cycles, secs(t2, t3));

    // 5. correctness
    const std::size_t wrong = o.check_golden ? check_logits(cc, sim, ref.gold) : 0;

    // 6. report
    std::uint64_t macs = 0;
    for (const auto& s : cc.stats) macs += s.macs;
    const report::Run run{ref.net, cc,     sch,    sim,         prog, ist,           fp,   lat,
                          tag,     o.depth, cycles, compute_end, macs, o.check_golden, wrong};
    print_performance(run);
    report::write_results(o.out, run);
    if (!o.events_dir.empty()) {
        report::write_events(o.events_dir, run);
        std::printf("events: %zu ops, %zu stream rects, %zu pipelines, %zu allocations -> %s\n", sch.ev_ops.size(),
                    sch.ev_streams.size(), sch.ev_pipes.size(), cc.alloc_log.size(), o.events_dir.c_str());
    }
    lat.dump_csv(o.out + "/latency_used.csv");

    const bool timing_clean = sim.violations == 0 && sim.underflows == 0 && sim.overflows == 0;
    const bool logits_ok = wrong == 0 || cc.weights_aliased;   // aliased weights: timing-only, logits meaningless
    return timing_clean && logits_ok ? 0 : 1;
}
