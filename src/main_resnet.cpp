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
//          [--chunk B]        bytes per Ifetch (default 320)
//          [--lat name=d]     override one latency's d_func (mem_read, mem_write, mem_gather, vxm_add,
//                             vxm_mul, vxm_convert, vxm_relu, vxm_max, mxm_lw, mxm_iw, mxm_abc)
//          [--alias-weights]  if weights overflow MEM, reuse earlier tiles' slices: a timing-only
//                             projection (used for ResNet-101/152, which do not fit on one chip)
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "compiler.h"
#include "ifetch.h"
#include "model.h"

using namespace lpu;
using namespace lpu::sim;
using namespace lpu::sched;
using namespace lpu::comp;
using namespace lpu::model;

static const char* opcode_name(Opcode op) {
    static const char* names[] = {"Nop", "Repeat", "Sync", "Notify", "Ifetch", "Config", "Read", "Write", "Gather", "Scatter",
                                  "add", "sub", "mul", "max", "min", "mask", "negate", "relu", "tanh", "exp", "rsqrt",
                                  "convert", "LW", "IW", "ABC", "ACC", "shift_n", "shift_s", "select", "permute",
                                  "distribute", "rotate", "transpose"};
    const auto i = static_cast<std::size_t>(op);
    return i < sizeof(names) / sizeof(names[0]) ? names[i] : "?";
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int depth = 50;
    unsigned seed = 1;
    std::string out = "../../results", latcsv = "../../planning/instruction-timing-table.csv";
    bool check_golden = true, alias = false, mem1 = false, no_overlap = false;
    int sxm_hops = -1;
    std::string tag_suffix, events_dir;
    IfetchOptions iopt;
    std::vector<std::pair<std::string, int>> lat_over;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--depth") && i + 1 < argc) depth = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed") && i + 1 < argc) seed = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--latency") && i + 1 < argc) latcsv = argv[++i];
        else if (!std::strcmp(argv[i], "--no-golden")) check_golden = false;
        else if (!std::strcmp(argv[i], "--alias-weights")) alias = true;
        else if (!std::strcmp(argv[i], "--mem-single-issue")) mem1 = true;
        else if (!std::strcmp(argv[i], "--sxm-hops") && i + 1 < argc) sxm_hops = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--tag") && i + 1 < argc) tag_suffix = argv[++i];
        else if (!std::strcmp(argv[i], "--events") && i + 1 < argc) events_dir = argv[++i];
        else if (!std::strcmp(argv[i], "--no-layer-overlap")) no_overlap = true;
        else if (!std::strcmp(argv[i], "--chunk") && i + 1 < argc) iopt.chunk_bytes = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--lat") && i + 1 < argc) {
            const std::string a = argv[++i];
            const auto eq = a.find('=');
            if (eq == std::string::npos) { std::fprintf(stderr, "--lat wants name=value\n"); return 2; }
            lat_over.emplace_back(a.substr(0, eq), std::atoi(a.c_str() + eq + 1));
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    const std::string tag = "resnet" + std::to_string(depth) + tag_suffix;
    auto now = [] { return std::chrono::steady_clock::now(); };
    auto secs = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };

    auto t0 = now();
    Net net = build_resnet(depth, seed);
    const auto image = random_tensor(static_cast<std::size_t>(3 * 224 * 224), seed + 1000);
    GoldenResult gold = golden(net, image, true);   // calibrates Ma / S / Mr per layer
    auto t1 = now();
    std::printf("%s: %zu layers, golden int8 reference + calibration %.1f s\n", tag.c_str(), net.layers.size(), secs(t0, t1));

    Floorplan fp;
    if (sxm_hops >= 0) fp = Floorplan(sxm_hops);
    LatencyTable lat;
    std::string latlog;
    lat.load_csv(latcsv, &latlog);
    for (const auto& [name, d] : lat_over) {
        Lat* L = name == "mem_read" ? &lat.mem_read : name == "mem_write" ? &lat.mem_write
               : name == "mem_gather" ? &lat.mem_gather : name == "vxm_add" ? &lat.vxm_add
               : name == "vxm_mul" ? &lat.vxm_mul : name == "vxm_convert" ? &lat.vxm_convert
               : name == "vxm_relu" ? &lat.vxm_relu : name == "vxm_max" ? &lat.vxm_max
               : name == "mxm_lw" ? &lat.mxm_lw : name == "mxm_iw" ? &lat.mxm_iw
               : name == "mxm_abc" ? &lat.mxm_abc : nullptr;
        if (!L) { std::fprintf(stderr, "--lat: unknown latency %s\n", name.c_str()); return 2; }
        std::printf("latency override: %s d_func %d -> %d\n", name.c_str(), L->d_func, d);
        L->d_func = d;
        L->source = "override --lat";
    }
    Scheduler sch(fp, lat);
    sch.record_events = !events_dir.empty();
    Simulator sim(fp, lat);
    sch.mem_single_issue = sim.mem_single_issue = mem1;
    CompileOptions opt;
    opt.alias_weights_when_full = alias;
    opt.mem_single_issue = mem1;
    opt.no_layer_overlap = no_overlap;
    Compiler cc(net, sim, sch, fp, opt);
    cc.load_input(im2col_input(net, image));
    cc.compile();
    const std::int64_t compute_end = sch.makespan();
    IfetchStats ist;
    Program prog = build_program(cc, sch, sim, fp, iopt, &ist);
    auto t2 = now();
    std::printf("compiled + scheduled: %llu pipelines, %zu IQ entries on %d of %d IQs, %llu search steps, %.1f s\n",
                static_cast<unsigned long long>(sch.placed()), prog.num_entries(), prog.iqs_used(), kNumIQs,
                static_cast<unsigned long long>(sch.search_steps()), secs(t1, t2));
    if (cc.weights_aliased)
        std::printf("PROJECTION: %llu of %llu weight tiles did not fit in MEM and alias earlier tiles (timing only)\n",
                    static_cast<unsigned long long>(cc.weights_aliased), static_cast<unsigned long long>(cc.tiles_total));
    std::printf("Ifetch: %d Ifetches on %d of %d IQs with code; text %llu B (%llu B loaded at boot, %llu B fetched) in %llu MEM words; "
                    "%llu read-port cycles, %llu stream-register cycles; min slack %lld; %d late, %d shrunk\n",
                    ist.ifetches, ist.iqs_fetching, ist.iqs_used, static_cast<unsigned long long>(ist.text_bytes),
                    static_cast<unsigned long long>(ist.preload_bytes), static_cast<unsigned long long>(ist.fetched_bytes),
                    static_cast<unsigned long long>(ist.text_words), static_cast<unsigned long long>(ist.read_port_cycles),
                    static_cast<unsigned long long>(ist.stream_cycles), static_cast<long long>(ist.ifetches ? ist.min_slack : 0),
                    ist.late, ist.shrunk);
    std::printf("Ifetch planner: %d candidate slices skipped (their own IQ text could not take the Read)\n", ist.rejected);
    std::printf("MEM: weights %llu words (%llu tiles), maps %llu words, peak runtime %llu words of %d\n",
                static_cast<unsigned long long>(cc.weight_words), static_cast<unsigned long long>(cc.tiles_total),
                static_cast<unsigned long long>(cc.map_words), static_cast<unsigned long long>(cc.peak_words),
                kMemSlices * kMemWords);

    sim.run(prog);
    auto t3 = now();
    const std::int64_t cycles = sim.last_cycle + 1 + lat.stagger;   // includes the 35-cycle Sync preamble
    std::printf("simulated %lld cycles in %.1f s: %llu instructions issued, %llu MXM vector ops, %llu violations\n",
                static_cast<long long>(cycles), secs(t2, t3), static_cast<unsigned long long>(sim.issued),
                static_cast<unsigned long long>(sim.abc_issued), static_cast<unsigned long long>(sim.violations));
    for (const auto& v : sim.violation_log) std::printf("  violation: %s\n", v.c_str());
    int max_fill = 0;
    for (int f : sim.iq_max_fill) max_fill = std::max(max_fill, f);
    std::printf("IQ text: %llu underflows, %llu overflows, max fill %d B, %llu Ifetch vectors\n",
                static_cast<unsigned long long>(sim.underflows), static_cast<unsigned long long>(sim.overflows), max_fill,
                static_cast<unsigned long long>(sim.ifetch_vectors));

    std::size_t wrong = 0;
    if (check_golden) {
        const auto lg = cc.read_logits(sim);
        for (std::size_t i = 0; i < lg.size(); ++i) wrong += lg[i] != gold.logits[i];
        if (cc.weights_aliased) std::printf("(weights aliased: a mismatch is expected)\n");
        std::printf("logits vs golden int8 reference: %zu of %zu differ -> %s\n", wrong, lg.size(),
                    wrong == 0 ? "BIT-EXACT" : "MISMATCH");
    }

    std::uint64_t macs = 0;
    for (const auto& s : cc.stats) macs += s.macs;
    const double lat_us_1g = static_cast<double>(cycles) / 1000.0;
    const double lat_us_09 = static_cast<double>(cycles) / 900.0;
    std::printf("latency %.1f us @ 1 GHz (%.0f IPS), %.1f us @ 0.9 GHz (%.0f IPS); %.2f GMAC, %.1f%% of 4-plane peak\n",
                lat_us_1g, 1e6 / lat_us_1g, lat_us_09, 1e6 / lat_us_09, macs / 1e9,
                100.0 * static_cast<double>(macs) / (static_cast<double>(cycles) * 4 * kLanes * kLanes));

    // ---- results ----
    {
        std::ofstream f(out + "/" + tag + "_layers.csv");
        f << "index,name,kind,first_cycle,last_cycle,macs,abc_vectors,rounds,tiles,kvecs,out_words,chunks\n";
        for (std::size_t i = 0; i < cc.stats.size(); ++i) {
            const auto& s = cc.stats[i];
            const char* kind = net.layers[i].kind == LayerDesc::Conv ? (net.layers[i].raw32 ? "fc" : "conv")
                               : net.layers[i].kind == LayerDesc::MaxPool ? "maxpool" : "avgpool";
            f << i << ',' << s.name << ',' << kind << ',' << s.first + lat.sync_cycles << ',' << s.last + lat.sync_cycles << ','
              << s.macs << ',' << s.abc_vectors << ',' << s.rounds << ',' << s.tiles << ',' << s.kvecs << ','
              << s.nwords << ',' << s.chunks << '\n';
        }
    }
    {
        std::ofstream f(out + "/" + tag + "_trace.csv");
        f << "iq,way,channel,unit,pos,start,n,kind,layer,label\n";
        for (const auto& o : sch.trace())
            f << channel_iq(o.queue) << ',' << channel_way(o.queue) << ',' << o.queue << ','
              << fp.iq_name(channel_iq(o.queue)) << ',' << o.pos << ',' << o.start + lat.sync_cycles << ',' << o.n
              << ',' << o.kind << ',' << o.layer << ",\"" << o.label << "\"\n";
    }
    {
        std::ofstream f(out + "/" + tag + "_activity.csv");
        f << "cycle,mxm_abc_planes,mxm_lw,vxm_alus,mem_reads,mem_writes,sr_writes\n";
        auto at = [](const std::vector<std::uint8_t>& v, std::size_t c) { return c < v.size() ? int(v[c]) : 0; };
        for (std::size_t c = 0; c < static_cast<std::size_t>(cycles); ++c)
            f << c << ',' << at(sim.act.abc, c) << ',' << at(sim.act.lw, c) << ',' << at(sim.act.vxm, c) << ','
              << at(sim.act.mem_read, c) << ',' << at(sim.act.mem_write, c) << ',' << at(sim.act.sr_write, c) << '\n';
    }
    {
        std::ofstream f(out + "/" + tag + "_iqbytes.csv");
        f << "iq,unit,program_bytes,preload_bytes,max_fill\n";
        for (int iq = 0; iq < kNumIQs; ++iq)
            f << iq << ',' << fp.iq_name(iq) << ',' << ist.program_bytes[static_cast<std::size_t>(iq)] << ','
              << prog.iqs[static_cast<std::size_t>(iq)].preload << ','
              << (static_cast<std::size_t>(iq) < sim.iq_max_fill.size() ? sim.iq_max_fill[static_cast<std::size_t>(iq)] : 0) << '\n';
    }
    {
        std::ofstream f(out + "/" + tag + "_iqfill.csv");   // occupancy steps, chip cycles
        f << "iq,unit,cycle,bytes\n";
        for (std::size_t iq = 0; iq < sim.iq_fill.size(); ++iq)
            for (const auto& [t, b] : sim.iq_fill[iq]) f << iq << ',' << fp.iq_name(static_cast<int>(iq)) << ',' << t << ',' << b << '\n';
    }
    {
        std::ofstream f(out + "/" + tag + "_summary.json");
        f << "{\n  \"model\": \"" << tag << "\",\n  \"cycles\": " << cycles << ",\n  \"sync_preamble\": " << lat.sync_cycles
          << ",\n  \"stagger_tail\": " << lat.stagger << ",\n  \"latency_us_1GHz\": " << lat_us_1g
          << ",\n  \"latency_us_0.9GHz\": " << lat_us_09 << ",\n  \"ips_1GHz\": " << 1e6 / lat_us_1g
          << ",\n  \"ips_0.9GHz\": " << 1e6 / lat_us_09 << ",\n  \"macs\": " << macs
          << ",\n  \"mxm_vector_ops\": " << sim.abc_issued << ",\n  \"instructions_issued\": " << sim.issued
          << ",\n  \"iq_entries\": " << prog.num_entries() << ",\n  \"iqs_total\": " << kNumIQs
          << ",\n  \"iqs_used\": " << prog.iqs_used() << ",\n  \"violations\": " << sim.violations
          << ",\n  \"golden_checked\": " << (check_golden ? "true" : "false") << ",\n  \"logits_wrong\": " << wrong
          << ",\n  \"weight_words\": " << cc.weight_words << ",\n  \"weight_tiles\": " << cc.tiles_total
          << ",\n  \"weight_tiles_aliased\": " << cc.weights_aliased << ",\n  \"timing_only_projection\": "
          << (cc.weights_aliased ? "true" : "false") << ",\n  \"map_words\": " << cc.map_words << ",\n  \"peak_runtime_words\": " << cc.peak_words
          << ",\n  \"compute_makespan\": " << compute_end
          << ",\n  \"ifetches\": " << ist.ifetches << ",\n  \"iqs_fetching\": " << ist.iqs_fetching
          << ",\n  \"text_bytes\": " << ist.text_bytes << ",\n  \"preload_bytes\": " << ist.preload_bytes
          << ",\n  \"fetched_bytes\": " << ist.fetched_bytes << ",\n  \"text_words\": " << ist.text_words
          << ",\n  \"ifetch_read_port_cycles\": " << ist.read_port_cycles << ",\n  \"ifetch_stream_cycles\": " << ist.stream_cycles
          << ",\n  \"ifetch_late\": " << ist.late << ",\n  \"ifetch_min_slack\": " << (ist.ifetches ? ist.min_slack : 0)
          << ",\n  \"iq_underflows\": " << sim.underflows << ",\n  \"iq_overflows\": " << sim.overflows
          << ",\n  \"iq_max_fill\": " << max_fill
          << ",\n  \"paper_latency_us\": 49,\n  \"paper_ips\": " << (depth == 50 ? 20400 : depth == 101 ? 14300 : 10700)
          << "\n}\n";
    }
    if (!events_dir.empty()) {
        // All cycles below are chip cycles (Sync preamble included), like the trace CSV.
        const std::int64_t sy = lat.sync_cycles;
        auto sg = [](std::ostream& f, const StreamGroup& g) {
            if (!g.used()) f << ",,,";
            else f << ',' << int(g.base) << ',' << int(g.width) << ',' << (g.dir == Direction::East ? 'E' : 'W');
        };
        {
            std::ofstream f(events_dir + "/" + tag + "_events.csv");
            f << "id,cycle,iq,way,channel,unit,opcode,pos,n,d_func,kind,layer,pipe,"
                 "src0_base,src0_w,src0_dir,src1_base,src1_w,src1_dir,dst_base,dst_w,dst_dir,"
                 "mem_slice,bank,addr_lo,addr_hi,stride,label\n";
            for (std::size_t i = 0; i < sch.ev_ops.size(); ++i) {
                const EventOp& e = sch.ev_ops[i];
                const int iq = channel_iq(e.queue);
                f << i << ',' << e.start + sy << ',' << iq << ',' << channel_way(e.queue) << ',' << e.queue << ','
                  << fp.iq_name(iq) << ',' << opcode_name(e.opcode) << ',' << e.pos << ',' << e.n << ',' << e.d_func << ','
                  << e.kind << ',' << e.layer << ',' << e.pipe;
                sg(f, e.src[0]);
                sg(f, e.src[1]);
                sg(f, e.dst);
                f << ',';
                if (e.mem_g >= 0) f << fp.iq_name(iq_mem(e.mem_g));
                f << ',';
                if (e.bank >= 0) f << e.bank;
                f << ',';
                if (e.addr_lo >= 0) f << e.addr_lo;
                f << ',';
                if (e.addr_hi >= 0) f << e.addr_hi;
                f << ',' << e.stride << ",\"" << e.label << "\"\n";
            }
        }
        {
            std::ofstream f(events_dir + "/" + tag + "_streams.csv");
            f << "pipe,edge,producer,dir,stream,d0,d1,p0,p1\n";
            for (const auto& r : sch.ev_streams) {
                const bool east = r.dir == Direction::East;
                // shift diagonals to chip time: East D = t - p, West D = t + p
                f << r.pipe << ',' << r.edge << ',' << r.producer << ',' << (east ? 'E' : 'W') << ',' << r.stream << ','
                  << r.d0 + sy << ',' << r.d1 + sy << ',' << r.p0 << ',' << r.p1 << '\n';
            }
        }
        {
            std::ofstream f(events_dir + "/" + tag + "_pipes.csv");
            f << "pipe,start,layer,name\n";
            for (std::size_t i = 0; i < sch.ev_pipes.size(); ++i)
                f << i << ',' << sch.ev_pipes[i].S + sy << ',' << sch.ev_pipes[i].layer << ",\"" << sch.ev_pipes[i].name << "\"\n";
        }
        {
            std::ofstream f(events_dir + "/" + tag + "_alloc.csv");
            f << "id,kind,bank,base,size,pipe_alloc,pipe_free,slices,what\n";
            for (std::size_t i = 0; i < cc.alloc_log.size(); ++i) {
                const auto& a = cc.alloc_log[i];
                f << i << ',' << a.kind << ',' << a.bank << ',' << a.base << ',' << a.size << ',' << a.pipe_alloc << ','
                  << a.pipe_free << ",\"";
                for (std::size_t k = 0; k < a.slices.size(); ++k) f << (k ? " " : "") << fp.iq_name(iq_mem(a.slices[k]));
                f << "\",\"" << a.what << "\"\n";
            }
        }
        {
            // the emitted IQ programs, entry by entry (op_id = row of _events.csv)
            std::ofstream f(events_dir + "/" + tag + "_programs.csv");
            f << "iq,way,idx,entry,n,d,op_id\n";
            for (int iq = 0; iq < kNumIQs; ++iq) {
                const auto& ways = prog.iqs[static_cast<std::size_t>(iq)].ways;
                for (std::size_t w = 0; w < ways.size(); ++w)
                    for (std::size_t k = 0; k < ways[w].size(); ++k) {
                        const QEntry& q = ways[w][k];
                        f << iq << ',' << w << ',' << k << ',' << (q.kind == QEntry::Nop ? 'N' : q.kind == QEntry::Ins ? 'I' : 'R') << ','
                          << q.n << ',' << q.d << ',' << (q.kind == QEntry::Ins ? q.ins.op_id : -1) << '\n';
                    }
            }
        }
        std::printf("events: %zu ops, %zu stream rects, %zu pipelines, %zu allocations -> %s\n", sch.ev_ops.size(),
                    sch.ev_streams.size(), sch.ev_pipes.size(), cc.alloc_log.size(), events_dir.c_str());
    }
    lat.dump_csv(out + "/latency_used.csv");
    return (sim.violations == 0 && sim.underflows == 0 && sim.overflows == 0 && (wrong == 0 || cc.weights_aliased)) ? 0 : 1;
}
