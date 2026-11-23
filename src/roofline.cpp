// roofline.cpp - Fig. 9-style sweep: arithmetic throughput vs offered load (ops per weight byte)
// for [M x 320] x [320 x 5120] int8 matmuls (16 weight tiles, 4 rounds on the 4 planes), with
// the weights placed in three MEM regions. Every point is also checked bit-exact.
//   roofline [--out ../results] [--m 4,8,16] [--csv roofline.csv]
//            [--latency ../planning/instruction-timing-table.csv]
//            [--place "name:wlo:whi:alo:ahi"]   run one custom placement instead of the three
//                                               standard ones (MEM slice index ranges per hemisphere)
// Exit status 1 if any point is not bit-exact or has a violation or IQ underflow.
#include <cstdio>
#include <cstdlib>
#include <vector>
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

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string out = "../results", csv = "roofline.csv", latency_csv = "../planning/instruction-timing-table.csv";
    std::vector<int> ms;
    for (int M = 1; M <= 2048; M *= 2) ms.push_back(M);
    struct Place { std::string name; int wlo, whi, alo, ahi; };
    std::vector<Place> places = {
        {"weights next to MXM (MEM28-43)", 28, 43, 0, 27},
        {"weights mid-hemisphere (MEM14-29)", 14, 29, 30, 43},
        {"weights next to VXM (MEM0-15)", 0, 15, 16, 43},
    };
    for (int i = 1; i < argc; ++i)
        if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!std::strcmp(argv[i], "--m") && i + 1 < argc) {   // comma-separated M values
            ms.clear();
            for (const char* q = argv[++i]; *q;) {
                ms.push_back(std::atoi(q));
                while (*q && *q != ',') ++q;
                if (*q) ++q;
            }
        } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!std::strcmp(argv[i], "--latency") && i + 1 < argc) latency_csv = argv[++i];
        else if (!std::strcmp(argv[i], "--place") && i + 1 < argc) {
            const std::string a = argv[++i];
            const auto c = a.find(':');
            Place p{a.substr(0, c), 0, 0, 0, 0};
            if (c == std::string::npos ||
                std::sscanf(a.c_str() + c + 1, "%d:%d:%d:%d", &p.wlo, &p.whi, &p.alo, &p.ahi) != 4) {
                std::fprintf(stderr, "--place wants name:wlo:whi:alo:ahi\n");
                return 2;
            }
            places = {p};
        } else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    {
        LatencyTable probe;
        if (!probe.load_csv(latency_csv, nullptr))
            std::fprintf(stderr, "roofline: %s not found, using the built-in default latencies\n", latency_csv.c_str());
    }
    int failures = 0;
    const int N = 16 * kLanes;
    std::ofstream f(out + "/" + csv);
    if (!f) {
        std::fprintf(stderr, "roofline: cannot write %s/%s (does the directory exist?)\n", out.c_str(), csv.c_str());
        return 2;
    }
    f << "placement,M,ops_per_weight_byte,cycles,ops_per_cycle,tops_1ghz,peak_tops_1ghz,bit_exact,violations,ifetches,iq_underflows\n";
    for (const Place& pl : places) {
        for (int M : ms) {
            NetBuilder b(42);
            const int x = b.tensor("act", M, 1, kLanes);
            b.conv("gemm", x, N, 1, 1, 0, false);
            Net net = b.net;
            const auto input = random_tensor(static_cast<std::size_t>(M) * kLanes, 7);
            GoldenResult gold = golden(net, input, true);
            Floorplan fp;
            LatencyTable lat;
            lat.load_csv(latency_csv, nullptr);
            Scheduler sch(fp, lat);
            Simulator sim(fp, lat);
            CompileOptions opt;
            opt.weight_index_lo = pl.wlo;
            opt.weight_index_hi = pl.whi;
            opt.act_index_lo = pl.alo;
            opt.act_index_hi = pl.ahi;
            Compiler cc(net, sim, sch, fp, opt);
            cc.load_input(input);
            cc.compile();
            IfetchStats ist;
            sim.run(build_program(cc, sch, sim, fp, IfetchOptions{}, &ist));   // bounded IQs + Ifetch
            const auto got = cc.read_tensor(net.output, sim);
            std::size_t diff = 0;
            for (std::size_t i = 0; i < got.size(); ++i) diff += got[i] != gold.act[static_cast<std::size_t>(net.output)][i];
            // cycles: first instruction to last result write of the layer (no Sync preamble)
            const std::int64_t cyc = cc.stats[0].last - cc.stats[0].first;
            const double ops = 2.0 * M * kLanes * N;
            const double opc = ops / static_cast<double>(cyc);
            f << '"' << pl.name << "\"," << M << ',' << 2 * M << ',' << cyc << ',' << opc << ',' << opc / 1000.0 << ','
              << 4.0 * 2 * kLanes * kLanes / 1000.0 << ',' << (diff == 0 ? 1 : 0) << ',' << sim.violations << ',' << ist.ifetches << ',' << sim.underflows << '\n';
            const bool ok = diff == 0 && sim.violations == 0 && sim.underflows == 0;
            failures += !ok;
            std::printf("%-36s M=%5d  ops/B=%5d  cycles=%6lld  %6.1f TOPS  %s\n", pl.name.c_str(), M, 2 * M,
                        static_cast<long long>(cyc), opc / 1000.0, ok ? "ok" : "MISMATCH");
        }
    }
    return failures ? 1 : 0;
}
