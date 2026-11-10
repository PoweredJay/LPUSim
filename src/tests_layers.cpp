// tests_layers.cpp - steps 5-9 of the plan: compiled layers checked bit-exactly against the
// golden int8 reference (model.cpp) on the cycle simulator.
#include <chrono>
#include <cstdio>
#include <string>

#include "compiler.h"
#include "ifetch.h"
#include "model.h"
#include "testutil.h"

using namespace lpu;
using namespace lpu::sim;
using namespace lpu::sched;
using namespace lpu::comp;
using namespace lpu::model;

static bool run_net_(Net net, const std::string& title, std::vector<std::int8_t> input, bool image_input);
static bool run_net(Net net, const std::string& title, std::vector<std::int8_t> input, bool image_input = false) {
    try {
        return run_net_(net, title, input, image_input);
    } catch (const std::exception& e) {
        return check(title, false, std::string("exception: ") + e.what());
    }
}
static bool run_net_(Net net, const std::string& title, std::vector<std::int8_t> input, bool image_input) {
    auto t0 = std::chrono::steady_clock::now();
    GoldenResult gold = golden(net, input, true);
    Floorplan fp;
    LatencyTable lat;
    lat.load_csv("../../planning/instruction-timing-table.csv", nullptr);
    Scheduler sch(fp, lat);
    Simulator sim(fp, lat);
    CompileOptions opt;
    opt.free_tensors = false;   // keep every tensor so all of them can be compared
    Compiler cc(net, sim, sch, fp, opt);
    cc.load_input(image_input ? im2col_input(net, input) : input);
    cc.compile();
    IfetchStats ist;
    Program prog = build_program(cc, sch, sim, fp, IfetchOptions{}, &ist);   // bounded 640 B IQs + Ifetch
    sim.run(prog);
    int bad_tensors = 0;
    std::string first_bad;
    for (std::size_t t = 0; t < net.tensors.size(); ++t) {
        if (static_cast<int>(t) == net.input || net.tensors[t].int32 || gold.act[t].empty()) continue;
        const auto got = cc.read_tensor(static_cast<int>(t), sim);
        std::size_t diff = 0;
        for (std::size_t i = 0; i < got.size(); ++i) diff += got[i] != gold.act[t][i];
        if (diff) {
            ++bad_tensors;
            if (first_bad.empty()) first_bad = net.tensors[t].name + " (" + std::to_string(diff) + " values differ)";
        }
    }
    if (net.tensors[static_cast<std::size_t>(net.output)].int32) {
        const auto lg = cc.read_logits(sim);
        std::size_t diff = 0;
        for (std::size_t i = 0; i < lg.size(); ++i) diff += lg[i] != gold.logits[i];
        if (diff) { ++bad_tensors; if (first_bad.empty()) first_bad = "logits (" + std::to_string(diff) + ")"; }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const std::int64_t cycles = sim.last_cycle + 1 + lat.stagger;
    std::string d = "cycles " + std::to_string(cycles) + " (incl. 35 Sync + 19 stagger), ABC vectors " +
                    std::to_string(sim.abc_issued) + ", violations " + std::to_string(sim.violations) +
                    ", tensors wrong " + std::to_string(bad_tensors) + " " + first_bad + ", " +
                    std::to_string(static_cast<int>(secs * 1000)) + " ms";
    if (!sim.violation_log.empty()) d += "\n       first violation: " + sim.violation_log[0];
    return check(title, sim.violations == 0 && sim.underflows == 0 && sim.overflows == 0 && bad_tensors == 0, d);
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int fails = 0;
    {   // step 5: one 320x320 matmul over M = 256 vectors, requantize + ReLU
        NetBuilder b(1);
        int x = b.tensor("in", 16, 16, 320);
        b.conv("matmul", x, 320, 1, 1, 0, true);
        fails += !run_net(b.net, "5  matmul [256 x 320] x [320 x 320] + requant + ReLU", random_tensor(256 * 320, 2));
    }
    {   // step 5b: K = 640, N = 640 -> 4 tiles on 4 planes, VXM sums two K halves
        NetBuilder b(3);
        int x = b.tensor("in", 8, 16, 640);
        b.conv("matmul2", x, 640, 1, 1, 0, true);
        fails += !run_net(b.net, "5b matmul [128 x 640] x [640 x 640]: 4 tiles, K split summed in VXM", random_tensor(128 * 640, 4));
    }
    {   // step 5c: K = 2048 -> 7 K-vectors: partial sums through MEM across rounds
        NetBuilder b(5);
        int x = b.tensor("in", 4, 8, 2048);
        b.conv("matmul_k2048", x, 256, 1, 1, 0, false);
        fails += !run_net(b.net, "5c matmul K = 2048: int32 partial sums kept in MEM between rounds", random_tensor(32 * 2048, 6));
    }
    {   // step 6: 3x3/2 max pool
        NetBuilder b(7);
        int x = b.tensor("in", 16, 16, 64);
        b.maxpool("maxpool", x, 3, 2, 1);
        fails += !run_net(b.net, "6  max pool 3x3/2 on 16x16x64 (9 gathers + 8-op VXM max tree)", random_tensor(16 * 16 * 64, 8, 0, 127));
    }
    {   // step 7: 3x3 conv, 64 channels (dy-packed gathers)
        NetBuilder b(9);
        int x = b.tensor("in", 14, 14, 64);
        int y = b.conv("c1", x, 64, 1, 1, 0, true);
        b.conv("c2", y, 64, 3, 1, 1, true);
        fails += !run_net(b.net, "7  1x1 -> 3x3 conv, 64 ch, 14x14 (M-split + dy-packed gathers)", random_tensor(14 * 14 * 64, 10));
    }
    {   // step 7b: 3x3 stride 2, 128 channels
        NetBuilder b(11);
        int x = b.tensor("in", 14, 14, 256);
        int y = b.conv("c1", x, 128, 1, 1, 0, true);
        b.conv("c2", y, 128, 3, 2, 1, true);
        fails += !run_net(b.net, "7b 1x1 -> 3x3/2 conv, 128 ch", random_tensor(14 * 14 * 256, 12));
    }
    {   // step 8/9: a full ResNet bottleneck with downsample and residual add
        NetBuilder b(13);
        int x = b.tensor("in", 14, 14, 256);
        int res = b.conv("ds", x, 512, 1, 2, 0, false);
        int y = b.conv("c1", x, 128, 1, 1, 0, true);
        y = b.conv("c2", y, 128, 3, 2, 1, true);
        b.conv("c3", y, 512, 1, 1, 0, true, res);
        fails += !run_net(b.net, "9  bottleneck block 256->128->512, stride 2, downsample + residual", random_tensor(14 * 14 * 256, 14));
    }
    {   // stem: conv1 (host im2col) + max pool on a 64x64 image
        NetBuilder b(15);
        int x = b.conv1_im2col("conv1", 64, 64, 3, 64, 7, 2, 3);
        b.maxpool("maxpool", x, 3, 2, 1);
        fails += !run_net(b.net, "9  stem: conv1 7x7/2 + max pool on a 64x64 image", random_tensor(3 * 64 * 64, 16), true);
    }
    {   // FC alone
        NetBuilder b(19);
        int x = b.tensor("in", 1, 1, 2048);
        b.fc("fc", x, 1000);
        fails += !run_net(b.net, "9  FC 2048->1000 alone", random_tensor(2048, 20));
    }
    {   // head: avg pool + FC
        NetBuilder b(17);
        int x = b.tensor("in", 7, 7, 2048);
        x = b.avgpool("avgpool", x);
        b.fc("fc", x, 1000);
        fails += !run_net(b.net, "9  head: global avg pool 7x7x2048 + FC 2048->1000 (int32 logits)", random_tensor(49 * 2048, 18, 0, 127));
    }
    std::printf("\n%s (%d failed)\n", fails ? "FAILED" : "ALL PASSED", fails);
    return fails ? 1 : 0;
}
