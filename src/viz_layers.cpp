// viz_layers.cpp - write the layer shapes of a ResNet for the schedule explorer.
//   viz_layers [--depth 50] [--out ../../results/viz]  ->  <out>/resnet<depth>_layerdesc.csv
// One row per layer, in the same order as resnet<depth>_layers.csv. Shapes only: no compile, no run.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "model.h"

int main(int argc, char** argv) {
    int depth = 50;
    std::string out = "../../results/viz";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--depth") && i + 1 < argc) depth = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
    }
    const lpu::comp::Net net = lpu::model::build_resnet(depth, 1);
    const std::string path = out + "/resnet" + std::to_string(depth) + "_layerdesc.csv";
    std::ofstream f(path);
    f << "index,name,kind,k,stride,pad,cin,cout,in,in_h,in_w,in_c,out,out_h,out_w,out_c,res,relu\n";
    for (std::size_t i = 0; i < net.layers.size(); ++i) {
        const auto& L = net.layers[i];
        const auto& ti = net.tensors[static_cast<std::size_t>(L.in)];
        const auto& to = net.tensors[static_cast<std::size_t>(L.out)];
        const char* kind = L.kind == lpu::comp::LayerDesc::Conv ? (L.raw32 ? "fc" : "conv")
                           : L.kind == lpu::comp::LayerDesc::MaxPool ? "maxpool" : "avgpool";
        f << i << ',' << L.name << ',' << kind << ',' << (L.k_orig ? L.k_orig : L.k) << ',' << (L.k_orig ? L.stride_orig : L.stride)
          << ',' << (L.k_orig ? L.pad_orig : L.pad) << ',' << L.cin << ',' << L.cout << ',' << ti.name << ',' << ti.H << ','
          << ti.W << ',' << ti.C << ',' << to.name << ',' << to.H << ',' << to.W << ',' << to.C << ','
          << (L.res >= 0 ? net.tensors[static_cast<std::size_t>(L.res)].name : "") << ',' << (L.relu ? 1 : 0) << '\n';
    }
    std::printf("%zu layers -> %s\n", net.layers.size(), path.c_str());
    return 0;
}
