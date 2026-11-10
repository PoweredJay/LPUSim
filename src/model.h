// model.h - network builders (ResNet-50/101/152 and small test nets), random int8 weights,
// per-layer requantization calibration, and an independent golden int8 reference.
//
// The golden model computes every layer directly from its definition (direct convolution over
// CHW tensors, conv1 as a real 7x7/2 convolution of the image), so it shares no code with the
// compiler's im2col / gather / tiling path. Requantization is defined as
//     y = sat8((acc * Ma + res * Mr + 2^(S-1)) >> S), then ReLU if the layer has one,
// evaluated in int64 and checked to stay inside int32 (so it equals the VXM's int32 math).
// Weights are random (no pretrained checkpoint is used), so accuracy is not measured; the check
// is bit-exact agreement between the cycle simulator and this reference.
#pragma once

#include <cstdint>
#include <vector>

#include "compiler.h"

namespace lpu {
namespace model {

using comp::LayerDesc;
using comp::Net;
using comp::TensorDesc;

class NetBuilder {
public:
    explicit NetBuilder(unsigned seed) : seed_(seed) {}
    int tensor(const std::string& name, int H, int W, int C, bool int32 = false);
    // conv with random weights; res = residual tensor or -1
    int conv(const std::string& name, int in, int cout, int k, int stride, int pad, bool relu, int res = -1);
    int maxpool(const std::string& name, int in, int k, int stride, int pad);
    int avgpool(const std::string& name, int in);
    int fc(const std::string& name, int in, int cout);
    // conv1 of ResNet: 7x7/2 on the image, lowered as a 1x1 conv over a host-side im2col tensor
    int conv1_im2col(const std::string& name, int image_h, int image_w, int image_c, int cout, int k,
                     int stride, int pad);
    Net net;

private:
    std::vector<std::int8_t> rand_w(std::size_t n);
    unsigned seed_;
    unsigned counter_ = 0;
};

Net build_resnet(int depth, unsigned seed);

// Host-side im2col of the image for conv1 ([147][112][112] for ResNet).
std::vector<std::int8_t> im2col_input(const Net& net, const std::vector<std::int8_t>& image);

struct GoldenResult {
    std::vector<std::vector<std::int8_t>> act;   // per tensor (CHW), empty if not int8
    std::vector<std::int32_t> logits;            // net.output if int32
};

// Runs the reference. With calibrate = true it first chooses Ma / S / Mr for every layer.
// `input` is the image (if the net starts with conv1_im2col) or the input tensor (CHW).
GoldenResult golden(Net& net, const std::vector<std::int8_t>& input, bool calibrate);

std::vector<std::int8_t> random_tensor(std::size_t n, unsigned seed, int lo = -128, int hi = 127);

}  // namespace model
}  // namespace lpu
