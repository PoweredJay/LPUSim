// model.cpp - see model.h.
#include "model.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>

namespace lpu {
namespace model {

std::vector<std::int8_t> random_tensor(std::size_t n, unsigned seed, int lo, int hi) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(lo, hi);
    std::vector<std::int8_t> v(n);
    for (auto& x : v) x = static_cast<std::int8_t>(d(rng));
    return v;
}

std::vector<std::int8_t> NetBuilder::rand_w(std::size_t n) {
    return random_tensor(n, seed_ * 7919u + (counter_++) * 104729u + 1u, -100, 100);
}

int NetBuilder::tensor(const std::string& name, int H, int W, int C, bool int32) {
    TensorDesc t;
    t.name = name;
    t.H = H;
    t.W = W;
    t.C = C;
    t.int32 = int32;
    net.tensors.push_back(t);
    const int id = static_cast<int>(net.tensors.size()) - 1;
    if (net.input < 0) net.input = id;
    net.output = id;
    return id;
}

int NetBuilder::conv(const std::string& name, int in, int cout, int k, int stride, int pad, bool relu, int res) {
    const TensorDesc d = net.tensors[static_cast<std::size_t>(in)];
    const int Ho = (d.H + 2 * pad - k) / stride + 1, Wo = (d.W + 2 * pad - k) / stride + 1;
    const int out = tensor(name, Ho, Wo, cout);
    LayerDesc L;
    L.kind = LayerDesc::Conv;
    L.name = name;
    L.in = in;
    L.out = out;
    L.res = res;
    L.k = k;
    L.stride = stride;
    L.pad = pad;
    L.cin = d.C;
    L.cout = cout;
    L.relu = relu;
    L.w = rand_w(static_cast<std::size_t>(cout) * d.C * k * k);
    net.layers.push_back(L);
    return out;
}

int NetBuilder::maxpool(const std::string& name, int in, int k, int stride, int pad) {
    const TensorDesc d = net.tensors[static_cast<std::size_t>(in)];
    const int out = tensor(name, (d.H + 2 * pad - k) / stride + 1, (d.W + 2 * pad - k) / stride + 1, d.C);
    LayerDesc L;
    L.kind = LayerDesc::MaxPool;
    L.name = name;
    L.in = in;
    L.out = out;
    L.k = k;
    L.stride = stride;
    L.pad = pad;
    L.cin = L.cout = d.C;
    net.layers.push_back(L);
    return out;
}

int NetBuilder::avgpool(const std::string& name, int in) {
    const TensorDesc d = net.tensors[static_cast<std::size_t>(in)];
    const int out = tensor(name, 1, 1, d.C);
    LayerDesc L;
    L.kind = LayerDesc::AvgPool;
    L.name = name;
    L.in = in;
    L.out = out;
    L.cin = L.cout = d.C;
    L.relu = false;
    net.layers.push_back(L);
    return out;
}

int NetBuilder::fc(const std::string& name, int in, int cout) {
    const TensorDesc d = net.tensors[static_cast<std::size_t>(in)];
    const int out = tensor(name, 1, 1, cout, true);
    LayerDesc L;
    L.kind = LayerDesc::Conv;
    L.name = name;
    L.in = in;
    L.out = out;
    L.k = 1;
    L.cin = d.C;
    L.cout = cout;
    L.relu = false;
    L.raw32 = true;
    L.w = rand_w(static_cast<std::size_t>(cout) * d.C);
    net.layers.push_back(L);
    return out;
}

int NetBuilder::conv1_im2col(const std::string& name, int ih, int iw, int ic, int cout, int k, int stride, int pad) {
    const int Ho = (ih + 2 * pad - k) / stride + 1, Wo = (iw + 2 * pad - k) / stride + 1;
    net.image_h = ih;
    net.image_w = iw;
    net.image_c = ic;
    const int K = k * k * ic;
    const int in = tensor("im2col", Ho, Wo, K);
    const int out = tensor(name, Ho, Wo, cout);
    LayerDesc L;
    L.kind = LayerDesc::Conv;
    L.name = name;
    L.in = in;
    L.out = out;
    L.k = 1;
    L.cin = K;
    L.cout = cout;
    L.relu = true;
    L.w_orig = rand_w(static_cast<std::size_t>(cout) * ic * k * k);
    L.k_orig = k;
    L.stride_orig = stride;
    L.pad_orig = pad;
    L.w.assign(static_cast<std::size_t>(cout) * K, 0);
    for (int co = 0; co < cout; ++co)
        for (int c = 0; c < ic; ++c)
            for (int ky = 0; ky < k; ++ky)
                for (int kx = 0; kx < k; ++kx)
                    L.w[static_cast<std::size_t>(co * K + (ky * k + kx) * ic + c)] =
                        L.w_orig[static_cast<std::size_t>(((co * ic + c) * k + ky) * k + kx)];
    net.layers.push_back(L);
    return out;
}

Net build_resnet(int depth, unsigned seed) {
    std::vector<int> blocks;
    if (depth == 50) blocks = {3, 4, 6, 3};
    else if (depth == 101) blocks = {3, 4, 23, 3};
    else if (depth == 152) blocks = {3, 8, 36, 3};
    else throw std::runtime_error("depth must be 50, 101 or 152");
    NetBuilder b(seed);
    int x = b.conv1_im2col("conv1", 224, 224, 3, 64, 7, 2, 3);
    x = b.maxpool("maxpool", x, 3, 2, 1);
    const int widths[4] = {64, 128, 256, 512};
    for (int s = 0; s < 4; ++s)
        for (int i = 0; i < blocks[static_cast<std::size_t>(s)]; ++i) {
            const std::string n = "l" + std::to_string(s + 1) + "." + std::to_string(i);
            const int stride = (i == 0 && s > 0) ? 2 : 1;
            const int w = widths[s];
            int res = x;
            if (i == 0) res = b.conv(n + ".ds", x, 4 * w, 1, stride, 0, false);
            int y = b.conv(n + ".c1", x, w, 1, 1, 0, true);
            y = b.conv(n + ".c2", y, w, 3, stride, 1, true);
            x = b.conv(n + ".c3", y, 4 * w, 1, 1, 0, true, res);
        }
    x = b.avgpool("avgpool", x);
    b.fc("fc", x, 1000);
    return b.net;
}

std::vector<std::int8_t> im2col_input(const Net& net, const std::vector<std::int8_t>& image) {
    const LayerDesc& L = net.layers[0];
    const TensorDesc& t = net.tensors[static_cast<std::size_t>(L.in)];
    const int k = L.k_orig, s = L.stride_orig, pad = L.pad_orig, ic = net.image_c;
    std::vector<std::int8_t> out(static_cast<std::size_t>(t.C * t.H * t.W), 0);
    for (int ky = 0; ky < k; ++ky)
        for (int kx = 0; kx < k; ++kx)
            for (int c = 0; c < ic; ++c) {
                const int i = (ky * k + kx) * ic + c;
                for (int y = 0; y < t.H; ++y)
                    for (int x = 0; x < t.W; ++x) {
                        const int yy = y * s + ky - pad, xx = x * s + kx - pad;
                        if (yy < 0 || yy >= net.image_h || xx < 0 || xx >= net.image_w) continue;
                        out[static_cast<std::size_t>((i * t.H + y) * t.W + x)] =
                            image[static_cast<std::size_t>((c * net.image_h + yy) * net.image_w + xx)];
                    }
            }
    return out;
}

namespace {
constexpr std::int64_t kI32 = 2147483647LL;

void calibrate(LayerDesc& L, const std::vector<std::int64_t>& acc, bool has_res) {
    std::vector<std::int64_t> a;
    a.reserve(acc.size());
    std::int64_t mx = 1;
    for (std::int64_t v : acc) { a.push_back(static_cast<std::int64_t>(std::llabs(v))); mx = std::max<std::int64_t>(mx, std::llabs(v)); }
    std::nth_element(a.begin(), a.begin() + static_cast<long>(a.size() * 999 / 1000), a.end());
    const double p = std::max<double>(1.0, static_cast<double>(a[a.size() * 999 / 1000]));
    const double target = has_res ? 60.0 : 90.0;
    for (int S = 24; S >= 1; --S) {
        const std::int64_t Ma = std::max<std::int64_t>(1, std::llround(target * std::ldexp(1.0, S) / p));
        const std::int64_t Mr = has_res ? std::llround(0.5 * std::ldexp(1.0, S)) : 0;
        const std::int64_t worst = mx * Ma + 128 * Mr + (1LL << (S - 1));
        if (mx * Ma <= kI32 && 128 * Mr <= kI32 && worst <= kI32) {
            L.S = S;
            L.Ma = static_cast<std::int32_t>(Ma);
            L.Mr = static_cast<std::int32_t>(Mr);
            return;
        }
    }
    throw std::runtime_error("calibration failed for " + L.name);
}

std::int8_t requant(std::int64_t acc, std::int64_t res, const LayerDesc& L) {
    const std::int64_t a = acc * L.Ma, r = res * L.Mr;
    const std::int64_t s = a + r + (L.S > 0 ? (1LL << (L.S - 1)) : 0);
    if (std::llabs(a) > kI32 || std::llabs(r) > kI32 || std::llabs(a + r) > kI32 || std::llabs(s) > kI32)
        throw std::runtime_error("int32 overflow in requantize of " + L.name);
    std::int64_t y = s >> L.S;
    y = std::max<std::int64_t>(-128, std::min<std::int64_t>(127, y));
    if (L.relu) y = std::max<std::int64_t>(0, y);
    return static_cast<std::int8_t>(y);
}
}  // namespace

GoldenResult golden(Net& net, const std::vector<std::int8_t>& input, bool calib) {
    GoldenResult R;
    R.act.resize(net.tensors.size());
    const bool image_in = !net.layers.empty() && !net.layers[0].w_orig.empty();
    if (!image_in) R.act[static_cast<std::size_t>(net.input)] = input;
    for (LayerDesc& L : net.layers) {
        const TensorDesc& to = net.tensors[static_cast<std::size_t>(L.out)];
        const TensorDesc& ti = net.tensors[static_cast<std::size_t>(L.in)];
        const std::size_t no = static_cast<std::size_t>(to.C * to.H * to.W);
        if (L.kind == LayerDesc::Conv) {
            std::vector<std::int64_t> acc(no, 0);
            if (!L.w_orig.empty()) {
                // real 7x7/2 convolution of the image
                const int k = L.k_orig, s = L.stride_orig, pad = L.pad_orig, ic = net.image_c;
                for (int co = 0; co < L.cout; ++co)
                    for (int y = 0; y < to.H; ++y)
                        for (int x = 0; x < to.W; ++x) {
                            std::int64_t sum = 0;
                            for (int c = 0; c < ic; ++c)
                                for (int ky = 0; ky < k; ++ky) {
                                    const int yy = y * s + ky - pad;
                                    if (yy < 0 || yy >= net.image_h) continue;
                                    for (int kx = 0; kx < k; ++kx) {
                                        const int xx = x * s + kx - pad;
                                        if (xx < 0 || xx >= net.image_w) continue;
                                        sum += static_cast<std::int64_t>(L.w_orig[static_cast<std::size_t>(((co * ic + c) * k + ky) * k + kx)]) *
                                               input[static_cast<std::size_t>((c * net.image_h + yy) * net.image_w + xx)];
                                    }
                                }
                            acc[static_cast<std::size_t>((co * to.H + y) * to.W + x)] = sum;
                        }
            } else {
                const auto& in = R.act[static_cast<std::size_t>(L.in)];
                const int k = L.k, s = L.stride, pad = L.pad;
                std::vector<std::int32_t> col(static_cast<std::size_t>(to.H * to.W));
                for (int co = 0; co < L.cout; ++co) {
                    std::fill(col.begin(), col.end(), 0);
                    for (int ci = 0; ci < L.cin; ++ci)
                        for (int ky = 0; ky < k; ++ky)
                            for (int kx = 0; kx < k; ++kx) {
                                const int wv = L.w[static_cast<std::size_t>(((co * L.cin + ci) * k + ky) * k + kx)];
                                if (wv == 0) continue;
                                const std::int8_t* ip = &in[static_cast<std::size_t>(ci * ti.H * ti.W)];
                                for (int y = 0; y < to.H; ++y) {
                                    const int yy = y * s + ky - pad;
                                    if (yy < 0 || yy >= ti.H) continue;
                                    for (int x = 0; x < to.W; ++x) {
                                        const int xx = x * s + kx - pad;
                                        if (xx < 0 || xx >= ti.W) continue;
                                        col[static_cast<std::size_t>(y * to.W + x)] += wv * ip[yy * ti.W + xx];
                                    }
                                }
                            }
                    for (int i = 0; i < to.H * to.W; ++i) acc[static_cast<std::size_t>(co * to.H * to.W + i)] = col[static_cast<std::size_t>(i)];
                }
            }
            if (L.raw32) {
                R.logits.resize(no);
                for (std::size_t i = 0; i < no; ++i) {
                    if (std::llabs(acc[i]) > kI32) throw std::runtime_error("logit overflow");
                    R.logits[i] = static_cast<std::int32_t>(acc[i]);
                }
                continue;
            }
            if (calib) calibrate(L, acc, L.res >= 0);
            std::vector<std::int8_t> out(no);
            const std::vector<std::int8_t>* res = L.res >= 0 ? &R.act[static_cast<std::size_t>(L.res)] : nullptr;
            for (std::size_t i = 0; i < no; ++i) out[i] = requant(acc[i], res ? (*res)[i] : 0, L);
            R.act[static_cast<std::size_t>(L.out)] = std::move(out);
        } else if (L.kind == LayerDesc::MaxPool) {
            const auto& in = R.act[static_cast<std::size_t>(L.in)];
            std::vector<std::int8_t> out(no);
            for (int c = 0; c < to.C; ++c)
                for (int y = 0; y < to.H; ++y)
                    for (int x = 0; x < to.W; ++x) {
                        int m = -128;
                        for (int ky = 0; ky < L.k; ++ky)
                            for (int kx = 0; kx < L.k; ++kx) {
                                const int yy = y * L.stride + ky - L.pad, xx = x * L.stride + kx - L.pad;
                                // padding taps read the zero word on the TSP -> they count as 0
                                const int v = (yy < 0 || yy >= ti.H || xx < 0 || xx >= ti.W)
                                                  ? 0 : in[static_cast<std::size_t>((c * ti.H + yy) * ti.W + xx)];
                                m = std::max(m, v);
                            }
                        out[static_cast<std::size_t>((c * to.H + y) * to.W + x)] = static_cast<std::int8_t>(m);
                    }
            R.act[static_cast<std::size_t>(L.out)] = std::move(out);
        } else {
            const auto& in = R.act[static_cast<std::size_t>(L.in)];
            std::vector<std::int64_t> acc(no, 0);
            for (int c = 0; c < to.C; ++c)
                for (int i = 0; i < ti.H * ti.W; ++i) acc[static_cast<std::size_t>(c)] += in[static_cast<std::size_t>(c * ti.H * ti.W + i)];
            if (calib) calibrate(L, acc, false);
            std::vector<std::int8_t> out(no);
            for (std::size_t i = 0; i < no; ++i) out[i] = requant(acc[i], 0, L);
            R.act[static_cast<std::size_t>(L.out)] = std::move(out);
        }
    }
    return R;
}

}  // namespace model
}  // namespace lpu
