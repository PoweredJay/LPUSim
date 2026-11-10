// lpusim.cpp - see lpusim.h.
#include "lpusim.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace lpu {
namespace sim {

// ---------------------------------------------------------------------------------------------
// Floorplan
// ---------------------------------------------------------------------------------------------
Floorplan::Floorplan(int sxm_hops_) : sxm_hops(sxm_hops_) {
    int p = 0;
    mxm_west_pos = p++;
    const int sxm_w0 = p;
    p += sxm_hops;                                   // SXM_W
    for (int i = kMemPerHem - 1; i >= 0; --i) mem_pos[static_cast<std::size_t>(i)] = p++;   // MEM_W43..W0
    vxm_pos = p++;
    for (int i = 0; i < kMemPerHem; ++i) mem_pos[static_cast<std::size_t>(kMemPerHem + i)] = p++;
    const int sxm_e0 = p;
    p += sxm_hops;                                   // SXM_E
    mxm_east_pos = p++;
    num_positions = p;
    // The 14 SXM IQs of a side are spread over the SXM's sxm_hops positions, west to east [A].
    for (int k = 0; k < kSxmIqsPerHem; ++k) {
        sxm_pos[static_cast<std::size_t>(k)] = sxm_w0 + k * sxm_hops / kSxmIqsPerHem;
        sxm_pos[static_cast<std::size_t>(kSxmIqsPerHem + k)] = sxm_e0 + k * sxm_hops / kSxmIqsPerHem;
    }
}

IqKind iq_kind(int iq) {
    if (iq < kIqSxmW) return IqKind::Mxm;
    if (iq < kIqMemW) return IqKind::Sxm;
    if (iq < kIqVxm) return IqKind::Mem;
    if (iq < kIqMemE) return IqKind::Vxm;
    if (iq < kIqSxmE) return IqKind::Mem;
    if (iq < kIqMxmE) return IqKind::Sxm;
    return IqKind::Mxm;
}

int channel_iq(int q) {
    if (q_is_fetch(q)) return q - kQFetchBase;
    if (q_is_mem(q)) return iq_mem(q_mem_slice(q));
    if (q_is_vxm(q)) return iq_vxm(q - kQVxmBase);
    if (q_is_mxm(q)) return iq_mxm(q_plane(q), q_mxm_role(q));
    return iq_sxm(q_sxm_west(q), q_sxm_index(q));
}
int channel_way(int q) {
    if (q_is_fetch(q)) return iq_fetch_way(q - kQFetchBase);
    return q_is_mem(q) ? (q & 1) : 0;
}
int channel_of(int iq, int way) {
    static const std::vector<int> table = [] {
        std::vector<int> t(static_cast<std::size_t>(3 * kNumIQs), -1);
        for (int q = 0; q < kNumChannels; ++q) t[static_cast<std::size_t>(3 * channel_iq(q) + channel_way(q))] = q;
        return t;
    }();
    return table[static_cast<std::size_t>(3 * iq + way)];
}

int Floorplan::queue_pos(int q) const {
    if (q_is_fetch(q)) return queue_pos(channel_of(q - kQFetchBase, 0));
    if (q_is_mem(q)) return mem_pos[static_cast<std::size_t>(q_mem_slice(q))];
    if (q_is_vxm(q)) return vxm_pos;
    if (q_is_mxm(q)) return plane_pos(q_plane(q));
    return sxm_pos[static_cast<std::size_t>((q_sxm_west(q) ? 0 : kSxmIqsPerHem) + q_sxm_index(q))];
}

int Floorplan::iq_pos(int iq) const { return queue_pos(channel_of(iq, 0)); }

std::string Floorplan::iq_name(int iq) const {
    char buf[32];
    const int q = channel_of(iq, 0);
    switch (iq_kind(iq)) {
        case IqKind::Mem: {
            const int g = q_mem_slice(q);
            std::snprintf(buf, sizeof buf, "MEM_%c%d", mem_west(g) ? 'W' : 'E', mem_index(g));
            break;
        }
        case IqKind::Vxm: std::snprintf(buf, sizeof buf, "VXM.alu%d", q - kQVxmBase); break;
        case IqKind::Mxm: {
            static const char* role[] = {"lw", "iw", "abc"};
            const int p = q_plane(q);
            std::snprintf(buf, sizeof buf, "MXM_%c%d.%s", plane_west(p) ? 'W' : 'E', p % 2, role[q_mxm_role(q)]);
            break;
        }
        case IqKind::Sxm: std::snprintf(buf, sizeof buf, "SXM_%c%d", q_sxm_west(q) ? 'W' : 'E', q_sxm_index(q)); break;
    }
    return buf;
}

std::string Floorplan::queue_name(int q) const {
    std::string s = iq_name(channel_iq(q));
    if (q_is_fetch(q)) s += ".fetch";
    else if (q_is_mem(q)) s += (q & 1) ? ".W" : ".R";
    return s;
}

// ---------------------------------------------------------------------------------------------
// Latency table
// ---------------------------------------------------------------------------------------------
namespace {
std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool quoted = false;
    for (char c : line) {
        if (c == '"') {
            quoted = !quoted;
        } else if (c == ',' && !quoted) {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

bool parse_int(const std::string& s, int* v) {
    if (s.empty()) return false;
    std::size_t i = 0;
    if (s[0] == '-') i = 1;
    if (i >= s.size()) return false;
    for (std::size_t k = i; k < s.size(); ++k)
        if (s[k] < '0' || s[k] > '9') return false;
    *v = std::stoi(s);
    return true;
}
}  // namespace

bool LatencyTable::load_csv(const std::string& path, std::string* log) {
    std::ifstream f(path);
    if (!f) {
        if (log) *log += "latency table: " + path + " not found, using defaults\n";
        return false;
    }
    struct Key {
        const char* unit;
        const char* instr;
        const char* dtype;   // nullptr = any
        Lat* lat;
    };
    const Key keys[] = {
        {"MEM", "Read a s", nullptr, &mem_read},
        {"MEM", "Write a s", nullptr, &mem_write},
        {"MEM", "Gather s map", nullptr, &mem_gather},
        {"VXM", "add (binary)", "int32", &vxm_add},
        {"VXM", "mul (binary)", "int32", &vxm_mul},
        {"VXM", "convert int32->int8 (shift, round, sat)", nullptr, &vxm_convert},
        {"VXM", "ReLU", "int8", &vxm_relu},
        {"VXM", "max (binary)", "int8", &vxm_max},
        {"MXM", "LW", nullptr, &mxm_lw},
        {"MXM", "IW (from LW buffer)", nullptr, &mxm_iw},
        {"MXM", "ABC (matmul)", "int8", &mxm_abc},
    };
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line)) {
        const auto c = split_csv(line);
        if (c.size() < 13) continue;
        for (const Key& k : keys) {
            if (c[0] != k.unit || c[1] != k.instr) continue;
            if (k.dtype && c[2] != k.dtype) continue;
            int v = 0;
            bool any = false;
            if (parse_int(c[3], &v)) { k.lat->d_func = v; any = true; }
            if (parse_int(c[4], &v)) { k.lat->d_skew = v; any = true; }
            if (parse_int(c[5], &v)) { k.lat->occupancy = v; any = true; }
            if (any) {
                k.lat->tag = c[12].empty() ? "A" : c[12];
                k.lat->source = "instruction-timing-table.xlsx (" + c[1] + (c[2].empty() ? "" : " " + c[2]) + ")";
                if (log) *log += "latency table: " + c[0] + " " + c[1] + " " + c[2] + " from CSV\n";
            }
        }
    }
    // Occupancy of LW/IW in the CSV is "20 per plane" = 20 issues of 1 cycle each; keep 1.
    mxm_lw.occupancy = 1;
    return true;
}

const Lat& LatencyTable::of(Opcode op) const {
    switch (op) {
        case Opcode::MemRead: return mem_read;
        case Opcode::MemWrite: return mem_write;
        case Opcode::MemGather: return mem_gather;
        case Opcode::VxmAdd: return vxm_add;
        case Opcode::VxmMul: return vxm_mul;
        case Opcode::VxmConvert: return vxm_convert;
        case Opcode::VxmRelu: return vxm_relu;
        case Opcode::VxmMax: return vxm_max;
        case Opcode::MxmLoadWeights: return mxm_lw;
        case Opcode::MxmInstallWeights: return mxm_iw;
        case Opcode::MxmActivate: return mxm_abc;
        case Opcode::Ifetch: return icu_ifetch;
        default: break;
    }
    throw std::runtime_error("LatencyTable: opcode not modeled");
}

void LatencyTable::dump_csv(const std::string& path) const {
    std::ofstream f(path);
    f << "op,d_func,d_skew,occupancy,tag,source\n";
    auto row = [&](const char* n, const Lat& l) {
        f << n << ',' << l.d_func << ',' << l.d_skew << ',' << l.occupancy << ',' << l.tag << ",\""
          << l.source << "\"\n";
    };
    row("MEM Read", mem_read);
    row("MEM Write", mem_write);
    row("MEM Gather", mem_gather);
    row("VXM add", vxm_add);
    row("VXM mul", vxm_mul);
    row("VXM convert int32->int8", vxm_convert);
    row("VXM relu", vxm_relu);
    row("VXM max", vxm_max);
    row("MXM LW", mxm_lw);
    row("MXM IW (buffer)", mxm_iw);
    row("MXM ABC", mxm_abc);
    f << "Sync preamble," << sync_cycles << ",,,P,p.152\n";
    f << "Superlane stagger," << stagger << ",,,P,Fig. 6\n";
}

// ---------------------------------------------------------------------------------------------
// DiagStreams
// ---------------------------------------------------------------------------------------------
DiagStreams::DiagStreams(int num_positions)
    : num_positions_(num_positions),
      cells_(static_cast<std::size_t>(2 * kStreamsPerDirection * kRing)),
      pool_(kPool),
      pool_cycle_(kPool, -1) {}

DiagStreams::Cell& DiagStreams::cell(Direction dir, int stream, std::int64_t diag) {
    const std::int64_t m = ((diag % kRing) + kRing) % kRing;
    const std::size_t i = static_cast<std::size_t>(
        ((dir == Direction::East ? 0 : 1) * kStreamsPerDirection + stream) * kRing + m);
    return cells_[i];
}
const DiagStreams::Cell& DiagStreams::cell(Direction dir, int stream, std::int64_t diag) const {
    return const_cast<DiagStreams*>(this)->cell(dir, stream, diag);
}

bool DiagStreams::write(int pos, Direction dir, int stream, std::int64_t cycle, const Vector& v,
                        Tag tag) {
    const std::int64_t diag = dir == Direction::East ? cycle + 1 - pos : cycle + 1 + pos;
    Cell& c = cell(dir, stream, diag);
    if (c.diag != diag) {
        c.diag = diag;
        c.entries.clear();
    }
    for (const Entry& e : c.entries)
        if (e.q == pos) return false;
    const std::uint32_t slot = next_slot_;
    next_slot_ = (next_slot_ + 1) % kPool;
    pool_[slot] = v;
    pool_cycle_[slot] = cycle;
    c.entries.push_back(Entry{static_cast<std::int16_t>(pos), slot, cycle, tag});
    return true;
}

const Vector* DiagStreams::read(int pos, Direction dir, int stream, std::int64_t cycle,
                                Tag* tag) const {
    const bool east = dir == Direction::East;
    const std::int64_t diag = east ? cycle - pos : cycle + pos;
    const Cell& c = cell(dir, stream, diag);
    if (c.diag != diag) return nullptr;
    const Entry* best = nullptr;
    for (const Entry& e : c.entries) {
        if (east ? (e.q <= pos && (!best || e.q > best->q)) : (e.q >= pos && (!best || e.q < best->q)))
            best = &e;
    }
    if (!best) return nullptr;
    if (pool_cycle_[best->slot] != best->written)
        throw std::runtime_error("DiagStreams: value pool wrapped (too much traffic in flight)");
    if (tag) *tag = best->tag;
    return &pool_[best->slot];
}

// ---------------------------------------------------------------------------------------------
// Program / Activity
// ---------------------------------------------------------------------------------------------
Program::Program() : iqs(kNumIQs) {
    for (int iq = 0; iq < kNumIQs; ++iq) iqs[static_cast<std::size_t>(iq)].ways.resize(static_cast<std::size_t>(iq_ways(iq)));
}

std::vector<QEntry>& Program::channel(int q) {
    return iqs[static_cast<std::size_t>(channel_iq(q))].ways[static_cast<std::size_t>(channel_way(q))];
}

std::size_t Program::num_entries() const {
    std::size_t n = 0;
    for (const auto& iq : iqs)
        for (const auto& w : iq.ways) n += w.size();
    return n;
}

int Program::iqs_used() const {
    int n = 0;
    for (const auto& iq : iqs) {
        bool any = false;
        for (const auto& w : iq.ways)
            for (const auto& e : w) any = any || e.kind == QEntry::Ins;
        n += any;
    }
    return n;
}

void Activity::bump(std::vector<std::uint8_t>& v, std::int64_t cycle, int n) {
    const std::size_t c = static_cast<std::size_t>(cycle);
    if (v.size() <= c) v.resize(c + 4096, 0);
    v[c] = static_cast<std::uint8_t>(std::min(255, v[c] + n));
}

// ---------------------------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------------------------
Simulator::Simulator(const Floorplan& fp, const LatencyTable& lat)
    : fp_(fp),
      lat_(lat),
      sr_(fp.num_positions),
      mem_(kMemSlices, std::vector<Vector>(kMemWords)),
      bank_read_cycle_(2 * kMemSlices, -1),
      bank_write_cycle_(2 * kMemSlices, -1),
      iq_cycle_(kNumIQs, -1),
      iq_r_(kNumIQs, 0),
      iq_w_(kNumIQs, 0),
      iq_n_(kNumIQs, 0) {
    for (auto& m : mem_)
        for (auto& w : m) w.fill(0);
}

void Simulator::violation(std::int64_t cycle, int q, const std::string& what) {
    ++violations;
    if (violation_log.size() < 40)
        violation_log.push_back("cycle " + std::to_string(cycle) + " " + fp_.queue_name(q) + ": " + what);
}

const Vector* Simulator::operand(std::int64_t cycle, int q, const SimInstr& si, int k, int sub, int r) {
    static const Vector kZero{};
    const StreamGroup& sg = si.in.src[static_cast<std::size_t>(k)];
    const int stream = sg.base + sub;
    const std::int64_t t = cycle + lat_.of(si.in.opcode).d_skew;
    Tag tag;
    const Vector* v = sr_.read(fp_.queue_pos(q), sg.dir, stream, t, &tag);
    const std::uint32_t want_edge = si.in_edge[static_cast<std::size_t>(k)];
    const std::uint32_t want_idx = si.in_idx0[static_cast<std::size_t>(k)] + static_cast<std::uint32_t>(r);
    if (!v) {
        violation(cycle, q, "operand " + std::to_string(k) + " stream " + std::to_string(stream) +
                                " is empty");
        return &kZero;
    }
    if (want_edge != 0 && !(tag.edge == want_edge && tag.idx == want_idx)) {
        violation(cycle, q, "operand " + std::to_string(k) + " stream " + std::to_string(stream) +
                                " holds (" + std::to_string(tag.edge) + "," + std::to_string(tag.idx) +
                                ") expected (" + std::to_string(want_edge) + "," +
                                std::to_string(want_idx) + ")");
    }
    return v;
}

void Simulator::emit(std::int64_t cycle, int pos, Direction dir, int stream, const Vector& v, Tag tag) {
    auto& b = emits_[static_cast<std::size_t>(cycle % kEmitRing)];
    b.push_back(Emit{pos, dir, stream, v, tag});
    ++pending_emits_;
}

namespace {
inline void load32(const Vector* const* s, int width, std::int32_t* out) {
    if (width == 1) {
        for (int i = 0; i < kLanes; ++i) out[i] = static_cast<std::int8_t>((*s[0])[static_cast<std::size_t>(i)]);
    } else {
        for (int i = 0; i < kLanes; ++i) {
            const std::uint32_t u = static_cast<std::uint32_t>((*s[0])[static_cast<std::size_t>(i)]) |
                                    (static_cast<std::uint32_t>((*s[1])[static_cast<std::size_t>(i)]) << 8) |
                                    (static_cast<std::uint32_t>((*s[2])[static_cast<std::size_t>(i)]) << 16) |
                                    (static_cast<std::uint32_t>((*s[3])[static_cast<std::size_t>(i)]) << 24);
            out[i] = static_cast<std::int32_t>(u);
        }
    }
}
inline std::int8_t sat8(std::int32_t x) {
    return static_cast<std::int8_t>(std::max(-128, std::min(127, x)));
}
}  // namespace

void Simulator::exec_vxm(std::int64_t cycle, int q, const SimInstr& si, int r) {
    const Instruction& in = si.in;
    std::int32_t a[kLanes], b[kLanes], y[kLanes];
    auto fetch = [&](int k, std::int32_t* out) {
        const int w = in.src[static_cast<std::size_t>(k)].width;
        const Vector* s[4];
        for (int i = 0; i < w; ++i) s[i] = operand(cycle, q, si, k, i, r);
        load32(s, w, out);
    };
    bool to_int8_sat = in.saturate;
    switch (in.opcode) {
        case Opcode::VxmAdd:
            fetch(0, a);
            fetch(1, b);
            for (int i = 0; i < kLanes; ++i)
                y[i] = static_cast<std::int32_t>(static_cast<std::uint32_t>(a[i]) + static_cast<std::uint32_t>(b[i]));
            break;
        case Opcode::VxmMul:
            fetch(0, a);
            for (int i = 0; i < kLanes; ++i)
                y[i] = static_cast<std::int32_t>(static_cast<std::uint32_t>(a[i]) * static_cast<std::uint32_t>(in.imm));
            break;
        case Opcode::VxmConvert: {
            fetch(0, a);
            const int s = in.imm;
            const std::uint32_t rnd = s > 0 ? (1u << (s - 1)) : 0u;
            for (int i = 0; i < kLanes; ++i)
                y[i] = static_cast<std::int32_t>(static_cast<std::uint32_t>(a[i]) + rnd) >> s;
            to_int8_sat = true;
            break;
        }
        case Opcode::VxmRelu:
            fetch(0, a);
            for (int i = 0; i < kLanes; ++i) y[i] = std::max(0, a[i]);
            break;
        case Opcode::VxmMax:
            fetch(0, a);
            fetch(1, b);
            for (int i = 0; i < kLanes; ++i) y[i] = std::max(a[i], b[i]);
            break;
        default:
            violation(cycle, q, "opcode not executable on the VXM");
            return;
    }
    act.bump(act.vxm, cycle);
    const int d = lat_.of(in.opcode).d_func;
    const Tag tag{si.out_edge, si.out_idx0 + static_cast<std::uint32_t>(r)};
    const int pos = fp_.queue_pos(q);
    if (in.dst.width == 1) {
        Vector v;
        for (int i = 0; i < kLanes; ++i)
            v[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(to_int8_sat ? sat8(y[i]) : static_cast<std::int8_t>(y[i]));
        emit(cycle + d - 1, pos, in.dst.dir, in.dst.base, v, tag);
    } else {
        for (int bidx = 0; bidx < 4; ++bidx) {
            Vector v;
            for (int i = 0; i < kLanes; ++i)
                v[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(static_cast<std::uint32_t>(y[i]) >> (8 * bidx));
            emit(cycle + d - 1, pos, in.dst.dir, in.dst.base + bidx, v, tag);
        }
    }
}

bool Simulator::legal(int q, Opcode op) const {
    if (q_is_fetch(q)) return op == Opcode::Ifetch;
    if (q_is_mem(q))
        return (q & 1) ? (op == Opcode::MemWrite || op == Opcode::MemScatter)
                       : (op == Opcode::MemRead || op == Opcode::MemGather);
    if (q_is_vxm(q)) return op >= Opcode::VxmAdd && op <= Opcode::VxmConvert;
    if (q_is_mxm(q)) {
        switch (q_mxm_role(q)) {
            case kMxmLW: return op == Opcode::MxmLoadWeights;
            case kMxmIW: return op == Opcode::MxmInstallWeights;
            default: return op == Opcode::MxmActivate || op == Opcode::MxmAccumulate;
        }
    }
    return op >= Opcode::SxmShiftNorth && op <= Opcode::SxmTranspose;
}

void Simulator::execute(std::int64_t cycle, int q, const SimInstr& si, int r) {
    const Instruction& in = si.in;
    ++issued;
    if (!legal(q, in.opcode)) {
        violation(cycle, q, "opcode " + std::to_string(static_cast<int>(in.opcode)) + " is not legal on this IQ / way");
        return;
    }
    // Ifetch (fetch way): take 16 bytes of the instruction stream per cycle into the IQ and check
    // they are the next bytes of the IQ's text, in order.
    if (q_is_fetch(q)) {
        const int iq = q - kQFetchBase;
        const Vector* v = operand(cycle, q, si, 0, 0, r);
        const auto& text = prog_->iqs[static_cast<std::size_t>(iq)].text;
        const std::uint32_t off = si.text_off + static_cast<std::uint32_t>(kFetchBytesPerCycle * r);
        const std::uint32_t end = std::min(si.text_off + si.text_len, off + static_cast<std::uint32_t>(kFetchBytesPerCycle));
        ++ifetch_vectors;
        if (off >= end) return;   // padding vector of a short last piece
        std::uint32_t& have = fetched_[static_cast<std::size_t>(iq)];
        if (off != have) violation(cycle, q, "Ifetch out of order: brings byte " + std::to_string(off) + ", IQ has " + std::to_string(have));
        for (std::uint32_t b = off; b < end; ++b)
            if (b >= text.size() || (*v)[b - off] != text[b]) {
                violation(cycle, q, "Ifetch delivered wrong instruction bytes at " + std::to_string(b));
                break;
            }
        have = end;
        delivered_[static_cast<std::size_t>(iq)].push_back(ByteEv{cycle, end, end - off});
        return;
    }
    // Per-IQ issue limits in one cycle: MEM at most one R-type and one W-type (patent), every
    // other IQ one instruction [Ours].
    {
        const std::size_t iq = static_cast<std::size_t>(channel_iq(q));
        if (iq_cycle_[iq] != cycle) { iq_cycle_[iq] = cycle; iq_r_[iq] = iq_w_[iq] = iq_n_[iq] = 0; }
        ++iq_n_[iq];
        const bool rtype = in.opcode == Opcode::MemRead || in.opcode == Opcode::MemGather;
        const bool wtype = in.opcode == Opcode::MemWrite || in.opcode == Opcode::MemScatter;
        if (rtype) ++iq_r_[iq];
        if (wtype) ++iq_w_[iq];
        if (q_is_mem(q) && !mem_single_issue ? (iq_r_[iq] > 1 || iq_w_[iq] > 1) : iq_n_[iq] > 1)
            violation(cycle, q, q_is_mem(q) ? "more than one R-type or W-type instruction from one MEM IQ in one cycle"
                                            : "more than one instruction from a single-issue IQ in one cycle");
    }
    if (q_is_sxm(q)) {
        violation(cycle, q, "SXM instructions are not modeled yet");
        return;
    }
    const Tag out_tag{si.out_edge, si.out_idx0 + static_cast<std::uint32_t>(r)};
    const int pos = fp_.queue_pos(q);
    auto bank_check = [&](int g, int addr, bool is_write) {
        const std::size_t bi = static_cast<std::size_t>(g * 2 + addr / kBankWords);
        auto& mine = is_write ? bank_write_cycle_ : bank_read_cycle_;
        auto& other = is_write ? bank_read_cycle_ : bank_write_cycle_;
        if (other[bi] == cycle) violation(cycle, q, "read and write on the same bank in one cycle");
        mine[bi] = cycle;
    };
    if (q_is_mem(q)) {
        const int g = q_mem_slice(q);
        const bool write_port = (q & 1) != 0;
        if (in.opcode == Opcode::MemRead && !write_port) {
            const int addr = in.address + r * si.addr_stride;
            if (addr < 0 || addr >= kMemWords) { violation(cycle, q, "address out of range"); return; }
            bank_check(g, addr, false);
            act.bump(act.mem_read, cycle);
            emit(cycle + lat_.mem_read.d_func - 1, pos, in.dst.dir, in.dst.base, mem_[static_cast<std::size_t>(g)][static_cast<std::size_t>(addr)], out_tag);
            return;
        }
        if (in.opcode == Opcode::MemGather && !write_port) {
            const Vector* map = operand(cycle, q, si, 0, 0, r);
            Vector out;
            int last_bank = -1;
            for (int t = 0; t < kSuperlanes; ++t) {
                const int addr = (*map)[static_cast<std::size_t>(t * 16)] |
                                 ((*map)[static_cast<std::size_t>(t * 16 + 1)] << 8);
                if (addr >= kMemWords) { violation(cycle, q, "gather address out of range"); return; }
                if (addr / kBankWords != last_bank) { bank_check(g, addr, false); last_bank = addr / kBankWords; }
                const Vector& w = mem_[static_cast<std::size_t>(g)][static_cast<std::size_t>(addr)];
                std::copy(w.begin() + t * 16, w.begin() + t * 16 + 16, out.begin() + t * 16);
            }
            act.bump(act.mem_read, cycle);
            emit(cycle + lat_.mem_gather.d_func - 1, pos, in.dst.dir, in.dst.base, out, out_tag);
            return;
        }
        if (in.opcode == Opcode::MemWrite && write_port) {
            const int addr = in.address + r * si.addr_stride;
            if (addr < 0 || addr >= kMemWords) { violation(cycle, q, "address out of range"); return; }
            const Vector* v = operand(cycle, q, si, 0, 0, r);
            bank_check(g, addr, true);
            act.bump(act.mem_write, cycle);
            writes_.push_back(PendingWrite{g, addr, *v});
            return;
        }
        violation(cycle, q, "opcode not legal on this MEM queue");
        return;
    }
    if (q_is_vxm(q)) {
        exec_vxm(cycle, q, si, r);
        return;
    }
    const int p = q_plane(q);
    Plane& pl = planes_[static_cast<std::size_t>(p)];
    const int role = q_mxm_role(q);
    if (in.opcode == Opcode::MxmLoadWeights && role == kMxmLW) {
        if (r >= kSuperlanes) { violation(cycle, q, "LW beyond 20 issues"); return; }
        for (int s = 0; s < 16; ++s) {
            const Vector* v = operand(cycle, q, si, 0, s, r);
            const int f = 16 * r + s;
            for (int i = 0; i < kLanes; ++i)
                pl.buf[static_cast<std::size_t>(f * kLanes + i)] = static_cast<std::int8_t>((*v)[static_cast<std::size_t>(i)]);
        }
        act.bump(act.lw, cycle);
        return;
    }
    if (in.opcode == Opcode::MxmInstallWeights && role == kMxmIW) {
        pl.w = pl.buf;
        return;
    }
    if (in.opcode == Opcode::MxmActivate && role == kMxmABC) {
        const Vector* xv = operand(cycle, q, si, 0, 0, r);
        std::int8_t x[kLanes];
        for (int i = 0; i < kLanes; ++i) x[i] = static_cast<std::int8_t>((*xv)[static_cast<std::size_t>(i)]);
        std::int32_t y[kLanes];
        for (int f = 0; f < kLanes; ++f) {
            const std::int8_t* wf = &pl.w[static_cast<std::size_t>(f * kLanes)];
            std::int32_t acc = 0;
            for (int i = 0; i < kLanes; ++i) acc += static_cast<std::int16_t>(wf[i]) * static_cast<std::int16_t>(x[i]);
            y[f] = acc;
        }
        ++abc_issued;
        act.bump(act.abc, cycle);
        for (int bidx = 0; bidx < 4; ++bidx) {
            Vector v;
            for (int i = 0; i < kLanes; ++i)
                v[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(static_cast<std::uint32_t>(y[i]) >> (8 * bidx));
            emit(cycle + lat_.mxm_abc.d_func - 1, pos, in.dst.dir, in.dst.base + bidx, v, out_tag);
        }
        return;
    }
    violation(cycle, q, "MXM opcode not modeled (ACC)");
}

void Simulator::run(const Program& prog) {
    struct Cursor {
        std::size_t idx = 0;
        std::int64_t t = 0;
        const SimInstr* last = nullptr;
        int r = 0;
        std::uint32_t rep_left = 0;
        std::uint32_t rep_d = 1;
        std::int64_t last_issue = -1;
    };
    // one cursor per issue channel = (IQ, way); every way of an IQ is its own dispatch thread
    std::vector<Cursor> cur(kNumChannels);
    std::vector<const std::vector<QEntry>*> prog_of(kNumChannels);
    for (int q = 0; q < kNumChannels; ++q)
        prog_of[static_cast<std::size_t>(q)] =
            &prog.iqs[static_cast<std::size_t>(channel_iq(q))].ways[static_cast<std::size_t>(channel_way(q))];
    using HE = std::pair<std::int64_t, int>;
    std::priority_queue<HE, std::vector<HE>, std::greater<HE>> heap;
    prog_ = &prog;
    consumed_.assign(kNumIQs, {});
    delivered_.assign(kNumIQs, {});
    fetched_.assign(kNumIQs, 0);
    for (int iq = 0; iq < kNumIQs; ++iq) fetched_[static_cast<std::size_t>(iq)] = prog.iqs[static_cast<std::size_t>(iq)].preload;
    auto consume = [&](int q, std::int64_t t, const QEntry& e) {
        consumed_[static_cast<std::size_t>(channel_iq(q))].push_back(ByteEv{t, e.byte_off + e.byte_len, e.byte_len});
    };

    // Advance a queue's ICU to its next issue; returns -1 when the queue is drained.
    auto next_issue = [&](int q) -> std::int64_t {
        Cursor& c = cur[static_cast<std::size_t>(q)];
        const auto& entries = *prog_of[static_cast<std::size_t>(q)];
        while (true) {
            if (c.rep_left > 0) return c.last_issue + c.rep_d;
            if (c.idx >= entries.size()) return -1;
            const QEntry& e = entries[c.idx++];
            if (e.kind == QEntry::Nop) {
                consume(q, c.t, e);   // a NOP leaves the IQ when it starts counting
                c.t += e.n;
            } else if (e.kind == QEntry::Ins) {
                consume(q, c.t, e);   // issue
                c.last = &e.ins;
                c.r = -1;
                return c.t;
            } else {
                if (!c.last) throw std::runtime_error("Repeat with no previous instruction");
                consume(q, c.last_issue + e.d, e);   // first repeat
                c.rep_left = e.n;
                c.rep_d = e.d;
            }
        }
    };
    for (int iq = 0; iq < kNumIQs; ++iq)
        if (prog.iqs[static_cast<std::size_t>(iq)].ways.size() > static_cast<std::size_t>(iq_ways(iq)))
            throw std::runtime_error("program has more ways than IQ " + fp_.iq_name(iq) + " supports");
    for (int q = 0; q < kNumChannels; ++q) {
        const std::int64_t t = next_issue(q);
        if (t >= 0) heap.push({t, q});
    }
    std::int64_t T = 0;
    while (!heap.empty() || pending_emits_ > 0) {
        std::int64_t next = heap.empty() ? INT64_MAX : heap.top().first;
        if (pending_emits_ > 0) {
            for (std::int64_t c = T; c < T + kEmitRing; ++c) {
                if (!emits_[static_cast<std::size_t>(c % kEmitRing)].empty()) {
                    next = std::min(next, c);
                    break;
                }
            }
        }
        T = next;
        while (!heap.empty() && heap.top().first == T) {
            const int q = heap.top().second;
            heap.pop();
            Cursor& c = cur[static_cast<std::size_t>(q)];
            if (c.rep_left > 0) {
                --c.rep_left;
            }
            ++c.r;
            c.last_issue = T;
            c.t = T + 1;
            execute(T, q, *c.last, c.r);
            const std::int64_t t = next_issue(q);
            if (t >= 0) heap.push({t, q});
        }
        auto& bucket = emits_[static_cast<std::size_t>(T % kEmitRing)];
        for (const Emit& e : bucket) {
            if (!sr_.write(e.pos, e.dir, e.stream, T, e.v, e.tag))
                violation(T, kQVxmBase, "two writers on stream " + std::to_string(e.stream) + " at SR position " + std::to_string(e.pos));
            act.bump(act.sr_write, T);
        }
        pending_emits_ -= bucket.size();
        bucket.clear();
        for (const PendingWrite& w : writes_) mem_[static_cast<std::size_t>(w.g)][static_cast<std::size_t>(w.addr)] = w.v;
        writes_.clear();
        last_cycle = T;
    }
    check_text(prog);
}

// After the run: every instruction's bytes must have reached its IQ (preload or an Ifetch that
// finished an earlier cycle) before it issues (no underflow, patent col. 13), and the bytes in an
// IQ (fetched, not yet issued) may never exceed its capacity.
void Simulator::check_text(const Program& prog) {
    iq_fill.assign(kNumIQs, {});
    iq_max_fill.assign(kNumIQs, 0);
    for (int iq = 0; iq < kNumIQs; ++iq) {
        const IQProgram& P = prog.iqs[static_cast<std::size_t>(iq)];
        auto cons = consumed_[static_cast<std::size_t>(iq)];
        const auto& del = delivered_[static_cast<std::size_t>(iq)];
        if (cons.empty()) continue;
        std::stable_sort(cons.begin(), cons.end(), [](const ByteEv& a, const ByteEv& b) { return a.t < b.t; });
        // underflow: prefix delivered strictly before t
        std::size_t di = 0;
        std::uint32_t prefix = P.preload;
        for (const ByteEv& e : cons) {
            while (di < del.size() && del[di].t < e.t) prefix = del[di++].end;
            if (e.end > prefix) {
                ++underflows;
                if (underflows <= 20)
                    violation(e.t, channel_of(iq, 0), "IQ underflow: needs text byte " + std::to_string(e.end - 1) +
                                                          ", only " + std::to_string(prefix) + " fetched");
            }
        }
        // fill level: bytes fetched minus bytes issued; a delivery and an issue in one cycle
        // count the issue first
        std::int64_t fetched = P.preload, used = 0;
        std::size_t ci = 0;
        di = 0;
        auto& fill = iq_fill[static_cast<std::size_t>(iq)];
        fill.push_back({0, static_cast<int>(fetched)});
        if (P.preload > P.capacity) {
            ++overflows;
            violation(0, channel_of(iq, 0), "IQ overflow: preload " + std::to_string(P.preload) + " bytes > " + std::to_string(P.capacity));
        }
        while (ci < cons.size() || di < del.size()) {
            const std::int64_t t = std::min(ci < cons.size() ? cons[ci].t : INT64_MAX, di < del.size() ? del[di].t : INT64_MAX);
            while (ci < cons.size() && cons[ci].t == t) used += cons[ci++].len;
            while (di < del.size() && del[di].t == t) fetched += del[di++].len;
            const int occ = static_cast<int>(fetched - used);
            iq_max_fill[static_cast<std::size_t>(iq)] = std::max(iq_max_fill[static_cast<std::size_t>(iq)], occ);
            if (occ > static_cast<int>(P.capacity)) {
                ++overflows;
                if (overflows <= 20) violation(t, channel_of(iq, 0), "IQ overflow: " + std::to_string(occ) + " bytes > " + std::to_string(P.capacity));
            }
            if (fill.back().first == t) fill.back().second = occ;
            else fill.push_back({t, occ});
        }
    }
}

}  // namespace sim
}  // namespace lpu
