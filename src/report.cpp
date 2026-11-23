// report.cpp - see report.h.
#include "report.h"

#include <algorithm>
#include <fstream>
#include <ostream>

namespace lpu {
namespace report {

using namespace lpu::sim;
using namespace lpu::sched;
using namespace lpu::comp;

namespace {

std::ofstream open(const std::string& dir, const Run& r, const char* suffix) {
    return std::ofstream(dir + "/" + r.tag + suffix);
}

// A flat JSON object written field by field: {"key": value, ...}, one field per line.
class JsonObject {
public:
    explicit JsonObject(std::ostream& f) : f_(f) {}
    ~JsonObject() { f_ << (first_ ? "{" : "") << "\n}\n"; }

    template <class T>
    JsonObject& num(const char* key, const T& v) { this->key(key); f_ << v; return *this; }
    JsonObject& str(const char* key, const std::string& v) { this->key(key); f_ << '"' << v << '"'; return *this; }
    JsonObject& boolean(const char* key, bool v) { this->key(key); f_ << (v ? "true" : "false"); return *this; }
    JsonObject& null(const char* key) { this->key(key); f_ << "null"; return *this; }

private:
    void key(const char* k) {
        f_ << (first_ ? "{\n  \"" : ",\n  \"") << k << "\": ";
        first_ = false;
    }
    std::ostream& f_;
    bool first_ = true;
};

const char* layer_kind(const LayerDesc& L) {
    if (L.kind == LayerDesc::Conv) return L.raw32 ? "fc" : "conv";
    return L.kind == LayerDesc::MaxPool ? "maxpool" : "avgpool";
}

void write_layers(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_layers.csv");
    f << "index,name,kind,first_cycle,last_cycle,macs,abc_vectors,rounds,tiles,kvecs,out_words,chunks\n";
    for (std::size_t i = 0; i < r.cc.stats.size(); ++i) {
        const auto& s = r.cc.stats[i];
        f << i << ',' << s.name << ',' << layer_kind(r.net.layers[i]) << ',' << r.chip_cycle(s.first) << ','
          << r.chip_cycle(s.last) << ',' << s.macs << ',' << s.abc_vectors << ',' << s.rounds << ',' << s.tiles << ','
          << s.kvecs << ',' << s.nwords << ',' << s.chunks << '\n';
    }
}

void write_trace(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_trace.csv");
    f << "iq,way,channel,unit,pos,start,n,kind,layer,label\n";
    for (const auto& o : r.sch.trace())
        f << channel_iq(o.queue) << ',' << channel_way(o.queue) << ',' << o.queue << ',' << r.fp.iq_name(channel_iq(o.queue))
          << ',' << o.pos << ',' << r.chip_cycle(o.start) << ',' << o.n << ',' << o.kind << ',' << o.layer << ",\""
          << o.label << "\"\n";
}

void write_activity(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_activity.csv");
    f << "cycle,mxm_abc_planes,mxm_lw,vxm_alus,mem_reads,mem_writes,sr_writes\n";
    auto at = [](const std::vector<std::uint8_t>& v, std::size_t c) { return c < v.size() ? int(v[c]) : 0; };
    const auto& a = r.sim.act;
    for (std::size_t c = 0; c < static_cast<std::size_t>(r.cycles); ++c)
        f << c << ',' << at(a.abc, c) << ',' << at(a.lw, c) << ',' << at(a.vxm, c) << ',' << at(a.mem_read, c) << ','
          << at(a.mem_write, c) << ',' << at(a.sr_write, c) << '\n';
}

void write_iqbytes(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_iqbytes.csv");
    f << "iq,unit,program_bytes,preload_bytes,max_fill\n";
    for (int iq = 0; iq < kNumIQs; ++iq) {
        const auto i = static_cast<std::size_t>(iq);
        f << iq << ',' << r.fp.iq_name(iq) << ',' << r.ist.program_bytes[i] << ',' << r.prog.iqs[i].preload << ','
          << (i < r.sim.iq_max_fill.size() ? r.sim.iq_max_fill[i] : 0) << '\n';
    }
}

void write_iqfill(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_iqfill.csv");   // occupancy steps, chip cycles
    f << "iq,unit,cycle,bytes\n";
    for (std::size_t iq = 0; iq < r.sim.iq_fill.size(); ++iq)
        for (const auto& [t, b] : r.sim.iq_fill[iq])
            f << iq << ',' << r.fp.iq_name(static_cast<int>(iq)) << ',' << t << ',' << b << '\n';
}

int max_iq_fill(const Simulator& sim) {
    int m = 0;
    for (int f : sim.iq_max_fill) m = std::max(m, f);
    return m;
}

void write_summary(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_summary.json");
    const auto& cc = r.cc;
    const auto& sim = r.sim;
    const auto& ist = r.ist;
    JsonObject j(f);
    j.str("model", r.tag)
        .num("cycles", r.cycles)
        .num("sync_preamble", r.lat.sync_cycles)
        .num("stagger_tail", r.lat.stagger)
        .num("latency_us_1GHz", r.latency_us(1000.0))
        .num("latency_us_0.9GHz", r.latency_us(900.0))
        .num("ips_1GHz", r.ips(1000.0))
        .num("ips_0.9GHz", r.ips(900.0))
        .num("macs", r.macs)
        .num("mxm_vector_ops", sim.abc_issued)
        .num("instructions_issued", sim.issued)
        .num("iq_entries", r.prog.num_entries())
        .num("iqs_total", kNumIQs)
        .num("iqs_used", r.prog.iqs_used())
        .num("violations", sim.violations)
        .boolean("golden_checked", r.golden_checked)
        .num("logits_wrong", r.logits_wrong)
        .num("weight_words", cc.weight_words)
        .num("weight_tiles", cc.tiles_total)
        .num("weight_tiles_aliased", cc.weights_aliased)
        .boolean("timing_only_projection", cc.weights_aliased != 0)
        .num("map_words", cc.map_words)
        .num("peak_runtime_words", cc.peak_words)
        .num("compute_makespan", r.compute_end)
        .num("ifetches", ist.ifetches)
        .num("ifetch_vectors", ist.ifetch_vectors)
        .num("iqs_fetching", ist.iqs_fetching)
        .num("text_bytes", ist.text_bytes)
        .num("preload_bytes", ist.preload_bytes)
        .num("fetched_bytes", ist.fetched_bytes)
        .num("text_words", ist.text_words)
        .num("ifetch_read_port_cycles", ist.read_port_cycles)
        .num("ifetch_stream_cycles", ist.stream_cycles)
        .num("ifetch_late", ist.late)
        .num("ifetch_min_slack", ist.ifetches ? ist.min_slack : 0)
        .num("iq_underflows", sim.underflows)
        .num("iq_overflows", sim.overflows)
        .num("iq_max_fill", max_iq_fill(sim));
    // The paper states a latency only for ResNet-50 (< 49 us); its 101/152 IPS figures are projections.
    if (r.depth == 50) j.num("paper_latency_us", 49);
    else j.null("paper_latency_us");
    j.num("paper_ips", r.depth == 50 ? 20400 : r.depth == 101 ? 14300 : 10700);
}

// ---- visualization export ----

void write_stream_group(std::ostream& f, const StreamGroup& g) {
    if (!g.used()) f << ",,,";
    else f << ',' << int(g.base) << ',' << int(g.width) << ',' << (g.dir == Direction::East ? 'E' : 'W');
}

void write_event_ops(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_events.csv");
    f << "id,cycle,iq,way,channel,unit,opcode,pos,n,d_func,kind,layer,pipe,"
         "src0_base,src0_w,src0_dir,src1_base,src1_w,src1_dir,dst_base,dst_w,dst_dir,"
         "mem_slice,bank,addr_lo,addr_hi,stride,label\n";
    for (std::size_t i = 0; i < r.sch.ev_ops.size(); ++i) {
        const EventOp& e = r.sch.ev_ops[i];
        const int iq = channel_iq(e.queue);
        f << i << ',' << r.chip_cycle(e.start) << ',' << iq << ',' << channel_way(e.queue) << ',' << e.queue << ','
          << r.fp.iq_name(iq) << ',' << opcode_name(e.opcode) << ',' << e.pos << ',' << e.n << ',' << e.d_func << ','
          << e.kind << ',' << e.layer << ',' << e.pipe;
        write_stream_group(f, e.src[0]);
        write_stream_group(f, e.src[1]);
        write_stream_group(f, e.dst);
        // optional MEM fields: empty when the op has none
        f << ',';
        if (e.mem_g >= 0) f << r.fp.iq_name(iq_mem(e.mem_g));
        f << ',';
        if (e.bank >= 0) f << e.bank;
        f << ',';
        if (e.addr_lo >= 0) f << e.addr_lo;
        f << ',';
        if (e.addr_hi >= 0) f << e.addr_hi;
        f << ',' << e.stride << ",\"" << e.label << "\"\n";
    }
}

void write_event_streams(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_streams.csv");
    f << "pipe,edge,producer,dir,stream,d0,d1,p0,p1\n";
    for (const auto& s : r.sch.ev_streams) {
        // shift diagonals to chip time: East D = t - p, West D = t + p
        f << s.pipe << ',' << s.edge << ',' << s.producer << ',' << (s.dir == Direction::East ? 'E' : 'W') << ','
          << s.stream << ',' << r.chip_cycle(s.d0) << ',' << r.chip_cycle(s.d1) << ',' << s.p0 << ',' << s.p1 << '\n';
    }
}

void write_event_pipes(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_pipes.csv");
    f << "pipe,start,layer,name\n";
    for (std::size_t i = 0; i < r.sch.ev_pipes.size(); ++i) {
        const auto& p = r.sch.ev_pipes[i];
        f << i << ',' << r.chip_cycle(p.S) << ',' << p.layer << ",\"" << p.name << "\"\n";
    }
}

void write_event_allocs(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_alloc.csv");
    f << "id,kind,bank,base,size,pipe_alloc,pipe_free,slices,what\n";
    for (std::size_t i = 0; i < r.cc.alloc_log.size(); ++i) {
        const auto& a = r.cc.alloc_log[i];
        f << i << ',' << a.kind << ',' << a.bank << ',' << a.base << ',' << a.size << ',' << a.pipe_alloc << ','
          << a.pipe_free << ",\"";
        for (std::size_t k = 0; k < a.slices.size(); ++k) f << (k ? " " : "") << r.fp.iq_name(iq_mem(a.slices[k]));
        f << "\",\"" << a.what << "\"\n";
    }
}

// The emitted IQ programs, entry by entry (op_id = row of _events.csv).
void write_event_programs(const std::string& dir, const Run& r) {
    auto f = open(dir, r, "_programs.csv");
    f << "iq,way,idx,entry,n,d,op_id\n";
    for (int iq = 0; iq < kNumIQs; ++iq) {
        const auto& ways = r.prog.iqs[static_cast<std::size_t>(iq)].ways;
        for (std::size_t w = 0; w < ways.size(); ++w)
            for (std::size_t k = 0; k < ways[w].size(); ++k) {
                const QEntry& q = ways[w][k];
                const char kind = q.kind == QEntry::Nop ? 'N' : q.kind == QEntry::Ins ? 'I' : 'R';
                f << iq << ',' << w << ',' << k << ',' << kind << ',' << q.n << ',' << q.d << ','
                  << (q.kind == QEntry::Ins ? q.ins.op_id : -1) << '\n';
            }
    }
}

}  // namespace

void write_results(const std::string& dir, const Run& r) {
    write_layers(dir, r);
    write_trace(dir, r);
    write_activity(dir, r);
    write_iqbytes(dir, r);
    write_iqfill(dir, r);
    write_summary(dir, r);
}

void write_events(const std::string& dir, const Run& r) {
    write_event_ops(dir, r);
    write_event_streams(dir, r);
    write_event_pipes(dir, r);
    write_event_allocs(dir, r);
    write_event_programs(dir, r);
}

}  // namespace report
}  // namespace lpu
