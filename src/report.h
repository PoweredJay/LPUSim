// report.h - the files a ResNet run writes: results/ CSVs + summary JSON, and the optional
// visualization event export. Nothing here changes a result; it only reads the finished run.
//
// All cycles written by these functions are chip cycles (Sync preamble included). The scheduler
// works in program time, which starts after the preamble; chip_cycle() converts.
#pragma once

#include <cstdint>
#include <string>

#include "compiler.h"
#include "ifetch.h"

namespace lpu {
namespace report {

// Everything a finished run produced, plus the headline numbers main() computed from it.
struct Run {
    const comp::Net& net;
    const comp::Compiler& cc;
    const sched::Scheduler& sch;
    const sim::Simulator& sim;
    const sim::Program& prog;
    const comp::IfetchStats& ist;
    const sim::Floorplan& fp;
    const sim::LatencyTable& lat;

    std::string tag;               // e.g. "resnet50" + --tag suffix; prefix of every file name
    int depth = 50;
    std::int64_t cycles = 0;       // chip cycles: Sync preamble + program + superlane stagger
    std::int64_t compute_end = 0;  // compute makespan before Ifetch (program time)
    std::uint64_t macs = 0;
    bool golden_checked = false;
    std::size_t logits_wrong = 0;

    std::int64_t chip_cycle(std::int64_t program_cycle) const { return program_cycle + lat.sync_cycles; }
    double latency_us(double clock_mhz) const { return static_cast<double>(cycles) / clock_mhz; }
    double ips(double clock_mhz) const { return 1e6 / latency_us(clock_mhz); }
};

// <dir>/<tag>_layers.csv, _trace.csv, _activity.csv, _iqbytes.csv, _iqfill.csv, _summary.json
void write_results(const std::string& dir, const Run& r);

// <dir>/<tag>_events.csv, _streams.csv, _pipes.csv, _alloc.csv, _programs.csv (for build_viz.py)
void write_events(const std::string& dir, const Run& r);

}  // namespace report
}  // namespace lpu
