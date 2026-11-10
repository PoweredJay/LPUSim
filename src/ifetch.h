// ifetch.h - bounded instruction queues: instruction text in MEM, Ifetch pipelines, and the
// mapping of every program entry to its place in its IQ's text.
//
// Model (tags: [P] patent US 11,360,934 / ISCA 2020, [Ours] this project):
//  * Each IQ holds 640 bytes [P col. 12]. NOP = 1 byte, every other instruction 4 bytes [P col. 10;
//    encoding Ours]. The first 640 bytes of each IQ are loaded before the Sync (boot) [Ours].
//  * The rest of the program lives in MEM as constants [P col. 16 "microprograms"], allocated in
//    never-used space and counted against capacity. One MEM word carries 16 bytes of text in lanes
//    0-15, because an IQ takes 16 bytes per cycle from the stream [ISCA 2020, per planning doc].
//  * The compiler inserts Ifetch: each one is a pipeline (MEM Read of the text words -> stream ->
//    Ifetch on the IQ's fetch way) that the scheduler places like any other, so it reserves the
//    read port, the bank and the streams. Ifetches are planned after the compute schedule, in
//    deadline order, and use only resources the compute pipelines left free.
//  * Each Ifetch brings `chunk_bytes` (default 320 = half the IQ, so a refill can land while the
//    other half drains) [Ours]. It may only start when the IQ has room for the whole piece, and must
//    finish before the first instruction it brings has to issue. The simulator re-checks both:
//    no underflow (patent col. 13: no IQ may run empty) and no overflow.
#pragma once

#include <cstdint>
#include <vector>

#include "compiler.h"

namespace lpu {
namespace comp {

struct IfetchOptions {
    int chunk_bytes = 320;    // bytes per Ifetch [Ours]
    int candidates = 8;       // MEM slices tried per Ifetch, nearest to the IQ first
    int mem_headroom = 48;    // spare bytes per piece of a MEM IQ, for text Reads placed on it later
};

struct IfetchStats {
    int ifetches = 0, iqs_used = 0, iqs_fetching = 0, late = 0, shrunk = 0, rejected = 0;
    std::uint64_t text_bytes = 0;        // all IQ programs
    std::uint64_t preload_bytes = 0;     // loaded before the Sync
    std::uint64_t fetched_bytes = 0;     // brought in by Ifetch
    std::uint64_t text_words = 0;        // MEM words holding fetched text
    std::uint64_t read_port_cycles = 0;  // MEM read-port cycles spent on Ifetch
    std::uint64_t stream_cycles = 0;     // stream-register cycles (vectors x registers passed)
    std::int64_t min_slack = INT64_MAX;  // smallest (deadline - arrival) over all Ifetches
    std::vector<std::uint32_t> program_bytes;   // per IQ
};

// Emits the scheduler's program, plans and places the Ifetch pipelines, assigns every
// entry its bytes in its IQ's text, and writes the fetched text into MEM.
sim::Program build_program(Compiler& cc, sched::Scheduler& sch, sim::Simulator& sim, const sim::Floorplan& fp,
                           const IfetchOptions& opt, IfetchStats* st);

}  // namespace comp
}  // namespace lpu
