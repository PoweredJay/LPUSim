// ifetch.h - bounded instruction queues: instruction text in MEM, Ifetch pipelines, and the
// mapping of every program entry to its place in its IQ's text.
//
// Model (tags: [P] patent US 11,360,934 / ISCA 2020, [Ours] this project):
//  * Each IQ holds 640 bytes [P col. 12]. NOP = 1 byte, every other instruction 4 bytes [P col. 10;
//    encoding Ours]. The first 640 bytes of each IQ are loaded before the Sync (boot) [Ours].
//  * The rest of the program lives in MEM as constants [P col. 16 "microprograms"], allocated in
//    never-used space and counted against capacity. One MEM word is one 320-byte text vector.
//  * One Ifetch climbs the 20 tiles and collects one full 320-byte vector, 16 B per tile. A refill
//    is Ifetch + Repeat 1 = two vectors = 640 B, filling the IQ [P col. 12]; the last refill of an
//    IQ may be one vector. Bytes enter the IQ at 16 B per cycle [ISCA 2020]: chunk j of a vector
//    fetched at cycle c lands at c + j, and the Repeat is spaced 20 cycles (40 cycles per 640 B) [Ours].
//  * Each vector is a pipeline (MEM Read of one text word -> stream -> Ifetch on the IQ's fetch way)
//    that the scheduler places like any other, so it reserves the read port, the bank and the
//    streams; the second vector goes exactly 20 cycles after the first on the same streams and is
//    emitted as the Ifetch's Repeat. Refills are planned after the compute schedule, in deadline
//    order, using only resources the compute pipelines left free.
//  * Per 16-byte chunk, a refill must start late enough that every chunk has room in the 640 B IQ
//    when it lands, and early enough that every instruction's chunk lands before it issues. The
//    simulator re-checks both per chunk: no underflow (patent col. 13: no IQ may run empty) and no
//    overflow.
#pragma once

#include <cstdint>
#include <vector>

#include "compiler.h"

namespace lpu {
namespace comp {

struct IfetchOptions {
    int chunk_bytes = 640;    // max bytes per refill (Ifetch + Repeat 1 = 2 x 320 B) [P col. 12]
    int candidates = 8;       // MEM slices tried per Ifetch, nearest to the IQ first
    int mem_headroom = 48;    // spare bytes per piece of a MEM IQ, for text Reads placed on it later
};

struct IfetchStats {
    int ifetches = 0, iqs_used = 0, iqs_fetching = 0, late = 0, shrunk = 0, rejected = 0;
    std::uint64_t text_bytes = 0;        // all IQ programs
    std::uint64_t preload_bytes = 0;     // loaded before the Sync
    std::uint64_t fetched_bytes = 0;     // brought in by Ifetch
    std::uint64_t text_words = 0;        // MEM words holding fetched text (320 B each)
    std::uint64_t ifetch_vectors = 0;    // text vectors fetched (2 per full refill)
    std::uint64_t read_port_cycles = 0;  // MEM read-port cycles spent on Ifetch
    std::uint64_t stream_cycles = 0;     // stream-register cycles (vectors x registers passed)
    std::int64_t min_slack = INT64_MAX;  // smallest (latest allowed start - start) over all refills
    std::vector<std::uint32_t> program_bytes;   // per IQ
};

// Emits the scheduler's program, plans and places the Ifetch pipelines, assigns every
// entry its bytes in its IQ's text, and writes the fetched text into MEM.
sim::Program build_program(Compiler& cc, sched::Scheduler& sch, sim::Simulator& sim, const sim::Floorplan& fp,
                           const IfetchOptions& opt, IfetchStats* st);

}  // namespace comp
}  // namespace lpu
