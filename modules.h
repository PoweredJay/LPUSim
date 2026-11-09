// modules.h - shared pieces for the LPU simulator.
// MEM, VXM, MXM and SXM slices all inherit from the Slice class at the bottom.

#pragma once

#include <array>
#include <vector>

const int NUM_LANES = 320;     // 20 superlanes x 16 lanes
const int NUM_STREAMS = 32;    // streams in each direction

// One stream's data for one cycle: 1 byte per lane.
using Vector = std::array<unsigned char, NUM_LANES>;

enum Direction { EAST, WEST };

// Every instruction any slice can run. Add new ones here as the modules grow.
enum Opcode {
    NOP,

    // MEM
    MEM_READ,             // memory[address] -> stream dst
    MEM_WRITE,            // stream src1 -> memory[address]

    // VXM
    VXM_ADD,              // src1 + src2 -> dst
    VXM_SUB,
    VXM_MUL,
    VXM_MAX,
    VXM_RELU,             // max(0, src1) -> dst
    VXM_TANH,
    VXM_EXP,
    VXM_RSQRT,

    // MXM
    MXM_LOAD_WEIGHTS,     // streams -> weight buffer
    MXM_INSTALL_WEIGHTS,  // weight buffer -> matrix array
    MXM_MATMUL,           // activations on src1 x weights -> dst
    MXM_ACCUMULATE,       // src1 + src2 -> dst

    // SXM
    SXM_SHIFT_NORTH,      // move lanes up by `value`
    SXM_SHIFT_SOUTH,      // move lanes down by `value`
    SXM_PERMUTE,          // reorder lanes
    SXM_TRANSPOSE,        // 16 streams in -> 16 streams out, rows and columns swapped

    C2C_DESKEW,
    C2C_SEND,
    C2C_RECEIVE,
};

// One instruction. Just data: each slice decides what to do with it.
// Fields an instruction doesn't need are left at 0.
struct Instruction {
    Opcode opcode = NOP;
    int src1 = 0;               // first input stream
    int src2 = 0;               // second input stream
    int dst = 0;                // output stream
    Direction in_dir = EAST;    // direction of the input streams
    Direction out_dir = EAST;   // direction the result is sent
    int address = 0;            // MEM: memory address
    int value = 0;              // extra number, e.g. shift amount or MXM plane
    int delay = 1;              // cycles until the result appears (d_func)
};

// All the stream registers on the chip. Each position (one per slice) has
// 32 East-flowing and 32 West-flowing streams.
struct Streams {
    std::vector<std::array<Vector, NUM_STREAMS>> east;   // east[position][stream]
    std::vector<std::array<Vector, NUM_STREAMS>> west;   // west[position][stream]
    int num_positions = 0;

    Streams(int positions) {
        east.resize(positions);
        west.resize(positions);
        num_positions = positions;
    }

    // Call once at the end of every cycle: everything moves one position.
    void advance() {
        // East streams move to p + 1. Loop backwards so nothing is overwritten before it moves.
        for (int p = num_positions - 1; p > 0; p--) {
            east[p] = east[p - 1];
        }
        east[0] = {};

        // West streams move to p - 1, so loop forwards.
        for (int p = 0; p < num_positions - 1; p++) {
            west[p] = west[p + 1];
        }
        west[num_positions - 1] = {};
    }
};

// The base class. Every module inherits from this and fills in dispatch() and tick().
class Slice {
public:
    Slice(Streams& streams, int position) : streams_(streams), position_(position) {}
    virtual ~Slice() {}

    // Hand the slice an instruction to run.
    virtual void dispatch(Instruction& inst) = 0;

    // Run one clock cycle.
    virtual void tick(int cycle) = 0;

protected:
    // Stream number s at this slice's position. A slice can only touch streams where it sits.
    Vector& stream(Direction dir, int s) {
        if (dir == EAST) {
            return streams_.east[position_][s];
        }
        return streams_.west[position_][s];
    }

    Streams& streams_;   // shared by all slices
    int position_;       // where this slice sits on the chip (0 = West end)
};