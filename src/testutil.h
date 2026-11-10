// testutil.h - small helpers to build pipeline ops by hand (tests and examples).
#pragma once

#include <cstdio>
#include <string>

#include "sched.h"

namespace lpu {
namespace sched {

inline POp make_read(int g, int addr, int n, int m0 = 0, int stride = 1) {
    POp o;
    o.queue = sim::q_mem_read(g);
    o.ins.in.opcode = Opcode::MemRead;
    o.mk = MemKind::ReadSeq;
    o.mem_g = g;
    o.addr0 = addr;
    o.stride = stride;
    o.n = n;
    o.m0 = m0;
    o.kind = 'R';
    o.label = "Read";
    return o;
}

inline POp make_write(int g, int addr, int n, int m0 = 0) {
    POp o;
    o.queue = sim::q_mem_write(g);
    o.ins.in.opcode = Opcode::MemWrite;
    o.mk = MemKind::WriteSeq;
    o.mem_g = g;
    o.addr0 = addr;
    o.n = n;
    o.m0 = m0;
    o.kind = 'W';
    o.label = "Write";
    return o;
}

inline POp make_vxm(Opcode op, int n, int imm = 0, int m0 = 0) {
    POp o;
    o.queue = -1;
    o.ins.in.opcode = op;
    o.ins.in.imm = imm;
    o.n = n;
    o.m0 = m0;
    o.kind = 'V';
    o.label = "VXM";
    return o;
}

inline bool check(const std::string& name, bool ok, const std::string& detail) {
    std::printf("[%s] %s\n       %s\n", ok ? "PASS" : "FAIL", name.c_str(), detail.c_str());
    return ok;
}

}  // namespace sched
}  // namespace lpu
