#include <modules.h>
#include <algorithm>
#include <cmath>

class VXM : public Slice {
    void dispatch(Instruction& inst) override {
        switch(inst.opcode) {
            case NOP:
                return;
            case VXM_ADD:
                inst.dst = inst.src1 + inst.src2;              
            case VXM_SUB:
                inst.dst = inst.src1 - inst.src2;
            case VXM_MUL:
                inst.dst = inst.src1 * inst.src2;
            case VXM_MAX:
                inst.dst = std::max(inst.src1, inst.src2);
            case VXM_RELU:
                inst.dst = std::max(inst.src1, 0);
            case VXM_EXP:
                inst.dst = exp(inst.src1);
            case VXM_TANH:
                inst.dst = tanh(inst.src1);
            case VXM_RSQRT:
                inst.dst = 1/sqrt(inst.src1);      
        }
    };

    // Run one clock cycle.
    void tick(int cycle) override {
        return;
    };
};