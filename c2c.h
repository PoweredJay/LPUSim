#include <modules.h>

class C2C : public Slice {
    void dispatch(Instruction& inst) override {
        switch(inst.opcode) {
            case NOP:
                return;
            case C2C_DESKEW:
                return;            
            case C2C_RECEIVE:
                return;
            case C2C_SEND:
                return;   
        }
    };

    // Run one clock cycle.
    void tick(int cycle) override {
        return;
    };
};