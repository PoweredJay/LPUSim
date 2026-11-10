import math
import numpy as np

class Module:
    def __init__(self, name, row, col):
        self.name = name
        self.state = {}
        self.next_state = {}
        self.pos = (row, col)

    def inst_fetch(self, control):
        return control.fetch()
    
    # def execute(self, cycle, inputs, opcode):
    #     self.state = self.state
    
    def commit(self):
        self.state = self.next_state
        self.next_state = {}
    
    def run(modules, wires, num_cycles, trace):
        for cycle in range(num_cycles):
            for m in modules:
                m.execute(cycle, inputs, inst.opcode)
            for m in modules:
                m.commit()
            trace.record(cycle, modules, wires)

class VXM(Module):
    def execute(self, cycle, inputs, opcode):
        match opcode:
            case 0: # negate
                return not inputs
            case 1: # add
                return inputs[0] + inputs[1]
            case 2: # mul
                return inputs[0] * inputs[1]
            case 3: # sub
                return inputs[0] - inputs[1]
            case 4: # fixed <-> float
                return float(inputs)
            case 5: #ReLU
                return max(0,inputs)
            case 6: # tanh
                return math.tanh(inputs)
            case 7: # exp
                return math.exp(inputs)
            case 8: # reciprocal sqrt
                return 1/math.sqrt(inputs)

class MEM(Module):
    def execute(self, cycle, inputs, opcode):
        match opcode:
            case 0: # read
                return
            case 1: # write
                return
            case 2: # gather
                return 
            case 3: # scatter
                return 

class SXM(Module):
    def execute(self, cycle, inputs, opcode):
        match opcode:
            case 0: # shift
                return
            case 1: # permute
                return
            case 2: # distribute
                return 
            case 3: # rotate
                return 
            case 4: # transpose
                return 

class MXM(Module):
    def execute(self, cycle, inputs, opcode):
        match opcode:
            case 0: # load weights
                return
            case 1: # install weights
                return
            case 2: # activation buffer control
                return 
            case 3: # acc
                return 
    def macc(self, inputs):
        return np.matmul(inputs[0],inputs[1])

class ICU(Module):
    def __init__(self, name, col):
        self.name = name
        self.state = {}
        self.next_state = {}
        self.inst_queue = {}
        self.current_inst = 0
        self.pos = (0, col)

    def fetch(self):
        return self.inst_queue[self.current_inst]

    def broadcast(modules, op):
        for m in modules:
            # sync = 0
            if (op == 0):
                m.execute(cycle, null, "sync")
            # notify = 1
    
    def execute(self, cycle, inputs, opcode):
        if(opcode == "sync"):
            self.next_state = "sync"
        if(opcode == "notify"):
            self.next_state = "notify"

    def commit(self):
        if(self.next_state == "sync"):
            self.state = "sync"
        if(self.next_state == "notify"):
            self.state = "notify"
            self.next_state = "fetch"
        
        
            
