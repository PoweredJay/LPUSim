import heapq
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class Op:
    id: str
    kind: str            
    unit_type: str       # functional unit
    latency: int         # cycles delay
    deps: list = field(default_factory=list)   

@dataclass
class Unit:
    id: str
    unit_type: str
    location: tuple

def transit_delay(a: Unit, b: Unit) -> int:
    (r1, c1), (r2, c2) = a.location, b.location
    return abs(r1 - r2) + abs(c1 - c2)

def critical_path_priority(ops: dict) -> dict:
    
    dependents = defaultdict(list)
    for op in ops.values():
        for dep_id in op.deps:
            dependents[dep_id].append(op.id)

    priority = {}

    def longest_path(op_id):
        if op_id in priority:
            return priority[op_id]
        op = ops[op_id]
        if not dependents[op_id]:
            priority[op_id] = op.latency
        else:
            priority[op_id] = op.latency + max(longest_path(d) for d in dependents[op_id])
        return priority[op_id]

    for op_id in ops:
        longest_path(op_id)
    return priority

def list_schedule_order(ops: dict, priority: dict) -> list:
    
    indegree = {op_id: len(op.deps) for op_id, op in ops.items()}
    dependents = defaultdict(list)
    for op in ops.values():
        for dep_id in op.deps:
            dependents[dep_id].append(op.id)

    ready = [(-priority[op_id], op_id) for op_id, deg in indegree.items() if deg == 0]
    heapq.heapify(ready)

    order = []
    while ready:
        _, op_id = heapq.heappop(ready)
        order.append(op_id)
        for nxt in dependents[op_id]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(ready, (-priority[nxt], nxt))

    if len(order) != len(ops):
        raise ValueError("cycle detected in op graph — not a DAG")
    return order

def asap_schedule(ops: dict, units: list) -> list:
    # Returns a schedule sorted by cycle
    priority = critical_path_priority(ops)
    order = list_schedule_order(ops, priority)

    unit_by_id = {u.id: u for u in units}
    units_by_type = defaultdict(list)
    for u in units:
        units_by_type[u.unit_type].append(u)

    issue_slot = {u.id: set() for u in units}     
    result_ready = {}                              
    schedule = []

    for op_id in order:
        op = ops[op_id]
        candidates = units_by_type[op.unit_type]
        if not candidates:
            raise ValueError(f"no unit of type {op.unit_type!r} for op {op_id!r}")

        best_unit_id, best_cycle = None, None
        for u in candidates:
            
            earliest = 0
            for dep_id in op.deps:
                dep_unit_id, dep_ready = result_ready[dep_id]
                earliest = max(earliest, dep_ready + transit_delay(unit_by_id[dep_unit_id], u))

            
            cycle = earliest
            while cycle in issue_slot[u.id]:
                cycle += 1

            if best_cycle is None or cycle < best_cycle:
                best_unit_id, best_cycle = u.id, cycle

        issue_slot[best_unit_id].add(best_cycle)
        ready_cycle = best_cycle + op.latency
        result_ready[op_id] = (best_unit_id, ready_cycle)

        schedule.append({
            "cycle": best_cycle,
            "unit_id": best_unit_id,
            "op_id": op_id,
            "kind": op.kind,
            "ready_cycle": ready_cycle,
        })

    schedule.sort(key=lambda e: e["cycle"])
    return schedule

# test
if __name__ == "__main__":
    
    units = [
        Unit("mem0", "MEM", location=(0, 0)),
        Unit("mxm0", "MXM", location=(0, 1)),
        Unit("mxm1", "MXM", location=(0, 4)),   
        Unit("vxm0", "VXM", location=(0, 2)),
    ]

    ops = {
        "load":     Op("load", "load", "MEM", latency=2, deps=[]),
        "matmul_a": Op("matmul_a", "matmul", "MXM", latency=4, deps=["load"]),
        "matmul_b": Op("matmul_b", "matmul", "MXM", latency=4, deps=["load"]),
        "add":      Op("add", "add", "VXM", latency=1, deps=["matmul_a", "matmul_b"]),
    }

    schedule = asap_schedule(ops, units)
    for entry in schedule:
        print(f"cycle {entry['cycle']:>3}  {entry['unit_id']:<6} "
              f"{entry['op_id']:<10} ready@{entry['ready_cycle']}")

    total_len = max(e["ready_cycle"] for e in schedule)
    print(f"\ntotal: {total_len} cycles")
