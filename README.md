# LPU scheduler and ResNet compiler

A compiler and static list scheduler that lowers int8 ResNet-50/101/152 onto the TSP/LPU model,
emits programs for the chip's 144 instruction queues (IQs), runs them on a cycle simulator, and checks the logits
bit-exactly against an independent golden int8 model.

Tags used throughout: **[P]** taken from the paper (or patent US 11,360,934), **[I]** inferred
from the paper, **[A]** assumed (no published number), **[Ours]** a design choice of this project.
Latencies live in `planning/instruction-timing-table.xlsx`. Cells shaded yellow there were filled
with assumed values. `planning/xlsx_to_csv.py` exports it to the `.csv` that the binaries load,
and the values a run used are written to `results/latency_used.csv`.

## Headline results (from `results/`)

Bounded 640 B IQs refilled by Ifetch, the only mode. Before the preloaded-IQ mode was removed it gave
exactly the same cycles on all three nets (`results/preload_vs_ifetch.csv`). Batch 1,
host I/O excluded, Sync preamble and superlane stagger included.

| model | cycles | IPS @ 0.9 GHz (nominal) | latency @ 0.9 GHz | IPS @ 1 GHz | paper | check |
|---|---|---|---|---|---|---|
| ResNet-50 | 49,482 | 18,188 | 55.0 us | 20,209 | 20.4K IPS, "< 49 us" (measured, clock not stated) | bit-exact, 0 violations, 0 IQ underflows |
| ResNet-101 | 80,219 | 11,219 | 89.1 us | 12,466 | 14.3K (projected) | timing-only projection |
| ResNet-152 | 112,723 | 7,984 | 125.2 us | 8,871 | 10.7K (projected) | timing-only projection |

Ifetch on ResNet-50: 270 Ifetches on 53 IQs, 73.9 KB of text fetched from 5,342 MEM words, 5,342
read-port cycles, tightest Ifetch 1,329 cycles early. See `ARCHITECTURE.md` section 6.

The ResNet-50 match to the paper is close, but it rests on the [A] latencies and on this
lowering, not on the paper's compiler. ResNet-101/152 do not fit on-chip with this lowering
(weights are padded to 320x320 tiles, so they take about twice their raw size, plus gather maps);
with `--alias-weights` the overflow tiles (106 of 891 and 479 of 1251) reuse an earlier tile's slices.
That keeps the port and bank pattern but not the data, so those runs are timing projections and
their logits are not checked. The paper also gives projections for these two.

Roofline (`results/roofline.csv`): a [M x 320] x [320 x 5120] matmul reaches 11.7 TOPS at M = 1
and 803 TOPS at M = 2048 (peak 819 TOPS @ 1 GHz), with weights next to the MXM. Weights further
from the MXM cost 60-76 cycles of extra pipeline fill, which hurts most at small M. All 36 points are
bit-exact.

## Build and run

```
cd src/scheduler
make B=/tmp/lpu-build          # build dir is configurable (the project mount is noexec)
/tmp/lpu-build/tests           # timing-model unit tests (read/write, streaming add, conflicts)
/tmp/lpu-build/tests_layers    # 10 layer tests: matmul, partial sums, max pool, 3x3, bottleneck, stem, FC
/tmp/lpu-build/resnet --depth 50 --out ../../results
/tmp/lpu-build/resnet --depth 50 --lat vxm_add=2 ...        # override one latency (sensitivity)
/tmp/lpu-build/resnet --depth 50 --no-layer-overlap ...     # baseline: each layer waits for the previous
/tmp/lpu-build/resnet --depth 101 --alias-weights --out ../../results
/tmp/lpu-build/roofline --out ../../results
cd ../../results && python3 plots.py      # writes results/figures/*.png
```

After editing the xlsx, run `python3 ../../planning/xlsx_to_csv.py` to regenerate the CSV.
Run the binaries from `src/scheduler`, because they find `../../planning/instruction-timing-table.csv`
relative to the working directory (override this with `--latency`). ResNet-50 takes about 6 s for the
golden model, under 1 s to compile and schedule, and 3 s to simulate.

## Files

| file | contents |
|---|---|
| `lpusim.h/.cpp` | floorplan, latency table, stream-register model, event-driven ICU simulator with conflict checks |
| `sched.h/.cpp` | pipelines of rigid ops, reservation tables, greedy earliest-start list scheduler, program emission |
| `compiler.h/.cpp` | layouts, memory allocation, lowering of conv / max pool / avg pool / FC |
| `model.h/.cpp` | ResNet builders, random int8 weights, requantize calibration, golden int8 reference |
| `tests.cpp`, `tests_layers.cpp` | unit and layer tests (all pass), incl. the 144-IQ map and MEM dual-issue tests |
| `ifetch.h/.cpp` | 640 B IQs: instruction encoding, text in MEM, Ifetch pipeline planning |
| `main_resnet.cpp` | end-to-end driver, writes `results/resnetNN_*` (incl. `_iqbytes.csv`, `_iqfill.csv`) |
| `roofline.cpp` | Fig. 9 sweep |
| `../../results/plots.py` | Figs. 9, 10, 11 style plots and the comparison charts |

## Timing model

* T = N + d_func + delta. A stream moves one stream-register hop per cycle [P].
* Floorplan [I]: MXM_W at position 0, SXM, MEM_W43..W0 at 5..48, VXM at 49, MEM_E0..E43 at 50..93,
  SXM, MXM_E at 98. MEM index 0 is next to the VXM.
* 144 instruction queues (IQs) in the ICU [P], each dispatching up one column. The split by unit is
  from our planning doc [Ours]: per hemisphere 44 MEM + 14 SXM + 6 MXM, plus 16 VXM shared.
  IQ ids follow the column order, west to east (see the "IQ map" sheet in the xlsx).
  * MEM [Ours]: one IQ per slice with 2 ways, R-type (Read, Gather) and W-type (Write, Scatter), so
    a Read and a Write can issue from one IQ in one cycle. This follows the patent (col. 16): "at
    most one R-type and one W-type instruction concurrently if they are accessing different internal
    SRAM banks". The bank rule still applies.
  * MXM [Ours]: 3 IQs per plane, `lw`, `iw` and `abc`. LW for the next tile streams while ABC
    runs; IW (1 cycle) goes between them.
  * VXM: one IQ per ALU. SXM: 14 IQs per side, empty for ResNet-50 (no SXM op is modeled yet).
  * Every IQ other than MEM is single-issue [Ours].
* Latencies: MEM Read 1, Write 0, Gather 2 [A]; VXM add 1, mul 2, convert 1, relu 1, max 1 [A];
  MXM LW 1 per issue, IW 1, ABC 20 [I]; Sync preamble 35 cycles [P]; superlane stagger 19 [P].
* Superlane stagger is analytic. Every op is lane-local except the MXM reduction, which ABC's
  d_func already covers, so the stagger adds 19 cycles to the end of the program.
* The simulator flags a tag mismatch or an empty operand, two writers on one stream register, a read
  and a write to one bank in one cycle (pseudo-dual-port), an opcode on an IQ or way that cannot run
  it, and an IQ issuing more than its limit in one cycle.

## Scheduler

A *pipeline* is a set of ops tied by stream edges. Inside a pipeline the timing is rigid: a consumer
issues exactly when the producer's data reaches its position. The scheduler places whole pipelines
greedily at the earliest start that is free in:

1. the busy intervals of each issue channel, which is one way of one IQ (232 in all),
2. the MEM bank read and write intervals,
3. the pool of 16 VXM ALUs,
4. stream ids and directions. Stream reservations are rectangles in (diagonal x position) space,
   with diagonal D = t + 1 - q going East and D = t + 1 + q going West,
5. per-word memory dependencies (RAW, WAR and WAW), so a consumer can stream a tensor while its
   producer is still writing it.

`emit()` writes one program per IQ way. It adds the Sync preamble, turns gaps into NOPs (split at
65535), and encodes runs as Repeat.

## 144-IQ remap: before vs after (ResNet-50)

These runs predate bounded IQs (whole programs preloaded); with bounded IQs + Ifetch every row gives
the same cycles.

| run | cycles | check | IQs with code | entries |
|---|---|---|---|---|
| before: 200 queues (separate MEM read and write queues, LW+IW on one queue) | 49,482 | bit-exact | n/a (200-queue layout) | 41,373 |
| **after: 144 IQs** (MEM dual-issue, MXM lw / iw / abc) | **49,482** | bit-exact, 0 violations | 116 of 144 (the 28 SXM IQs are empty) | 41,677 |
| sensitivity: MEM IQ single-issue (Read and Write serialize) | 54,563 (+10.3%) | bit-exact | 115 of 144 | 41,360 |
| sensitivity: SXM 14 hops wide per side instead of 4 | 50,244 (+1.5%) | bit-exact | 116 of 144 | 41,544 |

The count does not change, for these reasons:
* **MEM:** a merged IQ that issues one R-type and one W-type per cycle enforces exactly what the two
  separate queues enforced before, one read port and one write port per slice. The bank rule was
  already checked. The binding constraint stays the same.
* **MXM:** moving IW to its own IQ frees nothing. IW already had to wait for its LW to finish and for the
  previous ABC to end, and the next LW had to wait for that IW. Those per-plane bounds, not
  queue sharing, set the timing.
* **SXM:** these IQs are new but unused.
* **Entries:** the count rises by 304 because LW and IW now sit on separate IQs, so each needs its own
  NOP gaps. An IW that directly followed its LW used to need none.

The sensitivity runs show which constraints would bite. If a MEM IQ could issue only one
instruction per cycle, every pipeline that reads and writes the same slice would serialize. The
partial-sum ping-pong buffers would then need separate slices, and the network gets 5,081 cycles
slower. A 14-hop-wide SXM adds 10 hops between the MXM and every MEM slice, which costs 762 cycles.
Run these with
`resnet --mem-single-issue` and `resnet --sxm-hops 14`. Their outputs are in `results/sensitivity/`.
Set `LPU_SCHED_DEBUG=<pipeline name prefix>` to see why a pipeline was delayed.

## Lowering choices (all [A])

* Activations are int8, one 320-byte word per pixel per 320-channel group.
* A tensor read by a 3x3 conv is "dy-packed": its channel block is stored up to 3 times at different
  superlane offsets, so one per-superlane Gather builds the dx = -1, 0, +1 taps of a row. The
  gather maps are compile-time constants in MEM.
* A conv is a sum over K-vectors of [M x 320] x [320 x 320] tiles. Each round runs up to 4
  (tile, M-chunk) slots on the 4 planes. The VXM sums the results. Int32 partial sums ping-pong
  between two MEM banks, near the VXM.
* Requantize in the VXM: y = relu(sat8((acc*Ma + res*Mr + 2^(S-1)) >> S)).
* Weights are loaded with LW (16 streams x 20 cycles) while the previous tile computes, then a
  1-cycle IW. Weight tiles get their own slices (MEM28-43), so an activation stream never holds the
  read port the next LW needs. Activations, maps and partial sums use MEM0-27.
* A tensor read by several planes at once is replicated across slices (one read port per slice).
* conv1 runs as a 1x1 conv over a host-side im2col (147 channels). FC writes raw int32 logits.

## Known limits

* Weights and the input are random (no checkpoint), so accuracy is not measured. The check is
  bit-exact agreement with the golden model.
* The power figure (Fig. 10 style) is an activity proxy with assumed per-unit weights (listed in
  `plots.py`). It is not a power model.
* No SXM ops are used (no transposes or shifts are needed by this lowering). The 28 SXM IQs exist,
  and their latencies are filled in the xlsx [A], but the simulator does not execute SXM ops yet.
* A single MEM store must be at most 4095 words (one bank), which caps the roofline sweep at M = 2048.
* The utilization of 20% of the 4-plane peak reflects the 64- and 128-channel layers, which fill
  only part of the 320 lanes.

## Visualization export

`resnet --events DIR` also writes four CSVs used by the interactive viewer (results are unchanged:
the recording never feeds back into placement, and the layers / trace / activity / summary files
are byte-identical with and without it):

| file | one row per |
|---|---|
| `<tag>_events.csv` | placed instruction (+ its Repeat run): cycle, IQ, way, opcode, pos, N, d_func, src/dst stream groups and direction, MEM slice / bank / address range, layer, pipeline, label |
| `<tag>_streams.csv` | stream transfer: edge, producer event, direction, stream id, diagonals [d0, d1) and positions [p0, p1]. At cycle t a value is at pos p where t - p (East) or t + p (West) is in [d0, d1) |
| `<tag>_pipes.csv` | pipeline: start cycle, layer, name |
| `<tag>_alloc.csv` | MEM allocation: kind (W weight, A activation, P partial sum, M gather map), slices, bank, base, size, pipeline count at alloc and at free |

```
/tmp/lpu-build/resnet --depth 50 --out ../../results --events ../../results/viz
cd ../../results/viz && python3 build_viz.py     # -> lpu-viz.html (template.html + data)
```
