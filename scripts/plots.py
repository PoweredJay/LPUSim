#!/usr/bin/env python3
"""Figures from the CSVs the simulator writes. Every plotted "ours" value is a simulated point.

    python3 scripts/plots.py [--results results] [--planning planning]

Writes
  results/figures/roofline.png           roofline: three weight placements, log-log like the paper's Fig. 9
  results/figures/roofline_vs_paper.png  paper vs ours vs the 32-slice experiment, at the paper's intensities
  results/figures/latency_sensitivity.png  tornado chart: each [A] latency halved and doubled (ResNet-50)
from
  results/roofline.csv               three weight placements (src/roofline.cpp)
  results/roofline_wide_weights.csv  experiment: weights spread over 32 slices per hemisphere
  planning/paper_fig9.csv            paper reference points (not simulated; see the file header)
  results/latency_sensitivity.csv    latency sweep (scripts/sensitivity.py)

The loaders and the roof constants are also used by scripts/report.py.
"""
import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reference categorical palette (light mode). Slots 1-3 = the three placements, slot 7 = the
# experiment (a different entity, so it never reuses a placement's color). Neutral inks for the rest.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
EXPERIMENT = "#6250d6"
SURFACE, INK, INK2, MUTED, GRID, PAPER = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1", "#8a8985"

CLOCK_GHZ = 1.0      # TOPS = ops per cycle / 1000 at 1 GHz (the paper's Fig. 9 axis)
NOMINAL_GHZ = 0.9    # the chip's nominal clock
LANES = 320
PEAK_TOPS = 4 * 2 * LANES * LANES / 1000.0          # 4 planes x 320 x 320 MACs x 2 ops = 819.2
# Weight-loading roofs (bytes of weights that can reach the MXMs per cycle, whole chip):
#   ours: an LW streams 16 rows per cycle and both planes of a hemisphere read the same 16 weight
#         slices, so each hemisphere delivers 16 streams x 320 B per cycle.
#   paper: 32 streams per direction load both planes of a hemisphere at once (ISCA 2020 p.152;
#         patent US 11,822,510), i.e. twice as fast.
OUR_WEIGHT_BPC = 2 * 16 * LANES                     # 10,240 B/cycle
PAPER_WEIGHT_BPC = 2 * 32 * LANES                   # 20,480 B/cycle


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def load_roofline(path):
    """{placement: [(M, ops_per_byte, cycles, tops, ok)]}, placements in file order."""
    out = {}
    for r in read_csv(path):
        ok = r["bit_exact"] == "1" and r["violations"] == "0" and r["iq_underflows"] == "0"
        out.setdefault(r["placement"], []).append(
            (int(r["M"]), int(r["ops_per_weight_byte"]), int(r["cycles"]), float(r["tops_1ghz"]), ok))
    for v in out.values():
        v.sort()
    return out


def load_paper(path):
    return [(int(r["ops_per_weight_byte"]), float(r["tops_1ghz"])) for r in read_csv(path)]


def pair_with_paper(points, paper, tol=0.01):
    """[(paper_x, paper_tops, M, sim_tops, pct)] for paper points with a simulated 2M within tol."""
    out = []
    for px, pt in paper:
        best = min(points, key=lambda p: abs(p[1] - px))
        if abs(best[1] - px) / px <= tol:
            out.append((px, pt, best[0], best[3], 100.0 * (best[3] / pt - 1.0)))
    return out


def roof(x, bytes_per_cycle):
    return min(PEAK_TOPS, x * bytes_per_cycle / 1000.0)


def _style(plt):
    plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})


def _axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def _legend(ax, **kw):
    leg = ax.legend(frameon=True, **kw)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)


def roofline_figure(results, planning, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    rl = load_roofline(os.path.join(results, "roofline.csv"))
    paper = load_paper(os.path.join(planning, "paper_fig9.csv"))
    names = list(rl)

    _style(plt)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    fig.patch.set_facecolor(SURFACE)
    _axes(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, lw=0.8)

    # the roof: compute peak and the two weight-loading limits (computed, not simulated)
    xs = [2 ** (k / 8) for k in range(0, 8 * 14)]
    ax.plot(xs, [roof(x, OUR_WEIGHT_BPC) for x in xs], color=INK2, lw=1.2, ls=(0, (5, 3)), zorder=1,
            label="roof, our layout: weights load at 16 streams per side (ridge at 80 ops/B)")
    ax.plot(xs, [roof(x, PAPER_WEIGHT_BPC) for x in xs], color=MUTED, lw=1.2, ls=(0, (1, 2)), zorder=1,
            label="roof, paper's hardware: 32 streams per side load both planes at once (ridge at 40 ops/B)")
    ax.text(2.2, PEAK_TOPS * 1.07, f"compute limit: {PEAK_TOPS:.0f} TOPS (all 4 MXM planes busy every cycle)",
            color=INK2, fontsize=9)

    ax.plot([p[0] for p in paper], [p[1] for p in paper], ls="none", marker="D", ms=7, mfc=SURFACE, mec=PAPER,
            mew=1.6, label="paper, Fig. 9 (measured on the chip; not simulated)", zorder=2)
    for i, n in enumerate(names):
        pts = rl[n]
        ax.plot([p[1] for p in pts], [p[3] for p in pts], color=SERIES[i], lw=2, marker=MARKERS[i], ms=6.5,
                mec=SURFACE, mew=1.2, label=f"simulated: {n}", zorder=3 + len(names) - i)
        bad = [p for p in pts if not p[4]]
        if bad:
            ax.scatter([p[1] for p in bad], [p[3] for p in bad], s=130, facecolors="none", edgecolors="#e34948",
                       lw=1.5, zorder=9, label="not bit-exact / violation" if i == 0 else None)
    # direct labels at 128 ops/B, where the three placements separate most
    for i, n in enumerate(names):
        p = next(q for q in rl[n] if q[1] == 128)
        off, ha = [((-10, 8), "right"), ((12, 4), "left"), ((12, -10), "left")][i]
        ax.annotate(f"{n.split(' (')[0].replace('weights ', '')}: {p[3]:.0f}", (p[1], p[3]), xytext=off,
                    textcoords="offset points", ha=ha, va="center", fontsize=9, color=INK2)

    ticks = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels([f"{t:,}" for t in ticks])
    yt = [5, 10, 20, 50, 100, 200, 500, 1000]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels([f"{t:,}" for t in yt])
    ax.set_xlim(1.8, 9500)
    ax.set_ylim(6, 1300)
    ax.set_xlabel("operational intensity (ops per weight byte = 2 x rows M), log scale")
    ax.set_ylabel(f"throughput (TOPS at {CLOCK_GHZ:g} GHz), log scale")
    ax.set_title("Roofline: [M x 320] x [320 x 5120] int8 matmul on the simulator\n"
                 f"{CLOCK_GHZ:g} GHz clock (as in the paper's Fig. 9; at the {NOMINAL_GHZ:g} GHz nominal clock "
                 f"multiply by {NOMINAL_GHZ:g}). Every colored point is a simulated run, checked bit-exact.",
                 loc="left", fontsize=11.5, color=INK, pad=10)
    _legend(ax, loc="lower right", fontsize=8.5)
    fig.text(0.01, 0.003, "Read it: left of ~80 ops/B the chip waits on weights (sloped roof); right of it the "
             "MXMs are the limit (flat roof). Points can only sit under the roof.",
             fontsize=8.5, color=INK2)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_png


def comparison_figure(results, planning, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rl = load_roofline(os.path.join(results, "roofline.csv"))
    wide = load_roofline(os.path.join(results, "roofline_wide_weights.csv"))
    paper = load_paper(os.path.join(planning, "paper_fig9.csv"))
    name0, wname = list(rl)[0], list(wide)[0]
    ours = pair_with_paper(rl[name0], paper)
    exp = {p[0]: p for p in pair_with_paper(wide[wname], paper)}

    _style(plt)
    fig, ax = plt.subplots(figsize=(13, 6.4))
    fig.patch.set_facecolor(SURFACE)
    _axes(ax)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    n = len(ours)
    w = 0.27
    xs = list(range(n))
    bars = [
        ([p[1] for p in ours], PAPER, "paper, Fig. 9 (measured)", -w),
        ([p[3] for p in ours], SERIES[0], f"ours, default layout: {name0}", 0),
        ([exp[p[0]][3] if p[0] in exp else 0 for p in ours], EXPERIMENT, f"ours, experiment: {wname}", w),
    ]
    for vals, col, lab, off in bars:
        b = ax.bar([x + off for x in xs], vals, width=w - 0.03, color=col, label=lab, zorder=2)
        for rect, v in zip(b, vals):
            if v:
                ax.text(rect.get_x() + rect.get_width() / 2, v + 8, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK2, rotation=90)
    ax.axhline(PEAK_TOPS, color=MUTED, lw=1, ls=(0, (2, 2)))
    ax.text(-0.55, PEAK_TOPS + 10, f"peak {PEAK_TOPS:.0f}", ha="left", fontsize=8.5, color=INK2)
    # difference rows above the bars
    for x, p in zip(xs, ours):
        ax.text(x, 1010, f"{p[4]:+.0f}%", ha="center", fontsize=9.5, color=INK, fontweight="bold")
        if p[0] in exp:
            ax.text(x, 955, f"{exp[p[0]][4]:+.0f}%", ha="center", fontsize=9, color=EXPERIMENT)
    ax.text(-0.75, 1010, "default vs paper", ha="right", fontsize=9, color=INK)
    ax.text(-0.75, 955, "experiment vs paper", ha="right", fontsize=9, color=EXPERIMENT)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{2 * p[2]:,}" for p in ours], fontsize=9.5)
    shifted = [p for p in ours if 2 * p[2] != p[0]]
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(0, 1060)
    ax.set_yticks([0, 200, 400, 600, 800])
    ax.set_xlabel("operational intensity (ops per weight byte); every simulated value is a real run at that intensity")
    ax.set_ylabel(f"throughput (TOPS at {CLOCK_GHZ:g} GHz)")
    ax.set_title("Ours vs the paper at the same operational intensity (1 GHz clock)\n"
                 "Paper's " + ", ".join(f"{p[0]:,}" for p in shifted) + " ops/B are compared with our "
                 + ", ".join(f"{2 * p[2]:,}" for p in shifted) + ". Its 8,179 point is left out: "
                 "M = 4096 does not fit one MEM bank.",
                 loc="left", fontsize=11, color=INK, pad=28)
    _legend(ax, loc="upper left", fontsize=9, bbox_to_anchor=(0.0, 1.075), ncol=3)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_png


def resnet_figure(results, out_png):
    """Images per second, batch 1: ours at 0.9 GHz (nominal) and 1 GHz vs the paper, ResNet-50/101/152.
    ResNet-101/152 are projections on both sides: ours alias weight tiles that do not fit on-chip
    (timing only), and the paper projects them from ResNet-50 (ISCA 2020 p.155)."""
    import json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = (50, 101, 152)
    s = {}
    for d in depths:
        with open(os.path.join(results, f"resnet{d}_summary.json")) as f:
            s[d] = json.load(f)
    BLUE_LIGHT = "#8fb9ec"   # lighter step of the series-1 blue: same entity (ours), other clock

    _style(plt)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.patch.set_facecolor(SURFACE)
    _axes(ax)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    w = 0.26
    series = [
        ("ours @ 0.9 GHz (nominal clock)", SERIES[0], lambda x: x["ips_0.9GHz"], -w),
        ("ours @ 1 GHz", BLUE_LIGHT, lambda x: x["ips_1GHz"], 0.0),
        ("paper (ISCA 2020)", PAPER, lambda x: x["paper_ips"], w),
    ]
    for label, col, get, off in series:
        for i, d in enumerate(depths):
            proj = d != 50
            v = get(s[d]) / 1000.0
            ax.bar(i + off, v, width=w - 0.03, color=col, edgecolor=SURFACE if not proj else INK2,
                   hatch="///" if proj else None, lw=0.8 if proj else 0, zorder=2,
                   label=label if i == 0 else None)
            ax.text(i + off, v + 0.3, f"{v:.1f}K", ha="center", va="bottom", fontsize=9.5, color=INK)
    for i, d in enumerate(depths):
        x = s[d]
        if d == 50:
            note = (f"{x['cycles']:,} cycles · bit-exact\n"
                    f"{x['latency_us_0.9GHz']:.1f} us @ 0.9 GHz, {x['latency_us_1GHz']:.1f} us @ 1 GHz\n"
                    "paper: measured, \"< 49 us\"")
        else:
            note = (f"{x['cycles']:,} cycles · timing only\n"
                    f"ours: {x['weight_tiles_aliased']} of {x['weight_tiles']} weight tiles\n"
                    "do not fit on-chip\n"
                    "paper: projected from ResNet-50")
        ax.annotate(note, xy=(i, 0), xycoords=("data", "axes fraction"), xytext=(0, -46),
                    textcoords="offset points", ha="center", va="top", fontsize=8.5, color=INK2, linespacing=1.4)
    ax.set_xticks(range(len(depths)))
    ax.set_xticklabels([f"ResNet-{d}" + ("" if d == 50 else "\n(projection)") for d in depths], fontsize=11)
    ax.tick_params(axis="x", pad=4)
    ymax = max(max(s[d]["ips_1GHz"], s[d]["paper_ips"]) for d in depths) / 1000.0
    ax.set_ylim(0, ymax * 1.15)
    ax.set_ylabel("images per second, batch 1 (thousands)")
    ax.set_title("ResNet inference throughput on the simulator vs the paper\n"
                 "Hatched bars are projections (ResNet-101/152). Host I/O excluded; Sync preamble and superlane "
                 "stagger included.", loc="left", fontsize=11.5, color=INK, pad=10)
    _legend(ax, loc="upper right", fontsize=9.5)
    fig.subplots_adjust(bottom=0.24)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out_png


def sensitivity_figure(results, out_png):
    """Tornado chart: change in ResNet-50 cycles when each assumed [A] value is halved or doubled alone."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    from matplotlib.transforms import blended_transform_factory

    rows = read_csv(os.path.join(results, "latency_sensitivity.csv"))
    base = int(rows[0]["baseline_cycles"])
    HALF, DOUBLE = SERIES[0], SERIES[1]
    FRAME_ALPHA = 0.35      # plot frame and legend box

    def signed(v, fmt="{:,.0f}"):
        return ("+" if v > 0 else "−" if v < 0 else "") + fmt.format(abs(v))

    style = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 12, "axes.linewidth": 0.8,
             "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
             "xtick.direction": "in", "xtick.major.size": 5, "xtick.minor.size": 3, "hatch.linewidth": 0.6}
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(10, 6.2))
        fig.patch.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        left = blended_transform_factory(ax.transAxes, ax.transData)
        lo = hi = 0
        for i, r in enumerate(rows):
            assumed = int(r["assumed"])
            dh = int(r["half_cycles"]) - base if r["half_cycles"] else None
            dd = int(r["double_cycles"]) - base
            for d, col in ((dh, HALF), (dd, DOUBLE)):
                if d:
                    ax.barh(i, d, height=0.6, color=col, edgecolor=INK, lw=0.6, zorder=2)
                    ax.annotate(signed(d), (d, i), xytext=(5 if d > 0 else -5, 0), textcoords="offset points",
                                ha="left" if d > 0 else "right", va="center", fontsize=11, color=INK)
                    lo, hi = min(lo, d), max(hi, d)
            if dh is None:
                ax.annotate(f"(cannot halve {assumed} cycle)", (0, i), xytext=(-5, 0), textcoords="offset points",
                            ha="right", va="center", fontsize=10, style="italic", color=INK2)
            if dd == 0 and not dh:
                ax.annotate("0", (0, i), xytext=(5, 0), textcoords="offset points", va="center", fontsize=11, color=INK)
            hops = r["unit"] == "hops"
            ax.text(-0.17, i, r["instruction"] + ("*" if hops else ""), transform=left, ha="right", va="center",
                    fontsize=12, color=INK)
            ax.text(-0.08, i, f"{assumed}" + (" hops" if hops else ""), transform=left, ha="center", va="center",
                    fontsize=12, color=INK)
        for x, ha, t in ((-0.17, "right", "Instruction"), (-0.08, "center", "Assumed\nlatency\n(cycles)")):
            ax.text(x, 1.015, t, transform=ax.transAxes, ha=ha, va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.axvline(0, color=INK, lw=1, zorder=3)
        ax.set_yticks([])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(v)))
        ax.set_xlabel("Δ cycles", labelpad=6, fontsize=13)
        top = ax.secondary_xaxis("top", functions=(lambda v: 100 * v / base, lambda v: v * base / 100))
        top.xaxis.set_major_locator(MultipleLocator(0.2))
        top.xaxis.set_major_formatter(FuncFormatter(lambda v, _: signed(round(v, 1) + 0.0, "{:.1f}")))
        top.set_xlabel("Δ cycles (% of baseline)", labelpad=6, fontsize=13)
        for sp in (*ax.spines.values(), *top.spines.values()):
            sp.set_alpha(FRAME_ALPHA)
        ax.spines["top"].set_visible(False)   # the percent axis draws this edge
        leg = ax.legend(handles=[Patch(facecolor=HALF, edgecolor=INK, lw=0.6, label="latency × 0.5"),
                           Patch(facecolor=DOUBLE, edgecolor=INK, lw=0.6, label="latency × 2")],
                  loc="lower right", frameon=True, fancybox=False, fontsize=11)
        leg.get_frame().set_linewidth(0.6)
        leg.get_frame().set_alpha(None)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor((*to_rgb(INK), FRAME_ALPHA))
        fig.subplots_adjust(left=0.35, right=0.97, top=0.78, bottom=0.19)
        fig.text(0.02, 0.955, "ResNet-50 sensitivity to assumed latencies", ha="left", va="center", fontsize=16,
                 fontweight="bold", color=INK)
        fig.text(0.02, 0.91, f"One assumed [A] value changed at a time; baseline = {base:,} cycles", ha="left",
                 va="center", fontsize=12, color=INK2)
        fig.text(0.02, 0.025, "* Floorplan distance in hops, not an instruction latency. Latencies are whole cycles, so a "
                 "1-cycle latency has no halved run.\n  Sorted by range including the baseline. "
                 "Every plotted run is bit-exact.", fontsize=10.5, color=INK2, linespacing=1.3)
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=200, facecolor="white")
        plt.close(fig)
    return out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--planning", default=os.path.join(ROOT, "planning"))
    ap.add_argument("--only", choices=["all", "roofline", "resnet", "sensitivity"], default="all")
    a = ap.parse_args()
    fig_dir = os.path.join(a.results, "figures")
    pngs = []
    if a.only in ("all", "roofline"):
        pngs.append(roofline_figure(a.results, a.planning, os.path.join(fig_dir, "roofline.png")))
        pngs.append(comparison_figure(a.results, a.planning, os.path.join(fig_dir, "roofline_vs_paper.png")))
    if a.only in ("all", "resnet"):
        pngs.append(resnet_figure(a.results, os.path.join(fig_dir, "resnet_ips.png")))
    if a.only in ("all", "sensitivity"):
        pngs.append(sensitivity_figure(a.results, os.path.join(fig_dir, "latency_sensitivity.png")))
    for png in pngs:
        print(f"wrote {os.path.relpath(png, ROOT)}")


if __name__ == "__main__":
    main()
