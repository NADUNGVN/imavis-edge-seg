"""V20 schematic figures in pseudo-3D block style (layers as extruded blocks, tilted
image/feature planes, colour per capacity, legend box).

fig2_pipeline_3d_v20     : inference pipeline (probe -> risk -> policy -> engine) + cost lane
fig3_architecture_3d_v20 : shared elastic encoder-decoder and the four static candidates

Images come from reports/qualitative_assets_v20 (held-out ACDC rain image, Run A).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from PIL import Image
from pace_style_v13 import CANDIDATE_COLORS, FULL_WIDTH_MM, INK, MUTED, apply_style, mm_to_inches, save_all
from render_palette import colorize

ROOT = Path(__file__).parents[3]
GEN = Path(__file__).parents[1] / "generated"
ASSETS = ROOT / "reports/qualitative_assets_v20"
AID = "acdc-rain_GP020402_frame_000863_rgb_anon"
LEVELS = ("tiny", "small", "medium", "large")
RED = "#D62728"
DX, DY = 0.55, 0.38  # depth direction per unit depth


def shade(color, f):
    c = np.array(plt.matplotlib.colors.to_rgb(color))
    return tuple(np.clip(c * f + (1 - f) * (1 if f > 1 else 0), 0, 1)) if f <= 1 else tuple(np.clip(c + (1 - c) * (f - 1), 0, 1))


def block(ax, x, y, w, h, d, color, z=3, hatch=None, alpha=1.0, lw=0.5):
    """Extruded box: front face (x,y,w,h), extruded by depth d up-right."""
    ox, oy = d * DX, d * DY
    front = Rectangle((x, y), w, h, fc=color, ec=INK, lw=lw, zorder=z, hatch=hatch, alpha=alpha)
    top = Polygon([(x, y + h), (x + ox, y + h + oy), (x + w + ox, y + h + oy), (x + w, y + h)],
                  closed=True, fc=shade(color, 1.35), ec=INK, lw=lw, zorder=z, alpha=alpha)
    side = Polygon([(x + w, y), (x + w + ox, y + oy), (x + w + ox, y + h + oy), (x + w, y + h)],
                   closed=True, fc=shade(color, 0.72), ec=INK, lw=lw, zorder=z, alpha=alpha)
    for p in (side, top, front):
        ax.add_patch(p)
    return (x + w + ox, y + h / 2 + oy / 2)


def layer_stack(ax, x, y, specs, color, gap=0.25, z0=3):
    """specs: list of (h, d) with front faces of width d*?; draws a funnel of thin slabs."""
    cx = x
    right = None
    for i, (h, w) in enumerate(specs):
        right = block(ax, cx, y - h / 2, w, h, w * 2.2, color, z=z0 + i)
        cx += w + gap
    return right


def tilted_image(ax, img, x, y, w, h, skew=18, z=4, border=INK):
    tr = Affine2D().skew_deg(0, skew).translate(0, 0) + ax.transData
    tr = Affine2D().translate(-x, -y).skew_deg(0, skew).translate(x, y) + ax.transData
    im = ax.imshow(img, extent=(x, x + w, y, y + h), transform=tr, zorder=z, interpolation="bilinear")
    sk = np.tan(np.radians(skew))
    ax.add_patch(Polygon([(x, y), (x + w, y + w * sk), (x + w, y + h + w * sk), (x, y + h)], closed=True,
                         fill=False, ec=border, lw=0.6, zorder=z + 1))
    return im


def arrow(ax, a, b, color=INK, lw=1.0, rad=0.0, ls="-", z=6):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, zorder=z))


def label(ax, x, y, s, size=6.5, bold=False, color=INK, ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, fontweight="bold" if bold else "normal", color=color, ha=ha, va=va, zorder=10, **kw)


def flat_box(ax, x, y, w, h, text, fc, tc="white", size=6.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.25", fc=fc, ec="none", zorder=5))
    label(ax, x + w / 2, y + h / 2, text, size=size, color=tc, bold=True)


def load_rgb(what, size=(256, 128)):
    ext = "jpg" if what == "rgb" else "png"
    im = Image.open(ASSETS / f"{AID}_{what}.{ext}")
    im = im.resize(size, Image.Resampling.BILINEAR if what == "rgb" else Image.Resampling.NEAREST)
    a = np.asarray(im)
    return a if what == "rgb" else colorize(a)


def legend_box(ax, x, y, w, h, items):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.8", fc="white", ec=INK,
                                lw=0.8, zorder=8))
    n = len(items)
    cols = 1 if w < 25 else 2
    rows = (n + 1) // cols
    for k, (kind, payload, text) in enumerate(items):
        c, r = divmod(k, rows)
        ix = x + 0.6 + c * w / cols
        iy = y + h - (r + 0.75) * h / rows
        if kind == "block":
            for j, col in enumerate(payload):
                block(ax, ix + j * 0.75, iy - 0.35, 0.45, 0.7, 0.45, col, z=9, lw=0.4)
            tx = ix + len(payload) * 0.75 + 0.4
        elif kind == "arrow":
            ax.add_patch(FancyArrowPatch((ix, iy), (ix + 2.0, iy), arrowstyle="-|>", mutation_scale=6,
                                         color=payload[0], lw=1.0, ls=payload[1], zorder=9))
            tx = ix + 2.4
        elif kind == "plane":
            ax.add_patch(Polygon([(ix, iy - 0.45), (ix + 0.9, iy - 0.1), (ix + 0.9, iy + 0.55), (ix, iy + 0.2)],
                                 closed=True, fc=payload, ec=INK, lw=0.4, zorder=9))
            tx = ix + 1.3
        elif kind == "chip":
            flat_box(ax, ix, iy - 0.4, 1.6, 0.8, "", payload)
            tx = ix + 2.0
        label(ax, tx, iy, text, size=6, ha="left")


# ======================================================================== Fig. 2
def fig_pipeline():
    apply_style()
    W = mm_to_inches(FULL_WIDTH_MM)
    fig = plt.figure(figsize=(W, W * 0.44))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 47); ax.set_aspect("equal"); ax.axis("off")

    # --- input
    tilted_image(ax, load_rgb("rgb"), 1.0, 26, 9, 4.5)
    label(ax, 5.5, 23.8, "input $x$\n(preprocessed tensor)", size=6)
    # --- tiny probe network
    arrow(ax, (10.5, 30.5), (13.0, 30.5), RED, lw=1.3)
    end = layer_stack(ax, 13.2, 30.5, [(7, 0.6), (5.5, 0.7), (4, 0.8), (5.5, 0.7), (7, 0.6)], CANDIDATE_COLORS["tiny"])
    label(ax, 17.5, 39.6, "(1) Tiny engine (probe)", bold=True)
    label(ax, 17.5, 24.8, "always executed", size=5.8, color=MUTED)
    # --- tiny output + entropy planes
    arrow(ax, (end[0] + 0.3, end[1]), (25.2, 33.0), RED, lw=1.3)
    tilted_image(ax, load_rgb("tiny"), 25.4, 31.2, 7, 3.5)
    label(ax, 29.0, 38.2, "tiny segmentation", size=5.8)
    ent = np.load(ASSETS / f"{AID}_entropy.npy").astype(np.float32)
    ent_rgb = (plt.get_cmap("magma")(np.clip(ent / np.log(19), 0, 1))[..., :3] * 255).astype(np.uint8)
    arrow(ax, (end[0] + 0.3, end[1] - 0.6), (25.2, 26.0), INK)
    tilted_image(ax, ent_rgb, 25.4, 23.8, 7, 3.5)
    label(ax, 29.0, 22.6, "pixel entropy", size=5.8)
    # --- s(x)
    ax.add_patch(plt.Circle((37.0, 27.0), 1.6, fc="#3B5BA9", ec="none", zorder=6))
    label(ax, 37.0, 27.0, "$s(x)$", size=6.5, color="white", bold=True)
    arrow(ax, (33.2, 27.5), (35.3, 27.1))
    label(ax, 37.0, 24.3, "(2) mean entropy", size=5.8)
    # --- calibrators -> risk bars
    rx, ry = 41.5, 22.5
    arrow(ax, (38.7, 27.0), (41.2, 27.0))
    ax.add_patch(FancyBboxPatch((rx, ry), 10, 9.5, boxstyle="round,pad=0.1,rounding_size=0.6", fc="#FDF2F8",
                                ec="#CC79A7", lw=0.8, zorder=4))
    risks = [0.189, 0.155, 0.130, 0.110]
    for k, (lv, r) in enumerate(zip(LEVELS, risks)):
        bx = rx + 1.2 + k * 2.2
        block(ax, bx, ry + 1.4, 1.2, r * 32, 0.9, CANDIDATE_COLORS[lv], z=6, lw=0.4)
        label(ax, bx + 0.6, ry + 0.75, lv[0].upper(), size=5.6)
    ax.plot([rx + 0.8, rx + 9.6], [ry + 1.4 + 0.062 * 32] * 2, color=INK, lw=0.7, ls=(0, (2, 1.2)), zorder=7)
    label(ax, rx + 9.9, ry + 1.4 + 0.062 * 32, r"$\tau$", size=6.5, ha="left")
    label(ax, rx + 5, ry + 11.0, r"(3) calibrated risk $\hat r_\ell = g_\ell(s(x))$", bold=True)

    # --- hardware lane
    ax.add_patch(FancyBboxPatch((1.0, 1.5), 51.5, 15.0, boxstyle="round,pad=0.1,rounding_size=0.8", fc="#F1F5F9",
                                ec="none", zorder=1))
    label(ax, 2.0, 15.4, "measured once per device", size=6, color=MUTED, ha="left", style="italic")
    flat_box(ax, 2.5, 9.2, 8.5, 3.0, "AGX Xavier\nTensorRT FP16", "#4C72B0", size=5.6)
    flat_box(ax, 2.5, 4.4, 8.5, 3.0, "Hailo-8\nINT8 HEF", "#C44E52", size=5.6)
    label(ax, 17.0, 12.8, "complete route cost $C_h(\\ell)$ (ms)", size=6, bold=True)
    costs = {"AGX": [1.8, 4.2, 9.4, 19.5], "Hailo": [44.3, 58.1, 76.2, 116.3]}
    for row, (name, vals) in enumerate(costs.items()):
        yy = 10.0 - row * 4.8
        for k, (lv, v) in enumerate(zip(LEVELS, vals)):
            ax.add_patch(Rectangle((13.0 + k * 4.6, yy), 4.2, 2.2, fc=CANDIDATE_COLORS[lv], ec="white", lw=0.6, zorder=5))
            label(ax, 15.1 + k * 4.6, yy + 1.1, f"{v:g}", size=5.8, color="white", bold=True)
        arrow(ax, (11.2, yy + 1.1), (12.8, yy + 1.1))
    label(ax, 22.3, 2.7, "tiny probe + decision + switch + engine;\npreprocessed tensor → synchronized output",
          size=5.3, color=MUTED)
    ax.plot([32.4, 34.0], [12.8, 12.8], color=INK, lw=0.7, ls=(0, (2, 1.2)))
    label(ax, 34.3, 12.8, "budget $B_h$", size=6, ha="left")
    # static cost
    label(ax, 43.5, 9.0, "static cost\n$S_h(\\ell)$: engine only,\nno probe", size=5.6, color=MUTED)

    # --- policy
    px, py = 56.0, 18.0
    ax.add_patch(FancyBboxPatch((px, py), 12.5, 14.0, boxstyle="round,pad=0.1,rounding_size=0.8", fc="#F0FDF4",
                                ec="#16A34A", lw=1.0, zorder=4))
    label(ax, px + 6.25, py + 12.6, "(4) Policy D", bold=True, size=7)
    label(ax, px + 0.8, py + 7.0, "1. keep $C_h(\\ell)\\leq B_h$\n2. cheapest with $\\hat r_\\ell\\leq\\tau$\n3. else lowest $\\hat r_\\ell$",
          size=5.8, ha="left", linespacing=1.5)
    label(ax, px + 6.25, py + 1.6, "T-hard: 3 thresholds\non $s(x)$ instead", size=5.2, color=MUTED)
    arrow(ax, (rx + 10.2, 27.0), (px - 0.2, 26.5))
    arrow(ax, (52.6, 9.0), (px + 4.0, py - 0.2), rad=-0.25)

    # --- outcomes
    # tiny selected: reuse
    arrow(ax, (px + 12.8, 30.0), (73.5, 40.0), RED, lw=1.3)
    tilted_image(ax, load_rgb("tiny"), 74.0, 38.2, 8, 4.0)
    label(ax, 89.5, 41.0, "if Tiny: reuse probe\noutput, no further\ninference", size=5.8, ha="left")
    # S/M/L
    arrow(ax, (px + 12.8, 22.0), (72.0, 22.0), RED, lw=1.3)
    end2 = layer_stack(ax, 72.2, 22.0, [(8, 0.7), (6.5, 0.85), (5, 1.0), (6.5, 0.85), (8, 0.7)], CANDIDATE_COLORS["large"])
    label(ax, 77.6, 31.4, "(5) one static engine\n(Small / Medium / Large)", bold=True, size=6.2)
    arrow(ax, (end2[0] + 0.3, end2[1]), (86.5, 22.0), RED, lw=1.3)
    tilted_image(ax, load_rgb("large"), 86.8, 19.5, 9, 4.5)
    label(ax, 91.3, 16.8, "routed output", size=6)

    legend_box(ax, 56.0, 3.0, 42.5, 7.0, [
        ("block", [CANDIDATE_COLORS[lv] for lv in LEVELS], "compiled static engine (T/S/M/L)"),
        ("plane", "#9CA3AF", "image / segmentation / entropy map"),
        ("arrow", (RED, "-"), "data path at inference"),
        ("arrow", (INK, "-"), "score / decision flow"),
    ])
    save_all(fig, GEN / "fig2_pipeline_3d_v20")


# ======================================================================== Fig. 3
def fig_architecture():
    apply_style()
    W = mm_to_inches(FULL_WIDTH_MM)
    fig = plt.figure(figsize=(W, W * 0.56))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.set_aspect("equal"); ax.axis("off")
    rows = {r["candidate"]: r for r in csv.DictReader(open(ROOT / "paper/tables/candidate_family_landscape.csv", encoding="utf-8"))}

    label(ax, 1.0, 54.5, "(a) One shared elastic encoder–decoder", bold=True, size=7.5, ha="left")
    tilted_image(ax, load_rgb("rgb"), 1.0, 38.0, 9, 4.5)
    label(ax, 5.5, 35.8, "input, resolution $R_\\ell$", size=6)
    enc = "#8FB8DE"; dec = "#A8D5A2"
    # (name, height, width(channels), color)
    layers = [("Stem", 10, 0.7, enc), ("Stage 1", 8, 1.0, enc), ("Stage 2", 6, 1.3, enc), ("Stage 3", 4.5, 1.7, enc),
              ("Decode 2", 6, 1.3, dec), ("Decode 1", 8, 1.0, dec), ("Decode 0", 10, 0.7, dec), ("Head", 10, 0.5, "#F2C14E")]
    x = 13.0
    yc = 41.5
    centers = []
    arrow(ax, (10.6, yc), (12.8, yc))
    for i, (name, h, w, col) in enumerate(layers):
        d = w * 2.2
        # active (coloured) and inactive channel slices for elasticity
        block(ax, x, yc - h / 2, w, h, d, col, z=3 + i)
        if 0 < i < 7:
            block(ax, x + w * 0.0, yc - h / 2, w, h, d * 0.35, shade(col, 0.55), z=4 + i, alpha=0.0)
        centers.append((x + w / 2, yc + h / 2 + d * DY))
        label(ax, x + w / 2 + d * DX / 2, yc - h / 2 - 1.3 - (1.6 if i % 2 else 0), name, size=5.6)
        nx = x + w + d * DX + 2.0
        if i < len(layers) - 1:
            arrow(ax, (x + w + d * DX + 0.1, yc), (nx - 0.1, yc))
        x = nx
    # skips
    for a, b in ((1, 5), (2, 4), (0, 6)):
        xa, ya = centers[a]; xb, yb = centers[b]
        top = 53.0 - 0.0 * a
        ax.plot([xa, xa, xb, xb], [ya + 0.3, top - a * 1.4, top - a * 1.4, yb + 0.3], color=MUTED, lw=0.7,
                ls=(0, (2.5, 1.5)), zorder=2)
        arrow(ax, (xb, yb + 1.0), (xb, yb + 0.2), MUTED, lw=0.7)
    label(ax, (centers[0][0] + centers[6][0]) / 2 + 22, 52.4, "additive encoder skips", size=5.8, color=MUTED)
    seg = load_rgb("large")
    arrow(ax, (x - 2.0 + 0.1, yc), (x - 0.4, yc))
    tilted_image(ax, seg, x - 0.2, 39.0, 9, 4.5)
    label(ax, x + 4.3, 36.8, "19-class map", size=6)
    label(ax, 50.0, 33.0, "elastic axes:  input resolution $R_\\ell$  ·  active channel prefix $w_\\ell$ (block depth)  ·  "
          "active block prefix $d_\\ell$", size=6, color=MUTED)

    # (b) candidates
    label(ax, 1.0, 30.0, "(b) Four static candidates extracted from the same weights", bold=True, size=7.5, ha="left")
    for k, lv in enumerate(LEVELS):
        r = rows[lv]
        col = CANDIDATE_COLORS[lv]
        cx0 = 1.5 + k * 19.0
        ax.add_patch(FancyBboxPatch((cx0, 8.0), 17.5, 19.0, boxstyle="round,pad=0.1,rounding_size=0.8", fc="white",
                                    ec=col, lw=1.2, zorder=1))
        ax.add_patch(Rectangle((cx0, 25.2), 17.5, 1.8, fc=col, ec="none", zorder=2))
        label(ax, cx0 + 0.6, 26.1, lv.title(), bold=True, color="white", ha="left", size=7)
        label(ax, cx0 + 16.9, 26.1, r["resolution_hxw"].replace("×", "×"), color="white", ha="right", size=6)
        wm = float(r["width_multiplier"]); dep = int(r["depth_blocks"])
        # input size proportional to resolution
        res_w = {"tiny": 3.4, "small": 4.0, "medium": 4.8, "large": 5.6}[lv]
        tilted_image(ax, load_rgb("rgb", (128, 64)), cx0 + 0.6, 17.5, res_w, res_w / 2, skew=18, z=4)
        bx = cx0 + res_w + 1.4
        for j in range(dep):
            block(ax, bx + j * 1.45, 16.5, 0.55, 6.0, wm * 2.0, col, z=5 + j, lw=0.4)
        label(ax, cx0 + 8.75, 13.6, f"$w$ = {wm:.2f}    $d$ = {dep}", size=6)
        label(ax, cx0 + 8.75, 11.7, f"{int(r['trainable_params']) / 1e6:.3f} M params · {float(r['gflops']):.3f} GFLOPs", size=5.6)
        label(ax, cx0 + 8.75, 9.4, f"candidate-only p95 (ms)\nAGX {float(r['e3_candidate_p95_ms']):.2f} · Hailo-8 {float(r['e1_candidate_p95_ms']):.2f}",
              size=5.4, color=MUTED, linespacing=1.2)
    # export
    ax.add_patch(FancyBboxPatch((1.5, 1.0), 74.5, 5.0, boxstyle="round,pad=0.1,rounding_size=0.6", fc="#F8FAFC",
                                ec="#CBD5E1", lw=0.6, zorder=1))
    label(ax, 3.0, 3.5, "each candidate →", size=6.5, ha="left")
    flat_box(ax, 18.0, 2.0, 11.0, 3.0, "fixed-shape ONNX", "#6B7280", size=6)
    arrow(ax, (29.4, 3.5), (33.6, 3.5))
    flat_box(ax, 34.0, 2.0, 18.0, 3.0, "TensorRT FP16 (AGX / NX / Orin)", "#4C72B0", size=6)
    flat_box(ax, 54.0, 2.0, 14.0, 3.0, "Hailo-8 INT8 HEF", "#C44E52", size=6)
    arrow(ax, (31.5, 3.5), (53.8, 3.5), rad=-0.35)

    legend_box(ax, 78.0, 10.0, 21.0, 17.0, [
        ("block", [enc], "encoder stage"),
        ("block", [dec], "decoder stage"),
        ("block", ["#F2C14E"], "19-class head"),
        ("plane", "#9CA3AF", "image / output map"),
        ("arrow", (MUTED, (0, (2.5, 1.5))), "skip connection"),
        ("arrow", (INK, "-"), "feature flow"),
    ])
    save_all(fig, GEN / "fig3_architecture_3d_v20")


if __name__ == "__main__":
    fig_pipeline()
    fig_architecture()
