"""Render V20 qualitative figure SAMPLES from the audited server assets (local only).

Inputs : reports/qualitative_assets_v20/ (scripts/server/dump_qualitative_v20.sh)
         reports/landscape_20261004/run_a_{per_image,evaluation}.json (policy-D decisions)
Outputs: paper/figures/previews/v20_samples/*.pdf|png

Styles follow docs/figure_conventions_survey_20261008.md: figures drawn at final print
width (190 mm full / 90 mm column), 7 pt text, pure Cityscapes-palette masks with void
in black, fixed colour per capacity, dashed box + zoom inset on the region where the
capacities disagree most.

  A  qualitative grid   : Image | GT | Tiny | Large | Routed (D, badge)        + zoom row-insets
  B  qualitative grid   : Image | GT | Tiny | Large | Routed | Routed error    (no insets)
  C  dataset overview   : 5 conditions x (Image, GT, Routed)                   (ACDC-teaser style)
  D  routing decision   : one rain image, entropy -> per-candidate risk -> E3 vs E1 feasibility -> outputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, "scripts")
from router_review_analyses import RUNS, Split, load_costs, run_policy  # noqa: E402

ASSETS = Path("reports/qualitative_assets_v20")
OUT = Path("paper/figures/previews/v20_samples")
LEVELS = ("tiny", "small", "medium", "large")
CAP_COLOR = {"tiny": "#E69F00", "small": "#56B4E9", "medium": "#009E73", "large": "#0072B2"}  # Okabe-Ito
COND = {"cityscapes": "Clean", "acdc/fog": "Fog", "acdc/night": "Night", "acdc/rain": "Rain", "acdc/snow": "Snow"}
CLASSES = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation",
           "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
PALETTE = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153),
                    (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60),
                    (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)],
                   dtype=np.uint8)
MM = 1 / 25.4
FULL, COL = 190 * MM, 90 * MM
PW, PH = 512, 256  # panel pixels (2:1)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.5,
                     "pdf.fonttype": 42, "savefig.dpi": 600})


# ------------------------------------------------------------------ data
def manifest() -> list[dict]:
    return json.loads((ASSETS / "manifest.json").read_text())["entries"]


def load(aid: str, what: str) -> np.ndarray:
    ext = "jpg" if what == "rgb" else "png"
    im = Image.open(ASSETS / f"{aid}_{what}.{ext}")
    im = im.resize((PW, PH), Image.Resampling.BILINEAR if what == "rgb" else Image.Resampling.NEAREST)
    return np.asarray(im)


def colorize(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), np.uint8)
    ok = mask < len(PALETTE)
    out[ok] = PALETTE[mask[ok]]
    return out


def d_decisions(budget_route: str = "large", backend: str = "E3") -> dict[tuple[str, int], dict]:
    """Policy-D choice of every held-out image of Run A at the budget equal to the
    `budget_route` route cost (median same-harness table), plus its risk vector."""
    run = "a"
    base = Path("reports/landscape_20261004")
    dump = json.loads((base / f"run_{run}_per_image.json").read_text())
    ev = json.loads((base / f"run_{run}_evaluation.json").read_text())
    cost = load_costs("median")[backend]
    budget = cost[budget_route]
    out = {}
    for split in COND:
        sp = Split(dump[split])
        res = run_policy("D", sp, ev[split]["risk_target_grid"], [budget], cost, sp.cal)[budget]
        for i, (s, ch) in enumerate(zip(sp.test_scores, res["choice"])):
            out[(split, i)] = {"choice": ch, "target": res["target"], "budget": budget, "score": float(s),
                               "risk": {lv: float(sp.cal[lv].predict(s)) for lv in LEVELS}}
    return out


def zoom_box(a: np.ndarray, b: np.ndarray, gt: np.ndarray, frac: float = 0.28) -> tuple[int, int, int, int]:
    """Window (x, y, w, h) where predictions a and b disagree most on labelled pixels."""
    diff = ((a != b) & (gt < 255)).astype(np.float32)
    w, h = int(PW * frac), int(PH * frac * 1.0)
    ii = diff.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    s = ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w]
    s[:, int(PW * 0.6) - w:] = -1  # keep the box clear of the bottom-right inset
    y, x = np.unravel_index(np.argmax(s), s.shape)
    return int(x), int(y), w, h


# ------------------------------------------------------------------ drawing helpers
def show(ax, img, title=None, box=None, inset=None, badge=None):
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=7, pad=2)
    if box:
        x, y, w, h = box
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="white", lw=0.8, ls=(0, (2, 1.2))))
    if inset is not None:
        x, y, w, h = box
        ins = ax.inset_axes([0.64, 0.0, 0.36, 0.5])
        ins.imshow(inset[y:y + h, x:x + w], interpolation="nearest")
        ins.set_xticks([]); ins.set_yticks([])
        for s in ins.spines.values():
            s.set_edgecolor("white"); s.set_linewidth(0.8)
    if badge:
        ax.text(0.02, 0.96, badge, transform=ax.transAxes, va="top", ha="left", fontsize=6, color="white",
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc=CAP_COLOR[badge.lower()], ec="none"))


def legend_strip(fig, rect):
    ax = fig.add_axes(rect)
    ax.axis("off")
    n = len(CLASSES)
    per_row = 10
    for k, name in enumerate(CLASSES + ["void"]):
        r, c = divmod(k, per_row)
        col = PALETTE[k] / 255 if k < n else (0, 0, 0)
        x = c / per_row
        y = 1 - (r + 0.5) / 2
        ax.add_patch(Rectangle((x, y - 0.18), 0.012, 0.36, color=col, transform=ax.transAxes, clip_on=False))
        ax.text(x + 0.017, y, name, transform=ax.transAxes, va="center", fontsize=6)


def error_overlay(rgb, pred, gt):
    base = rgb.astype(np.float32) * 0.45
    wrong = (gt < 255) & (pred != gt)
    base[wrong] = 0.2 * base[wrong] + 0.8 * np.array((213, 94, 0))
    return base.clip(0, 255).astype(np.uint8)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", pad_inches=0.01, dpi=300)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf")


# ------------------------------------------------------------------ samples
def pick(kind: str) -> list[dict]:
    m = manifest()
    return [next(e for e in m if e["split"] == s and e["kind"] == kind) for s in COND]


def sample_grid(kind: str, dec, with_error: bool, name: str):
    rows = pick(kind)
    cols = ["Image", "Ground truth", "Tiny", "Large", "Routed (D)"] + (["Routed error"] if with_error else [])
    nc, nr = len(cols), len(rows)
    gap = 0.012
    cell_w = (1 - 0.035 - gap * (nc - 1)) / nc
    fig_h = FULL * (cell_w * nr * 0.5 + 0.11 + gap * (nr - 1))
    fig = plt.figure(figsize=(FULL, fig_h))
    ch = cell_w * 0.5 * FULL / fig_h
    gh = gap * FULL / fig_h
    top = 1 - 0.04 * FULL / fig_h
    for r, e in enumerate(rows):
        aid = e["asset_id"]
        d = dec[(e["split"], e["heldout_position"])]
        rgb, gt = load(aid, "rgb"), load(aid, "gt")
        pr = {lv: load(aid, lv) for lv in LEVELS}
        box = zoom_box(pr["tiny"], pr["large"], gt)
        panels = [(rgb, None), (colorize(gt), None), (colorize(pr["tiny"]), None), (colorize(pr["large"]), None),
                  (colorize(pr[d["choice"]]), d["choice"].capitalize())]
        if with_error:
            panels.append((error_overlay(rgb, pr[d["choice"]], gt), None))
        y0 = top - (r + 1) * ch - r * gh
        for c, (img, badge) in enumerate(panels):
            ax = fig.add_axes([0.035 + c * (cell_w + gap), y0, cell_w, ch])
            use_box = (not with_error) and c in (0, 1, 2, 3, 4)
            show(ax, img, cols[c] if r == 0 else None, box=box if use_box else None,
                 inset=img if (use_box and c > 0) else None, badge=badge)
            if c == 0:
                ax.text(-0.06, 0.5, COND[e["split"]], transform=ax.transAxes, rotation=90, va="center", ha="center")
    legend_strip(fig, [0.035, 0.0, 0.965, 0.045 * FULL / fig_h])
    save(fig, name)


def sample_overview(dec, name: str):
    rows_e = pick("median")
    fig, axes = plt.subplots(3, 5, figsize=(FULL, FULL * 0.33), gridspec_kw=dict(wspace=0.03, hspace=0.05))
    for c, e in enumerate(rows_e):
        aid = e["asset_id"]
        d = dec[(e["split"], e["heldout_position"])]
        show(axes[0, c], load(aid, "rgb"), COND[e["split"]])
        show(axes[1, c], colorize(load(aid, "gt")))
        show(axes[2, c], colorize(load(aid, d["choice"])), badge=d["choice"].capitalize())
    for r, lab in enumerate(["Image", "Ground truth", "Routed (D)"]):
        axes[r, 0].text(-0.06, 0.5, lab, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="center")
    save(fig, name)


def sample_decision(dec, name: str):
    e = next(x for x in manifest() if x["split"] == "acdc/rain" and x["kind"] == "gain")
    aid = e["asset_id"]
    d = dec[(e["split"], e["heldout_position"])]
    costs = load_costs("median")
    ent = np.load(ASSETS / f"{aid}_entropy.npy").astype(np.float32)
    fig = plt.figure(figsize=(FULL, FULL * 0.30))
    W, H = 1.0, 1.0
    pw, ph = 0.215, 0.215 * 2 * 0.5 / 0.30 * 0.5 * 2  # 2:1 panels in figure fractions
    ph = pw * 0.5 / 0.30
    # column 1: input and entropy
    show(fig.add_axes([0.0, 0.52, pw, ph]), load(aid, "rgb"), "(a) Input, ACDC rain")
    ax = fig.add_axes([0.0, 0.02, pw, ph])
    im = ax.imshow(ent, cmap="magma", interpolation="bilinear", vmin=0, vmax=float(np.log(19)))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"(b) Tiny-probe entropy, $s(x)$={d['score']:.3f}", fontsize=7, pad=2)
    cax = fig.add_axes([pw + 0.005, 0.02, 0.007, ph])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("nats (max ln 19)", fontsize=5.5, labelpad=1)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5)
    # column 2: risk vector
    ax = fig.add_axes([0.32, 0.14, 0.15, 0.70])
    xs = np.arange(4)
    ax.bar(xs, [d["risk"][lv] for lv in LEVELS], color=[CAP_COLOR[lv] for lv in LEVELS], width=0.65)
    ax.axhline(d["target"], color="k", lw=0.7, ls="--")
    ax.text(3.45, d["target"], r"$\tau$", va="center", fontsize=7)
    ax.set_xticks(xs, ["T", "S", "M", "L"])
    ax.set_ylabel(r"Predicted error $\hat r_\ell$", labelpad=2)
    ax.tick_params(width=0.5, length=2, labelsize=6)
    ax.set_title("(c) Calibrated risk", fontsize=7, pad=4)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    # column 3: feasibility per backend
    ax = fig.add_axes([0.50, 0.08, 0.25, 0.76]); ax.axis("off")
    ax.set_title("(d) Measured route cost (ms)", fontsize=7, pad=4)
    rows = []
    for k, be in enumerate(("E3", "E1")):
        b = costs[be]["large"] if be == "E3" else costs[be]["medium"]
        ch = d_choice_for(d, costs[be], b)
        rows.append((be, ch))
        y = 0.62 - k * 0.52
        ax.text(0.0, y + 0.27, f"{be} ({'TensorRT' if be == 'E3' else 'Hailo-8'}), budget {b:.1f}", transform=ax.transAxes, fontsize=6.5)
        for j, lv in enumerate(LEVELS):
            feas = costs[be][lv] <= b + 1e-9
            ax.add_patch(Rectangle((j * 0.25, y), 0.23, 0.22, transform=ax.transAxes,
                                   fc=CAP_COLOR[lv] if feas else "#DDDDDD", ec="k" if lv == ch else "none",
                                   lw=1.3, hatch=None if feas else "////"))
            ax.text(j * 0.25 + 0.115, y + 0.11, f"{costs[be][lv]:.1f}", transform=ax.transAxes, ha="center",
                    va="center", fontsize=6, color="white" if feas else "#666666")
    # column 4: outputs
    for k, (be, ch) in enumerate(rows):
        show(fig.add_axes([0.785, 0.52 - k * 0.50, pw, ph]), colorize(load(aid, ch)),
             f"({'ef'[k]}) {be} runs {ch}", badge=ch.capitalize())
    save(fig, name)


def d_choice_for(d, cost, budget):
    from router_review_analyses import d_choice, ordered_levels
    return d_choice(d["risk"], d["target"], budget, cost, ordered_levels(cost))


def sample_plain(name: str, budget_route: str = "medium"):
    """Conventional grid (SegFormer/DDRNet/ACDC style): plain panels, column headers only,
    no boxes/insets/badges; the routed capacity is written under the routed panel."""
    dec = d_decisions(budget_route)
    rows = pick("median")
    cols = ["Image", "Ground truth", "Tiny", "Medium", "Large", "Routed (D)"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(FULL, FULL * 0.47),
                             gridspec_kw=dict(wspace=0.02, hspace=0.04, bottom=0.08))
    for r, e in enumerate(rows):
        aid = e["asset_id"]
        ch = dec[(e["split"], e["heldout_position"])]["choice"]
        imgs = [load(aid, "rgb"), colorize(load(aid, "gt")), colorize(load(aid, "tiny")),
                colorize(load(aid, "medium")), colorize(load(aid, "large")), colorize(load(aid, ch))]
        for c, img in enumerate(imgs):
            show(axes[r, c], img, cols[c] if r == 0 else None)
        axes[r, 0].text(-0.06, 0.5, COND[e["split"]], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="center")
        axes[r, 5].text(1.02, 0.5, ch, transform=axes[r, 5].transAxes, rotation=90, va="center", ha="left", fontsize=6)
    legend_strip(fig, [0.125, 0.0, 0.78, 0.05])
    save(fig, name)


def main():
    dec = d_decisions()
    sample_grid("gain", dec, with_error=False, name="A_grid_zoom")
    sample_grid("gain", dec, with_error=True, name="B_grid_error")
    sample_overview(dec, name="C_overview")
    sample_decision(dec, name="D_decision")
    sample_plain("E_grid_plain")


if __name__ == "__main__":
    main()


def sample_teaser(name: str = "F_teaser"):
    """Thesis figure: (a) input-dependent choice on one device, (b) device-dependent choice for one
    image, (c) the routing overhead that erases the adaptive gain. Images are the frozen
    ACDC-rain 'easy' and 'gain' selections of selection_v20.json; decisions are policy D of Run A."""
    m = manifest()
    easy = next(x for x in m if x["split"] == "acdc/rain" and x["kind"] == "easy")
    hard = next(x for x in m if x["split"] == "acdc/rain" and x["kind"] == "gain")
    costs = load_costs("median")
    dE3 = d_decisions("large", "E3")
    de, dh = dE3[(easy["split"], easy["heldout_position"])], dE3[(hard["split"], hard["heldout_position"])]
    b1 = costs["E1"]["medium"]
    ch1 = d_choice_for(dh, costs["E1"], b1)
    fig = plt.figure(figsize=(FULL, FULL * 0.30))
    pw = 0.125
    ph = pw * 0.5 / 0.30

    def panel(x, y, img, title, badge=None):
        show(fig.add_axes([x, y, pw, ph]), img, title, badge=badge)

    fig.text(0.0, 0.98, "(a) Same device (E3), different inputs", fontsize=7, fontweight="bold", va="top")
    panel(0.0, 0.48, load(easy["asset_id"], "rgb"), f"easier: $s(x)$={de['score']:.2f}")
    panel(0.135, 0.48, colorize(load(easy["asset_id"], de["choice"])), "routed", de["choice"].capitalize())
    panel(0.0, 0.05, load(hard["asset_id"], "rgb"), f"harder: $s(x)$={dh['score']:.2f}")
    panel(0.135, 0.05, colorize(load(hard["asset_id"], dh["choice"])), "routed", dh["choice"].capitalize())

    fig.text(0.295, 0.98, "(b) Same image, other hardware", fontsize=7, fontweight="bold", va="top")
    panel(0.295, 0.48, colorize(load(hard["asset_id"], dh["choice"])),
          f"E3, budget {costs['E3']['large']:.1f} ms", dh["choice"].capitalize())
    panel(0.295, 0.05, colorize(load(hard["asset_id"], ch1)), f"E1, budget {b1:.1f} ms", ch1.capitalize())
    fig.text(0.425, 0.12, f"E1 large route\n{costs['E1']['large']:.1f} ms:\ninfeasible", fontsize=5.8,
             color="#B4442C", va="bottom")

    fig.text(0.575, 0.98, "(c) ...but routing costs time (E1, large)", fontsize=7, fontweight="bold", va="top")
    sh = json.loads(Path("reports/router_same_harness_analysis_20261004.json").read_text())["median"]["E1_explicit_float32"]
    S, C = sh["static_ms"]["large"], sh["route_ms"]["large"]
    ax = fig.add_axes([0.63, 0.36, 0.36, 0.42])
    ax.barh([1], [S], color=CAP_COLOR["large"], height=0.5)
    ax.barh([0], [S], color=CAP_COLOR["large"], alpha=0.45, height=0.5)
    ax.barh([0], [C - S], left=[S], color="white", ec="k", hatch="////", lw=0.5, height=0.5)
    ax.set_yticks([1, 0], ["static $S$", "routed $C$"])
    ax.text(S + 1.5, 1, f"{S:.1f} ms", va="center", fontsize=6)
    ax.text(C + 1.5, 0, f"{C:.1f} ms", va="center", fontsize=6)
    ax.text(S + (C - S) / 2, 0.36, f"+{C - S:.1f} ms probe + decision + switch", ha="center", va="bottom",
            fontsize=5.8, color="#B4442C")
    ax.set_xlim(0, C * 1.18)
    ax.set_xlabel("median latency (ms)", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=6, width=0.5, length=2)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    fig.text(0.575, 0.06, "Under mean-cost budgets routing pays off only if this overhead falls below\n"
             "about 4.3 ms on E1 and 1.3 ms on E3 (measured: 44 ms and 1.9 ms);\n"
             "under per-frame budgets static deployment wins at any overhead.",
             fontsize=6, va="bottom", color="#1F2937", linespacing=1.3)
    save(fig, name)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "teaser":
    sample_teaser()


def sample_stratified(name: str = "G_stratified"):
    """Route-class stratified grid (selection_v20_stratified.json; rule in that file)."""
    global ASSETS
    ASSETS = Path("reports/qualitative_assets_v20_stratified")
    rows = manifest()
    cols = ["Image", "Ground truth", "Tiny", "Medium", "Large", "Routed (D)"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(FULL, FULL * 0.105 * len(rows) + 0.25),
                             gridspec_kw=dict(wspace=0.02, hspace=0.06, bottom=0.12))
    for r, e in enumerate(rows):
        aid, ch = e["asset_id"], e["routed_to"]
        imgs = [load(aid, "rgb"), colorize(load(aid, "gt")), colorize(load(aid, "tiny")),
                colorize(load(aid, "medium")), colorize(load(aid, "large")), colorize(load(aid, ch))]
        for c, img in enumerate(imgs):
            show(axes[r, c], img, cols[c] if r == 0 else None)
        axes[r, 0].text(-0.06, 0.5, f"routed: {ch}", transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="center", fontsize=6.5)
        axes[r, 0].text(0.02, 0.04, COND[e["split"]], transform=axes[r, 0].transAxes, fontsize=5.5, color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.6, ec="none"))
    legend_strip(fig, [0.125, 0.0, 0.78, 0.07])
    save(fig, name)
