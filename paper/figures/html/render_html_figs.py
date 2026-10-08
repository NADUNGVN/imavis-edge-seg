"""Render HTML/CSS schematic figures to vector PDF + PNG with the local Chrome (Playwright).

  .venv/Scripts/python.exe paper/figures/html/render_html_figs.py

Assets (input image, tiny/large predictions, entropy map) are exported from the audited
reports/qualitative_assets_v20 so the figures show real model outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "scripts"))
from render_palette import colorize  # noqa: E402

GEN = HERE.parent / "generated"
ASSETS = ROOT / "reports/qualitative_assets_v20"
AID = "acdc-rain_GP020402_frame_000863_rgb_anon"
FIGS = {"fig2_pipeline_html_v20": "fig2_pipeline.html"}
WIDTH_MM = 178


def export_assets() -> None:
    out = HERE / "img"
    out.mkdir(exist_ok=True)
    Image.open(ASSETS / f"{AID}_rgb.jpg").resize((512, 256)).save(out / "rgb.jpg", quality=92)
    for lv in ("tiny", "small", "medium", "large"):
        m = np.asarray(Image.open(ASSETS / f"{AID}_{lv}.png").resize((512, 256), Image.Resampling.NEAREST))
        Image.fromarray(colorize(m)).save(out / f"{lv}.png")
    ent = np.load(ASSETS / f"{AID}_entropy.npy").astype(np.float32)
    rgb = (plt.get_cmap("magma")(np.clip(ent / np.log(19), 0, 1))[..., :3] * 255).astype(np.uint8)
    Image.fromarray(rgb).resize((512, 256), Image.Resampling.BILINEAR).save(out / "entropy.png")


def render() -> None:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome")
        for stem, html in FIGS.items():
            pg = b.new_page(viewport={"width": 1400, "height": 900}, device_scale_factor=3)
            pg.goto((HERE / html).resolve().as_uri())
            pg.wait_for_load_state("networkidle")
            box = pg.locator("#fig").bounding_box()
            pg.locator("#fig").screenshot(path=str(GEN / f"{stem}.png"))
            # vector PDF at final print width: scale the CSS-pixel figure to WIDTH_MM
            w_in = WIDTH_MM / 25.4
            h_in = w_in * box["height"] / box["width"]
            pg.pdf(path=str(GEN / f"{stem}.pdf"), width=f"{w_in}in", height=f"{h_in + 0.01}in",
                   scale=w_in * 96 / box["width"], print_background=True,
                   margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}, page_ranges="1")
            print("wrote", GEN / f"{stem}.pdf")
        b.close()


if __name__ == "__main__":
    export_assets()
    render()
