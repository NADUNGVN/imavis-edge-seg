"""Export a validated Archify HTML artifact as a light-theme SVG.

The script uses Archify's own vector exporter through a local Chromium session;
it does not rasterize or screenshot the diagram.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", type=Path, help="Validated Archify HTML artifact")
    parser.add_argument("output", type=Path, help="Destination .svg file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    html_path = args.html.resolve()
    output_path = args.output.resolve()
    if not html_path.is_file():
        raise FileNotFoundError(html_path)
    if output_path.suffix.lower() != ".svg":
        raise ValueError("The output path must end in .svg")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 2048, "height": 1320})
        page.goto(html_path.as_uri(), wait_until="load")
        page.evaluate("document.fonts && document.fonts.ready")
        page.wait_for_function(
            "window.Archify && Archify.exportMenu && typeof Archify.exportMenu.run === 'function'"
        )
        with page.expect_download(timeout=30_000) as download_info:
            page.evaluate("Archify.exportMenu.run('svg-light')")
        download_info.value.save_as(output_path)
        browser.close()

    svg = output_path.read_text(encoding="utf-8")
    if "<svg" not in svg or "<image" in svg:
        raise RuntimeError("Archify export is not a self-contained vector SVG")
    print(f"Wrote {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
