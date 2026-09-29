"""Convert a standalone SVG to a tightly cropped vector PDF with Chromium."""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", type=Path)
    parser.add_argument("pdf", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    svg_path = args.svg.resolve()
    pdf_path = args.pdf.resolve()
    if not svg_path.is_file():
        raise FileNotFoundError(svg_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page()
        page.goto(svg_path.as_uri(), wait_until="load")
        page.evaluate("document.fonts && document.fonts.ready")
        dimensions = page.evaluate(
            """() => {
                const svg = document.querySelector('svg');
                const box = svg.viewBox && svg.viewBox.baseVal;
                if (box && box.width > 0 && box.height > 0) {
                    return {width: box.width, height: box.height};
                }
                const rect = svg.getBoundingClientRect();
                return {width: rect.width, height: rect.height};
            }"""
        )
        width_inches = float(dimensions["width"]) / 96.0
        height_inches = float(dimensions["height"]) / 96.0
        page.pdf(
            path=str(pdf_path),
            width=f"{width_inches:.6f}in",
            height=f"{height_inches:.6f}in",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            prefer_css_page_size=False,
        )
        browser.close()
    print(f"Wrote {pdf_path} ({pdf_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
