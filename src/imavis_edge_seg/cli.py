"""Thin CLI. Long/server-side jobs live in `scripts/`; this only wraps fast, local
commands (config inspection, environment report) plus dispatch to those scripts."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import typer
from rich.console import Console

from imavis_edge_seg import __version__
from imavis_edge_seg.config import ExperimentConfig, load_config

app = typer.Typer(add_completion=False, help="PACE-Seg edge segmentation CLI.")
data_app = typer.Typer(add_completion=False, help="Dataset manifest commands.")
app.add_typer(data_app, name="data")
console = Console()


@app.command()
def env_report() -> None:
    """Print Python/OS/package versions relevant to reproducibility disclosure
    (see docs/RESEARCH_PLAN.md §9 — every measurement must disclose its toolchain)."""
    console.print(f"imavis-edge-seg version: {__version__}")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"Platform: {platform.platform()}")
    for pkg in ("torch", "onnx", "onnxruntime"):
        try:
            mod = __import__(pkg)
            console.print(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            console.print(f"{pkg}: not installed")


@app.command()
def config_show(
    config: Path = typer.Option(
        Path("configs/experiment/default.yaml"), "--config", help="Path to experiment YAML."
    ),
    override: list[str] = typer.Option([], "--override", help="dotlist override, key=value"),
    show_hash: bool = typer.Option(False, "--hash", help="Print config hash instead of full dump."),
) -> None:
    """Load, validate and print an experiment config."""
    cfg: ExperimentConfig = load_config(config, overrides=override or None)
    if show_hash:
        console.print(cfg.config_hash())
        return
    console.print_json(cfg.model_dump_json(indent=2))


@app.command()
def config_init(
    output: Path = typer.Option(..., "-o", "--output", help="Where to write the new config."),
    experiment_id: str = typer.Option(..., "--experiment-id"),
) -> None:
    """Write a default experiment config to disk."""
    import yaml

    cfg = ExperimentConfig(experiment_id=experiment_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    console.print(f"Wrote {output}")


@data_app.command("manifest")
def data_manifest(
    dataset: str = typer.Option(..., "--dataset", help="cityscapes or acdc"),
    data_root: Path = typer.Option(..., "--data-root", help="Dataset root directory"),
    split: str = typer.Option(..., "--split", help="train/val/test (cityscapes) or train/val (acdc)"),
    output: Path = typer.Option(..., "-o", "--output", help="Manifest CSV output path"),
    conditions: list[str] = typer.Option(
        [], "--condition", help="ACDC only: repeat for fog/night/rain/snow (default: all)"
    ),
    no_checksums: bool = typer.Option(
        False, "--no-checksums", help="Skip sha256 (faster, less provenance)"
    ),
) -> None:
    """Scan a dataset root and write a manifest CSV -- see docs/DATASET.md. Never run
    against a partial/still-downloading tree; the manifest is meant to freeze exactly
    what every training job will read."""
    from imavis_edge_seg.data.manifest import (
        build_manifest,
        discover_acdc_samples,
        discover_cityscapes_samples,
        write_manifest_csv,
    )

    if dataset == "cityscapes":
        samples = discover_cityscapes_samples(data_root, split)  # type: ignore[arg-type]
    elif dataset == "acdc":
        cond_tuple = tuple(conditions) if conditions else ("fog", "night", "rain", "snow")
        samples = discover_acdc_samples(data_root, split, cond_tuple)  # type: ignore[arg-type]
    else:
        raise typer.BadParameter("dataset must be 'cityscapes' or 'acdc'")

    if not samples:
        console.print(f"[red]No samples found under {data_root} for split={split!r}[/red]")
        raise typer.Exit(code=1)

    rows = build_manifest(data_root, samples, compute_checksums=not no_checksums)
    write_manifest_csv(rows, output)
    console.print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    app()
