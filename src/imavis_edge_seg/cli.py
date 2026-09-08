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


if __name__ == "__main__":
    app()
