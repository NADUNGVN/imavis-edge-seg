"""Run deployment-matched router evaluation, E1/E3 replay, and summarization.

Example (on a training server with datasets and checkpoints available):

    uv run python scripts/run_router_deployment_matched_pipeline.py \
      --run A=outputs/pace_seg_v1_aug_seed0/checkpoints/step_00100000.pt \
      --run B=outputs/pace_seg_v1_seed2/checkpoints/step_00100000.pt \
      --run C=outputs/pace_seg_v1_aug_seed3/checkpoints/step_00100000.pt \
      --output-dir reports/router_deployment_matched

The pipeline refuses to overwrite outputs.  Use a new output directory for every
scientific rerun so historical evidence remains immutable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], repo_root: Path, commands: list[list[str]]) -> None:
    commands.append(command)
    subprocess.run(command, cwd=repo_root, check=True)


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=CHECKPOINT")
    label, checkpoint = value.split("=", 1)
    label = label.strip().upper()
    if not re.fullmatch(r"[A-Z]", label):
        raise argparse.ArgumentTypeError("run label must be one letter, for example A")
    return label, Path(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=_parse_run, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment/default.yaml"))
    parser.add_argument("--lookup-table", type=Path, default=Path("outputs/benchmark_lookup_table.csv"))
    parser.add_argument("--e1-overhead", type=Path, default=Path("reports/router_overhead_E1_20260928.json"))
    parser.add_argument("--e3-overhead", type=Path, default=Path("reports/router_overhead_E3_20260922.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_map = dict(args.run)
    if set(run_map) != {"A", "B", "C"}:
        raise ValueError("exactly Runs A, B, and C are required")
    for label, checkpoint in run_map.items():
        if not checkpoint.is_absolute():
            checkpoint = repo_root / checkpoint
            run_map[label] = checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Run {label} checkpoint not found: {checkpoint}")

    commands: list[list[str]] = []
    replay_outputs: list[Path] = []
    for label in ("A", "B", "C"):
        slug = f"run_{label.lower()}"
        checkpoint = run_map[label]
        run_name = f"Run {label}"
        meta = output_dir / f"{slug}_evaluation.json"
        dump = output_dir / f"{slug}_per_image.json"
        _run(
            [
                sys.executable,
                "scripts/evaluate_router.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.config),
                "--lookup-table",
                str(args.lookup_table),
                "--device-id",
                "E3",
                "--backend",
                "tensorrt_gpu",
                "--device",
                args.device,
                "--run-label",
                run_name,
                "--output-json",
                str(meta),
                "--dump-per-image",
                str(dump),
            ],
            repo_root,
            commands,
        )

        replay_specs = (
            ("E1", "hailo_hef", "numpy", args.e1_overhead),
            ("E3", "tensorrt_gpu", "gpu", args.e3_overhead),
        )
        for device_id, backend, entropy_backend, overhead in replay_specs:
            replay = output_dir / f"{slug}_{device_id.lower()}_replay.json"
            _run(
                [
                    sys.executable,
                    "scripts/replay_router_with_overhead.py",
                    "--per-image-dump",
                    str(dump),
                    "--overhead-json",
                    str(overhead),
                    "--original-results-json",
                    str(meta),
                    "--lookup-table",
                    str(args.lookup_table),
                    "--device-id",
                    device_id,
                    "--backend",
                    backend,
                    "--entropy-backend",
                    entropy_backend,
                    "--run-label",
                    run_name,
                    "--output-json",
                    str(replay),
                ],
                repo_root,
                commands,
            )
            replay_outputs.append(replay)

    summary = output_dir / "summary.json"
    summary_command = [sys.executable, "scripts/summarize_router_replays.py"]
    for replay in replay_outputs:
        summary_command.extend(["--replay", str(replay)])
    summary_command.extend(["--output-json", str(summary)])
    _run(summary_command, repo_root, commands)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "scientific_change": "deployment-matched all-pixel entropy at fitting and inference",
        "code_commit": commit,
        "runs": {
            f"Run {label}": {
                "checkpoint": str(path),
                "checkpoint_sha256": _sha256(path),
            }
            for label, path in run_map.items()
        },
        "commands": commands,
        "outputs": {
            str(path.relative_to(output_dir)): _sha256(path)
            for path in sorted(output_dir.glob("*.json"))
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
