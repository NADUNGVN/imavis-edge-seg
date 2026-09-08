"""Export every elasticity level of a fresh (untrained) PaceSegSupernet to ONNX, at
each level's configured resolution -- the input to scripts/run_smoke_matrix.sh.

    uv run python scripts/export_all_levels.py --output-dir /tmp

Untrained weights only: this script is for compiler smoke tests (does the graph
compile?), not accuracy. Once training exists, export from a checkpoint instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imavis_edge_seg.config import ExperimentConfig
from imavis_edge_seg.export import export_subnet_onnx
from imavis_edge_seg.models import PaceSegSupernet, extract_subnet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/onnx_smoke"))
    parser.add_argument("--experiment-id", default="smoke")
    args = parser.parse_args()

    config = ExperimentConfig(experiment_id=args.experiment_id)
    supernet = PaceSegSupernet(config.supernet)
    supernet.eval()

    for level in config.supernet.levels:
        subnet = extract_subnet(supernet, level)
        height, width = config.supernet.input_resolutions[level]
        out_path = args.output_dir / f"pace_seg_{level}.onnx"
        export_subnet_onnx(subnet, out_path, height, width)
        n_params = sum(p.numel() for p in subnet.parameters())
        print(f"{level}: {out_path} ({width}x{height}, {n_params:,} params)")


if __name__ == "__main__":
    main()
