"""Build the candidate-only latency lookup table from raw protocol captures
(landscape regime, 2026-10-04).

Expected raw layout (one directory per device and level, as written by
scripts/benchmark/run_trtexec_protocol.sh / run_hailo_protocol.sh):
  <raw-root>/<DEVICE>/<level>/            DEVICE in E1 (Hailo) or E2/E3/E5 (TensorRT GPU FP16)

  python scripts/build_landscape_lut.py --raw-root outputs/benchmark_raw_landscape \
      --records-dir outputs/benchmark_records_landscape --output outputs/benchmark_lookup_table_landscape.csv
Resolutions are stored as (width, height) strings, e.g. 384x192 = 384 wide, 192 high.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from imavis_edge_seg.benchmark import build_record_from_hailo_dir, build_record_from_trtexec_dir
from imavis_edge_seg.benchmark.lookup_table import build_lookup_table, write_lookup_table_csv
from imavis_edge_seg.config import ExperimentConfig

LEVELS = ("tiny", "small", "medium", "large")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--records-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    res_hw = ExperimentConfig(experiment_id="lut").supernet.input_resolutions
    args.records_dir.mkdir(parents=True, exist_ok=True)
    for dev_dir in sorted(p for p in args.raw_root.iterdir() if p.is_dir()):
        dev = dev_dir.name
        for level in LEVELS:
            d = dev_dir / level
            if not d.is_dir():
                continue
            h, w = res_hw[level]
            if dev == "E1":
                rec = build_record_from_hailo_dir(d, dev, "hailo_hef", level, "int8", (w, h))
                name = f"{dev.lower()}_hailo_{level}_int8.json"
            else:
                rec = build_record_from_trtexec_dir(d, dev, "tensorrt_gpu", level, "fp16", (w, h))
                name = f"{dev.lower()}_tensorrt_{level}_fp16.json"
            path = args.records_dir / name
            rec.write(path)
            print(f"record {path}")
    table = build_lookup_table(args.records_dir)
    write_lookup_table_csv(table, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
