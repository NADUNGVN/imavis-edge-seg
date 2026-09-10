"""`BenchmarkRecord`: one (device, backend, level, precision) measurement, with the
full environment disclosure `docs/RESEARCH_PLAN.md` §9 rule 5 requires (toolchain
versions, power mode, etc.) attached, so a number is never usable without its
provenance. JSON on disk, one file per record -- `aggregate.py` combines many of these
into the latency/energy lookup table `docs/RESEARCH_PLAN.md` §5.3 B describes.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from imavis_edge_seg.benchmark.stats import LatencyStats
from imavis_edge_seg.config import Backend, ElasticityLevel


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


@dataclass
class BenchmarkRecord:
    device_id: str  # E1..E5, matches ../../docs/SHARED_INFRASTRUCTURE.md §3.1
    backend: Backend
    level: ElasticityLevel
    precision: str  # fp16 / int8 / int8_uncalibrated -- match config.py's QuantizationConfig
    resolution: tuple[int, int]
    run_index: int  # which of the >=3 independent runs this is (§9 rule 3)
    timestamp_utc: str
    toolchain: dict[str, str]  # e.g. {"trt": "8.5.2.2", "jetpack": "5.1.4", ...}
    end_to_end: LatencyStats | None = None
    kernel_only: LatencyStats | None = None
    throughput_fps: float | None = None
    power_mw: dict[str, float] | None = None  # min/average/max, telemetry-derived
    temp_c: dict[str, float] | None = None
    notes: str = ""
    git_commit: str = field(default_factory=_git_commit)

    def to_json(self) -> str:
        payload = asdict(self)
        return json.dumps(payload, indent=2)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def read_benchmark_record(path: str | Path) -> BenchmarkRecord:
    payload = json.loads(Path(path).read_text())
    for key in ("end_to_end", "kernel_only"):
        if payload.get(key) is not None:
            payload[key] = LatencyStats(**payload[key])
    payload["resolution"] = tuple(payload["resolution"])
    return BenchmarkRecord(**payload)
