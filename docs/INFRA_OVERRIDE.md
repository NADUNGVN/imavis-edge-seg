# Infra override — IMAVIS-EDGE-SEG

Base document: [`../../docs/SHARED_INFRASTRUCTURE.md`](../../docs/SHARED_INFRASTRUCTURE.md).
This file records only what differs or what this project additionally needs — not a copy
of the inventory.

## Train

Uses `SERVER-01..05` as-is. No dedicated allocation; coordinate GPU usage with CARE-ASD
before large supernet-training jobs (both projects share the same five machines).
Segmentation training needs meaningfully more VRAM/throughput than CARE-ASD's audio
models — prefer `SERVER-01`/`SERVER-02` (48 GB Quadro RTX 8000) for supernet training
with multiple subnets in the sandwich rule; `SERVER-03`/`SERVER-04` (24 GB RTX 3090) for
single-subnet ablations.

## Deploy — status vs what this project needs

| Device | Shared-infra status | What PACE-Seg additionally needs before use |
|---|---|---|
| E1 — Pi5 + Hailo-8 | **Reachable + Hailo software READY** (2026-09-08: `/dev/hailo0` present, `hailortcli` works, identify confirms **Hailo-8**, not 8L; HailoRT 4.23.0). SSH key access installed. | No compiled model/HEF run yet — still need `scripts/compiler_smoke_test.md` before any Hailo latency/energy claim |
| **E4 — RUBIK Pi 3 (new, 2026-09-08)** | Reachable, SSH key access installed. Qualcomm QCM6490, Hexagon DSP/NPU via QAIRT (formerly SNPE) — **not TensorRT/DLA/Hailo**. | **Open decision**, see below — is this a 4th target backend or out of scope? `qairt-tools` binary not found on default `$PATH` for the `ubuntu` user; needs investigation before any compile attempt. |
| E2 — Xavier NX | **Inventory pending** (JetPack 5.1.5 planned) | Full inventory per shared-infra §7 checklist; confirm DLA availability and JetPack/TensorRT versions before compiler-safe search space is finalized |
| E3 — AGX Xavier | **Inventory pending** (JetPack 5.1.5 planned) | Same as E2 |
| Jetson Nano | **Not yet in shared-infra inventory table at all** | Needs its own inventory row added to `SHARED_INFRASTRUCTURE.md` §3.1 before use; JetPack 4 is EOL (2024-11) — expect the most toolchain friction here |
| External power meter | Not recorded anywhere in shared infra | Confirm availability before claiming any J/frame number; without it, energy numbers must be clearly labeled as telemetry-derived (`tegrastats`), not system-level |
| Camera / video domain | Not recorded | Default to public datasets (Cityscapes/ACDC/Dark Zurich) unless a project-specific capture setup is confirmed |

## Open decision: does E4 (RUBIK Pi 3 / Qualcomm) belong in the device lineup?

`RESEARCH_PLAN.md` was written around three backend classes: TensorRT GPU, Xavier DLA,
Hailo dataflow NPU (Jetson Nano/NX/AGX + Pi5). E4 is a **fourth, unrelated silicon
family** (Qualcomm Hexagon DSP/NPU, Qualcomm AI Runtime toolchain) that showed up when
access was granted — it was not part of the original plan and Jetson NX/AGX/Nano are
still unconfirmed. This needs a decision from the researcher before the compiler-safe
search space (`config.py` `Backend` literal, currently `tensorrt_gpu` / `xavier_dla` /
`hailo_hef` / `onnxruntime_cpu`) is extended: is E4 replacing the Jetson boards in the
device lineup, added alongside them, or out of scope for this paper? A fourth backend
family would strengthen the "compiler portability" contribution (G2) but adds real
toolchain risk (QAIRT is different enough from TensorRT/Hailo DFC that Week 1-2's
compiler smoke test would need a fourth column).

## Access status (2026-09-08)

Resolved for E1 and E4: both reachable over Tailscale, dedicated SSH key
(`~/.ssh/id_ed25519_imavis_edge_seg`) installed on both, aliased as `pi5` and `rubik` in
`~/.ssh/config`. Still open: `SERVER-01..05` (no direct SSH — see
`COLLABORATION_PROTOCOL.md`) and Jetson NX/AGX/Nano (not yet provided).

## Mapping row for `../../docs/SHARED_INFRASTRUCTURE.md` §4

Added:

```text
| `IMAVIS_EDGE_SEG/` | NADUNGVN/imavis-edge-seg | TBD under NFS | TBD under NFS | SERVER-01..05 | E1 (Hailo SW ready), E4 (RUBIK Pi 3, role tbc), E2/E3 (inventory pending), Jetson Nano (not inventoried) | Vision segmentation track; independent of CARE-ASD audio pipeline |
```
