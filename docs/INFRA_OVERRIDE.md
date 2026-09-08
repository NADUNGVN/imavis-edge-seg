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
| E1 — Pi5 + Hailo-8 | **Reachable + Hailo software READY** (2026-09-08: `/dev/hailo0` present, `hailortcli` works, identify confirms **Hailo-8**, not 8L; HailoRT 4.23.0). SSH key access installed. | No compiled model/HEF run yet — still need a working Hailo DFC host before any Hailo compile/latency/energy claim |
| **E2 — Xavier NX (reachable, ML stack already installed, 2026-09-08)** | Reachable over direct LAN (`192.168.10.93`). SSH key access installed. L4T R35.4.1. CUDA 11.4.19 + cuDNN 8.6.0 + TensorRT 8.5.2.2 **already present** (not installed by this session). | Compiler smoke test run (see below) — TensorRT GPU clean, DLA has a confirmed hardware subgraph limit affecting the decoder |
| **E3 — AGX Xavier (reachable, ML stack installed 2026-09-08)** | Reachable over direct LAN (`192.168.10.91`, no Tailscale on this device). SSH key access installed. L4T R35.6.4. **CUDA 11.4.19 + cuDNN 8.6.0 + TensorRT 8.5.2.2 installed and verified** (`nvcc`, `trtexec` both run). RAM ~14GiB (likely a 16GB SKU, not the 32GB dev kit assumed in the original plan). Disk: 9.4G free after install. | Compiler smoke test run (see below) — same DLA finding as E2 |
| **E4 — RUBIK Pi 3 (new, 2026-09-08)** | Reachable, SSH key access installed. Qualcomm QCM6490, Hexagon DSP/NPU via QAIRT (formerly SNPE) — **not TensorRT/DLA/Hailo**. | **Provisional recommendation (2026-09-08, not yet confirmed by researcher): keep as opportunistic/secondary only** — not a required backend, not part of go/no-go criteria. `qairt-tools` binary not found on default `$PATH` for the `ubuntu` user. |
| Jetson Nano | **Not yet in shared-infra inventory table at all** | Needs its own inventory row added to `SHARED_INFRASTRUCTURE.md` §3.1 before use; JetPack 4 is EOL (2024-11) — expect the most toolchain friction here |

### Compiler smoke test result (E2 + E3, 2026-09-08)

TensorRT GPU (FP16 and uncalibrated INT8): 8/8 PASS on both devices, all 4 elasticity
levels, no unsupported ops. Xavier DLA: builds on both devices at every level, but the
**decoder falls back to GPU wholesale** — confirmed root cause directly from TensorRT's
log: *"DLA supports only 16 subgraphs per DLA core"*. The encoder fits the budget and
runs on DLA; the decoder's resize→add→conv pattern creates too many extra subgraph
partitions and exhausts it. Not an unsupported-operator problem — a subgraph-count
budget problem, plausibly fixable by restructuring the decoder. Full detail, per-level
throughput, and a note correcting an earlier wrong "0 fallback" claim are in
`reports/edge/E3_compiler_smoke_test_20260908.md` and
`reports/edge/E2_NX_compiler_smoke_test_20260908.md`.
| External power meter | Not recorded anywhere in shared infra | Confirm availability before claiming any J/frame number; without it, energy numbers must be clearly labeled as telemetry-derived (`tegrastats`), not system-level |
| Camera / video domain | Not recorded | Default to public datasets (Cityscapes/ACDC/Dark Zurich) unless a project-specific capture setup is confirmed |

## Open decision: does E4 (RUBIK Pi 3 / Qualcomm) belong in the device lineup?

`RESEARCH_PLAN.md` was written around three backend classes: TensorRT GPU, Xavier DLA,
Hailo dataflow NPU (Jetson Nano/NX/AGX + Pi5). E4 is a **fourth, unrelated silicon
family** (Qualcomm Hexagon DSP/NPU, Qualcomm AI Runtime toolchain) that showed up when
access was granted. Discussed with the researcher 2026-09-08: the working recommendation
is to **not** add it as a required backend — it would strengthen the "compiler
portability" framing (G2) but the toolchain risk (QAIRT is unfamiliar and unlike
TensorRT/Hailo DFC) outweighs that benefit given the deadline, especially while E3 (a
real Jetson) is now confirmed reachable. This is a working call, not a final one — revisit
if E2/Nano never materialize and a second confirmed backend besides Hailo is needed to
de-risk the paper's "multi-accelerator" claim.

## Access status (2026-09-08)

Resolved for E1, E2, E3, E4: all four reachable, dedicated SSH key
(`~/.ssh/id_ed25519_imavis_edge_seg`) installed, aliased as `pi5`, `nx` (E2), `agx` (E3),
`rubik` (E4) in `~/.ssh/config` (E1/E4 over Tailscale; E2/E3 over direct LAN, no
Tailscale on those two devices). Still open: `SERVER-01..05` (no direct SSH — see
`COLLABORATION_PROTOCOL.md`) and Jetson Nano (not yet provided). E2 and E3 both have
working TensorRT toolchains and a completed compiler smoke test.

## Mapping row for `../../docs/SHARED_INFRASTRUCTURE.md` §4

Added:

```text
| `IMAVIS_EDGE_SEG/` | NADUNGVN/imavis-edge-seg | TBD under NFS | TBD under NFS | SERVER-01..05 | E1 (Hailo SW ready), E2 (Xavier NX, ML stack ready), E3 (AGX Xavier, ML stack ready), E4 (RUBIK Pi 3, secondary only), Jetson Nano (not inventoried) | Vision segmentation track; independent of CARE-ASD audio pipeline |
```
