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
| E2 — Xavier NX | Reachable over direct LAN (`192.168.10.93`). SSH key access installed. L4T R35.4.1. CUDA 11.4.19 + cuDNN 8.6.0 + TensorRT 8.5.2.2 already present (not installed by this session). | Compiler smoke test complete — see below |
| E3 — AGX Xavier | Reachable over direct LAN (`192.168.10.91`, no Tailscale on this device). SSH key access installed. L4T R35.6.4. CUDA 11.4.19 + cuDNN 8.6.0 + TensorRT 8.5.2.2 installed 2026-09-08 and verified. RAM ~14GiB (likely a 16GB SKU, not the 32GB dev kit assumed in the original plan). | Compiler smoke test complete — see below |
| E4 — RUBIK Pi 3 | Reachable, SSH key access installed. Qualcomm QCM6490, Hexagon DSP/NPU via QAIRT (formerly SNPE) — not TensorRT/DLA/Hailo. | **Provisional call (not yet confirmed by researcher): kept as opportunistic/secondary only**, not a required backend — see "Open decisions" |
| E5 — Jetson Orin Nano Super | Reachable over Tailscale (relayed, ~230ms). SSH key access installed. L4T R36.5.0 / JetPack 6.x, CUDA 12.6 + cuDNN 9.3 + TensorRT 10.3 already present. **No DLA** (confirmed: `Cannot create DLA engine, 0 not available`). **Not the legacy "Jetson Nano" the original plan assumed.** | Compiler smoke test complete — see below. Role vs the legacy Nano is an open decision |
| External power meter | Not recorded anywhere in shared infra | Confirm availability before claiming any J/frame number; without it, energy numbers must be clearly labeled as telemetry-derived (`tegrastats`), not system-level |
| Camera / video domain | Not recorded | Default to public datasets (Cityscapes/ACDC/Dark Zurich) unless a project-specific capture setup is confirmed |

## Compiler smoke test results

**E2 + E3 (Xavier generation), 2026-09-08:** TensorRT GPU (FP16 and uncalibrated INT8):
8/8 PASS on both devices, all 4 elasticity levels, no unsupported ops. Xavier DLA: builds
on both devices at every level, but the **decoder falls back to GPU wholesale** —
confirmed root cause directly from TensorRT's log: *"DLA supports only 16 subgraphs per
DLA core"*. The encoder fits the budget and runs on DLA; the decoder's resize→add→conv
pattern creates too many extra subgraph partitions and exhausts it. Not an
unsupported-operator problem — a subgraph-count budget problem, plausibly fixable by
restructuring the decoder. Full detail, per-level throughput, and a note correcting an
earlier wrong "0 fallback" claim are in `reports/edge/E3_compiler_smoke_test_20260908.md`
and `reports/edge/E2_NX_compiler_smoke_test_20260908.md`.

**E5 (Orin generation), 2026-09-09:** TensorRT GPU FP16/INT8: 8/8 PASS, all 4 levels, no
unsupported ops, no DLA to test (device has none). CUDA/TensorRT are a full major version
ahead of E2/E3 (12.6/TRT10 vs 11.4/TRT8.5) — engines are not expected to be
cross-compatible; each device needs its own compiled engine, which the project's
static-engine-per-backend design already assumes. Full detail:
`reports/edge/E5_orin_nano_compiler_smoke_test_20260909.md`.

Not yet tested anywhere: Hailo DFC (ONNX -> HEF; needs an x86 host with the Dataflow
Compiler installed, not set up), calibrated INT8, a decoder redesign that fits the
16-subgraph DLA budget.

## Open decisions

**Does E4 (RUBIK Pi 3 / Qualcomm) belong in the device lineup?** `RESEARCH_PLAN.md` was
written around three backend classes: TensorRT GPU, Xavier DLA, Hailo dataflow NPU
(Jetson Nano/NX/AGX + Pi5). E4 is a fourth, unrelated silicon family (Qualcomm Hexagon
DSP/NPU, Qualcomm AI Runtime toolchain). Discussed with the researcher 2026-09-08: the
working recommendation is to **not** add it as a required backend — it would strengthen
the "compiler portability" framing (G2) but the toolchain risk (QAIRT is unfamiliar and
unlike TensorRT/Hailo DFC) outweighs that benefit given the deadline, especially now that
E2/E3 (real Jetsons) are confirmed reachable. Not a final call — revisit if a second
confirmed backend besides Hailo is needed to de-risk the "multi-accelerator" claim.

**Does E5 (Orin Nano Super) satisfy the "Jetson Nano" slot in the original hardware
plan, or is the legacy EOL Nano (Maxwell GPU, JetPack 4) still separately wanted?** The
original plan explicitly frames the legacy Nano as a "worst-case/legacy target" to prove
reproducibility on an EOL software stack (`docs/SOURCE_RESEARCH_GAP_2026.md` §3). E5 is
the opposite: a *newer*, actively-supported device (JetPack 6, CUDA 12.6, TensorRT 10.3)
that happens to share the "Nano" name. Using E5 in place of the legacy Nano changes that
part of the paper's narrative (no more EOL-stack reproducibility story on this device) —
needs an explicit researcher decision, not an assumption.

## Access status (2026-09-09)

Resolved for E1, E2, E3, E4, E5: all five reachable, dedicated SSH key
(`~/.ssh/id_ed25519_imavis_edge_seg`) installed, aliased as `pi5`, `nx` (E2), `agx` (E3),
`rubik` (E4), `nano` (E5) in `~/.ssh/config` (E1/E4/E5 over Tailscale — E5 relayed;
E2/E3 over direct LAN, no Tailscale on those two devices). Still open: `SERVER-01..05`
(no direct SSH — see `COLLABORATION_PROTOCOL.md`) and the legacy Jetson Nano (not
provided, and may not be needed — see "Open decisions"). E2, E3 and E5 all have working
TensorRT toolchains and a completed compiler smoke test.

## Mapping row for `../../docs/SHARED_INFRASTRUCTURE.md` §4

Added:

```text
| `IMAVIS_EDGE_SEG/` | NADUNGVN/imavis-edge-seg | TBD under NFS | TBD under NFS | SERVER-01..05 | E1 (Hailo SW ready), E2 (Xavier NX, ready), E3 (AGX Xavier, ready), E4 (RUBIK Pi 3, secondary only), E5 (Orin Nano Super, ready — role vs legacy Nano open) | Vision segmentation track; independent of CARE-ASD audio pipeline |
```
