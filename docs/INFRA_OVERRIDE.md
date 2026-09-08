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
| E1 — Pi5 + Hailo-8 | HW present, PCIe OK; **Hailo software NOT installed** (`/dev/hailo*` missing, no `hailortcli`) | Install `hailo-all`/`hailort`/`hailo-tappas-core` (candidates already listed in shared infra §3.2.1); run `hailortcli fw-control identify` and record chip (Hailo-8 vs Hailo-8L), firmware, HailoRT, Model Zoo, DFC versions **before any Hailo latency/energy claim** |
| E2 — Xavier NX | **Inventory pending** (JetPack 5.1.5 planned) | Full inventory per shared-infra §7 checklist; confirm DLA availability and JetPack/TensorRT versions before compiler-safe search space is finalized |
| E3 — AGX Xavier | **Inventory pending** (JetPack 5.1.5 planned) | Same as E2 |
| Jetson Nano | **Not yet in shared-infra inventory table at all** | Needs its own inventory row added to `SHARED_INFRASTRUCTURE.md` §3.1 before use; JetPack 4 is EOL (2024-11) — expect the most toolchain friction here |
| External power meter | Not recorded anywhere in shared infra | Confirm availability before claiming any J/frame number; without it, energy numbers must be clearly labeled as telemetry-derived (`tegrastats`), not system-level |
| Camera / video domain | Not recorded | Default to public datasets (Cityscapes/ACDC/Dark Zurich) unless a project-specific capture setup is confirmed |

## Current blocker (2026-09-08)

This session has no SSH/Tailscale path to any of `SERVER-01..05` or device `E1`
(`raspberrypi` is not present in the tailnet visible from this machine, and no LAN route
to `192.168.50.0/24` / `172.16.160.0/20` exists from this laptop). Everything under
Phase 1 of `README.md` (hardware inventory, `hailortcli identify`, JetPack/TensorRT
version pinning, first compiler smoke test) requires either:

1. Running the relevant commands directly on the lab machines (someone with physical/
   VPN access), or
2. Granting this session a working connection (Tailscale login for the account that owns
   `raspberrypi`/`SERVER-0x`, or a reachable SSH jump host).

Until then, work in this repo stays at the code/config/method-design level — buildable
and testable against synthetic tensors, not against the real accelerators.

## Mapping row for `../../docs/SHARED_INFRASTRUCTURE.md` §4

Added:

```text
| `IMAVIS_EDGE_SEG/` | NADUNGVN/imavis-edge-seg | TBD under NFS | TBD under NFS | SERVER-01..05 | E1 (blocked on Hailo SW), E2/E3 (inventory pending), Jetson Nano (not inventoried) | Vision segmentation track; independent of CARE-ASD audio pipeline |
```
