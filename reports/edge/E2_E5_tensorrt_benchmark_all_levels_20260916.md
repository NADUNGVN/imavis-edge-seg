# E2 (Xavier NX) + E5 (Orin Nano) — TensorRT GPU/FP16 latency benchmark, all 4 levels — 2026-09-16

Closes the last 2 of 4 devices' TensorRT latency data. Run directly via SSH from
the dev machine (no server or user round-trip needed): the 4 elasticity-level
ONNX exports (`scripts/export_all_levels.py`, untrained weights -- `trtexec`
latency is weight-independent) were copied to both devices, then
`scripts/benchmark/run_trtexec_protocol.sh` run for all 4 levels x 3 independent
runs each, on both devices in parallel.

## Result: 4/4 levels, 3/3 runs each, on both devices, trtexec exit 0 throughout

| Level | Resolution | E2 (Xavier NX) mean (ms) | E5 (Orin Nano) mean (ms) |
|---|---|---:|---:|
| tiny | 384x192 | 2.061 | 1.081 |
| small | 512x256 | 4.964 | 2.604 |
| medium | 768x384 | 15.095 | 7.727 |
| large | 1024x512 | 40.257 | 17.951 |

For reference, the two already-benchmarked devices (2026-09-10/11):

| Level | E1 (Hailo-8) (ms) | E3 (AGX Xavier) (ms) |
|---|---:|---:|
| tiny | 3.714 | 0.912 |
| small | 6.597 | 1.781 |
| medium | 20.29 | 4.546 |
| large | 41.14 | 9.389 |

E2 (Xavier NX) is ~2.3-4.3x slower than E3 (AGX Xavier) at every level, consistent
with NX's lower tier in the Jetson lineup. E5 (Orin Nano) sits between E2 and E3,
consistent with Orin's newer architecture partially offsetting its
lower-power-tier positioning vs. the AGX Xavier.

## Now have a real, complete 4-device x 4-level latency table

`outputs/benchmark_lookup_table.csv` regenerated from all 16
device/level/backend combinations (`scripts/build_benchmark_lookup_table.py`) --
the full device-lineup latency coverage `RESEARCH_PLAN.md` §9 calls for (Hailo-8
on E1, TensorRT GPU on E2/E3/E5).

## Extended Pareto result: 4 real devices, same 10ms budget, 3 different choices

| Device | Best level under 10ms budget |
|---|---|
| E1 (Hailo-8) | small |
| E2 (Xavier NX) | small |
| E3 (AGX Xavier) | large |
| E5 (Orin Nano) | medium |

All four elasticity levels remain Pareto-optimal on every device (no
crossovers, same pattern as the earlier 2-device result). This is stronger
evidence than the earlier E1-vs-E3-only demonstration: 4 real, physically
distinct devices produce 3 different level choices at an identical latency
budget, reinforcing RQ1's premise that hardware-aware subnet selection is not
interchangeable with a one-size-fits-all choice. Still not yet: a FLOPs-aware
selection baseline to quantify this against (RQ1's actual "≥15-20% cost
reduction" hypothesis remains untested); energy axis (no power meter).

## Not yet done

- E4 (RUBIK Pi 3) uses a different toolchain (Qualcomm QAIRT), not
  TensorRT/Hailo HEF -- out of scope for this table under the current locked
  backend decision (TensorRT GPU + Hailo HEF are the two primary backends,
  DLA demoted 2026-09-10). Benchmarking E4 on its own toolchain, if ever
  needed, would be a separate, non-comparable table.
- DLA path not re-benchmarked on E2 (encoder-only, already demoted).
- No calibrated INT8 latency numbers anywhere yet (FP16 only).
