# Hardware re-measurement — static vs adaptive route (review round 2, 2026-10-03)

## Why this is needed before revising RQ3

Offline re-analysis (`reports/router_review_analyses_v2_20261003.json`) showed:

- With the static reference charged its own direct latency from the candidate-only LUT
  (no probe/router), static beats routing by about 7.2 mIoU points at about 1.6× lower
  mean cost, and static mixing beats routing at equal mean cost.
- But the two numbers come from **different harnesses**: routes from the Python
  overhead harness (full-logit D2H, per-call Hailo activation, FLOAT32 vstreams),
  static from `trtexec` / `hailortcli`. The comparison is therefore not clean in
  either direction.

The two new scripts time static and route classes in the **same process, same
engines, same inputs, same timer boundary**.

## E3 (Jetson AGX Xavier, TensorRT) — also E2/E5 if time allows

```bash
cd ~/<repo>                       # same checkout used for router_overhead_E3_20260922.json
git pull                          # needs scripts/measure_static_vs_route_trt.py
sudo nvpmodel -q                  # note the mode; use the SAME mode as 2026-09-22
sudo jetson_clocks --show > reports/edge/E3_clocks_20261003.txt
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
cd scripts
python measure_static_vs_route_trt.py --engine-dir <dir with pace_seg_{tiny,small,medium,large}.engine> \
    --device-label E3 --output-mode both \
    --output-json ../reports/static_vs_route_E3_20261003.json
```

Expected run time: about 16 classes × 550 iterations, a few minutes. The script records
`nvpmodel`, `jetson_clocks --show`, the CPU governor and temperatures in the JSON.

## E1 (Raspberry Pi 5 + Hailo-8)

Run the four combinations one after another (only one VDevice can be open):

```bash
cd ~/<repo>/scripts
for path in explicit scheduler; do
  for fmt in float32 uint8; do
    python measure_static_vs_route_hailo.py --hef-dir <dir with pace_seg_*.hef> \
        --path $path --vstream-format $fmt \
        --output-json ../reports/static_vs_route_E1_${path}_${fmt}_20261003.json
  done
done
```

If the scheduler path fails, the script stores the traceback under `errors` in the
JSON instead of numbers. Send the JSON as is.

## What to send back

The JSON files in `reports/` (5 files: 1 for E3, 4 for E1). The offline analysis then:

1. builds the static table S_h and the route table C_h from the SAME file;
2. reruns D, T-hard, A-hard, best feasible static and static mixing;
3. picks the E1 path (explicit vs scheduler) with the lower route cost, reporting both.

## Decision rule (fixed before seeing the numbers)

- If, with same-harness costs, routing still loses to static deployment at equal
  cost on both devices, the paper is reframed around measured-cost feasibility and
  hard-budget routing vs rank escalation; the static comparison is reported as a
  negative result with the overhead breakdown.
- If routing matches or beats static on at least one device or path (e.g. E3 with
  output-mode none, or E1 with the scheduler), the paper reports the two budget regimes
  with the measured overhead and the conditions under which routing pays off.
