# User–Claude–server collaboration protocol

This project uses two different collaboration modes, because the two hardware groups
in `../../docs/SHARED_INFRASTRUCTURE.md` are reachable differently:

| Hardware | Who connects | How |
|---|---|---|
| Train servers `SERVER-01..05` | The researcher only | Claude has no network path into `192.168.50.0/24` (or `SERVER-05`'s network). All server work goes through Git: Claude pushes code, the researcher runs one pasted command, the command commits a report back. |
| Edge devices `E1` (Pi5+Hailo), `E2`/`E3` (Jetson NX/AGX), Jetson Nano | Claude directly, once granted access | Once the researcher provides reachable SSH (Tailscale login for the owning account, or a forwarded IP/port + key), Claude SSHes in directly and runs inventory/install/benchmark commands itself — no copy-paste loop needed. |

This mirrors `../../CARE_ASD/docs/COLLABORATION_PROTOCOL.md`, adapted for this project's
repo name and hardware.

## Server track — responsibilities

| Party | Responsibility |
|---|---|
| Researcher | Runs the supplied command on the selected server, reports a failure immediately, reviews anything touching dataset/eval paths. |
| Claude | Writes/commits/pushes source changes, supplies one-line server commands, pulls and validates committed reports. |
| Server | Compute/storage only. Never the source of truth for code — all source changes and reviewable reports live in Git. |

## Command contract

- Every command given to paste into a server shell is **one physical line**.
- Commands use `&&` so a failed prerequisite stops later steps.
- Any command that takes material time writes a named report under `reports/server/`,
  then commits and pushes it. Claude pulls that commit to validate before issuing the
  next task.
- Report filenames include the server ID and a UTC timestamp/run ID; reports state git
  SHA, command, exit status, environment summary, and paths/hashes of relevant inputs.
- Never commit raw images/video, dataset archives, model checkpoints (`.pt`, `.onnx`,
  `.hef`, `.engine`), tokens, private paths, or evaluation labels. `data/raw/`,
  `outputs/`, logs and secrets stay on the server.

## Standard server lifecycle

1. Claude pushes source changes to `main`.
2. The researcher runs a one-line sync/setup command on the server.
3. The researcher runs the requested one-line task command.
4. The command writes a small, reviewable artifact in `reports/server/`, commits it, and
   pushes it to `main`.
5. Claude pulls the artifact, checks it against the intended contract, then issues the
   next task.

If the server worktree isn't clean before a new task, stop and send the output rather
than discarding changes.

## Required preflight artifact

Run this once after the first clone on a server, and again after any material
environment change (new CUDA driver, new JetPack, etc.). Produces only non-secret
metadata.

```bash
cd ~/IMAVIS_EDGE_SEG && git pull --ff-only && mkdir -p reports/server && RUN_ID="$(hostname)_preflight_$(date -u +%Y%m%dT%H%M%SZ)" && { printf '# IMAVIS-EDGE-SEG server preflight\n\n'; printf 'git_sha='; git rev-parse HEAD; printf '\ngit_status=\n'; git status --short; printf '\npython=\n'; python3 --version; printf '\nuv=\n'; uv --version; printf '\ndisk=\n'; df -h .; printf '\ngpu=\n'; nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader; printf '\nnfs_mounts=\n'; findmnt -t nfs4 2>&1; printf '\nimavis_edge_seg_environment=\n'; uv run imavis-edge-seg env-report; } > "reports/server/${RUN_ID}.md" 2>&1 && git add "reports/server/${RUN_ID}.md" && git commit -m "report: add ${RUN_ID}" && git push origin main
```

Replace `~/IMAVIS_EDGE_SEG` with wherever the repo is cloned on that server (record the
real path back into `docs/INFRA_OVERRIDE.md` once known). If `uv` is missing, report the
failure — Claude will supply a separate one-line install command rather than assuming a
different Python environment.

## Task report pattern (template for future tasks)

Report files use a `.md` extension (not `.log`) because `.gitignore` blanket-ignores
`*.log` — that rule exists so accidental scratch logs never land in git, and reports are
deliberately routed around it by naming.

```bash
cd ~/IMAVIS_EDGE_SEG && git pull --ff-only && RUN_ID="$(hostname)_<task>_$(date -u +%Y%m%dT%H%M%SZ)" && TASK_STATUS=99 && mkdir -p reports/server && { <task command>; TASK_STATUS=$?; printf '\ntask_status=%s\n' "$TASK_STATUS"; } > "reports/server/${RUN_ID}.md" 2>&1; git add "reports/server/${RUN_ID}.md" && git commit -m "report: add ${RUN_ID}" && git push origin main; printf 'task_status=%s (SSH shell stays open)\n' "$TASK_STATUS"
```

`<task>` and `<task command>` are always supplied by Claude for a specific step — never
improvised on the server. Failing tasks still commit their short log for inspection.

## Detached long-running jobs (supernet training, sustained benchmark runs)

For anything that must survive a closed SSH session, the repo provides a
`scripts/server/start_<job>.sh` wrapper (owns `nohup`/`setsid`, run-ID/state-file
creation) and a matching `scripts/server/status_<job>.sh` (reads `state.env`, no polling
sleep). Neither exists yet — added when the first real training job is ready (Phase 4 in
`../README.md`). Do not paste a full training loop directly into an interactive shell.

## Completion rule

A server task counts as done only once its report commit is on `origin/main` and Claude
has pulled and inspected it. A pasted terminal transcript is useful for quick triage but
is not the durable experiment record.
