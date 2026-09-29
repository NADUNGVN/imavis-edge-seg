# Manuscript review rounds

Each review round is an immutable directory named `YYYY-MM-DD_<reviewer-or-purpose>`.

## Required contents

| File | Purpose |
|---|---|
| `README.md` | Scope, repository snapshot, input manuscript, and outcome |
| `manuscript.md` or manuscript source link | Exact text reviewed |
| `reviewer_round_1.md` | Initial independent review |
| `reviewer_round_2_final.md` | Review after mandatory revisions |
| `submission_readiness.md` | Remaining blockers, accepted limitations, weakened claims, and canonical headline |

## Review protocol

1. Freeze the repository commit and list canonical reports and raw artifacts.
2. Review claims before prose quality: leakage, numerical consistency, hardware validity, dependence, novelty, and reproducibility.
3. Separate mandatory fixes from optional experiments.
4. Apply writing, citation, provenance, and claim-boundary fixes without altering experimental evidence.
5. Repeat review until no writing-fixable reject-level issue remains.
6. Create a new submission directory for the revised LaTeX source; do not overwrite a prior review input.

## Claim discipline

- Fair-cell results must be paired with full-grid budget violations.
- Correlated operating cells are not independent statistical samples.
- Shared prediction caches across backends do not provide independent accuracy replication.
- Fake-quant QAT is not compiled INT8 deployment accuracy.
- Hardware-cost-conditioned routing does not imply different decisions on every device.
- Every headline number must resolve to a canonical report or raw result file.
