# Data

Raw/processed images and video are never committed. Only manifests belong in git.

| Dataset | Expected root | Notes |
|---|---|---|
| Cityscapes | `<SHARED_ROOT>/datasets/cityscapes/` | Fine + coarse; see `../docs/RESEARCH_PLAN.md` §6 |
| ACDC | `<SHARED_ROOT>/datasets/acdc/` | Official split; do not tune on test |
| Dark Zurich | `<SHARED_ROOT>/datasets/dark_zurich/` | External validation only |

Follow `<SHARED_ROOT>` conventions in `../../docs/SHARED_INFRASTRUCTURE.md` §0.
