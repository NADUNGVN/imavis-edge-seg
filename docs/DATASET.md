# Dataset protocol

Minimum scope per `RESEARCH_PLAN.md` §6: **Cityscapes + ACDC**. Dark Zurich is external
validation only (never tuned on, zero-shot/adaptation-free). BDD100K is optional and
lower priority — do not add it at the cost of ablation depth.

## Primary corpora

| Resource | Location | Registration |
|---|---|---|
| Cityscapes | https://www.cityscapes-dataset.com/ | **Required.** Free account, agree to the dataset's own (non-standard) license terms — research/non-commercial use, no redistribution. |
| ACDC | https://acdc.vision.ee.ethz.ch/ | **Required.** Registration via the project site; similar research-only license. Test-set labels are not public — submit predictions to ACDC's own evaluation server for test-set numbers. |
| Dark Zurich (optional, external validation) | https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/ (or the ACDC site's related-datasets links) | Registration required, same research-only terms |
| cityscapesScripts (reference only, not a dependency) | https://github.com/mcordts/cityscapesScripts | MIT-licensed helper scripts; this project reimplements the id→trainId mapping itself (`src/imavis_edge_seg/data/labels.py`) so it does not need to be installed |

Both primary datasets require creating an account and accepting license terms manually
— this cannot be automated. Whoever downloads them should do so on the machine/server
that will hold the canonical copy (see `../../docs/SHARED_INFRASTRUCTURE.md` §0 for
`<SHARED_ROOT>` conventions), not on a laptop, unless that laptop is genuinely the
canonical store for this project.

## Expected directory layout

### Cityscapes

```text
<root>/leftImg8bit/<split>/<city>/<city>_<seq>_<frame>_leftImg8bit.png
<root>/gtFine/<split>/<city>/<city>_<seq>_<frame>_gtFine_labelIds.png
```

`<split>` is `train`/`val`/`test`. This is the layout produced by extracting
`leftImg8bit_trainvaltest.zip` and `gtFine_trainvaltest.zip` into the same root. Only
`labelIds` (raw 34-class IDs) ships with the download — `trainId` masks are **not**
precomputed; `CityscapesDataset` converts `labelIds` → `trainId` on the fly using the
standard 19-class collapse (`data/labels.py`), so `cityscapesScripts` does not need to
be run or installed separately. `test` split labels are not public (held out for the
Cityscapes benchmark server) — do not expect a `gtFine/test/` label tree.

### ACDC

```text
<root>/rgb_anon/<split>/<condition>/<scene>/<name>_rgb_anon.png
<root>/gt/<split>/<condition>/<scene>/<name>_gt_labelTrainIds.png
```

`<split>` is `train`/`val` (the public ones — `test` has no public labels).
`<condition>` is one of `fog`/`night`/`rain`/`snow`. Unlike Cityscapes, ACDC ships
`labelTrainIds` masks directly, already in the same 19-class scheme — no conversion
needed, and the label values from both datasets are directly comparable/mixable.

## Manifest workflow

A manifest is a small CSV (`image_path`, `label_path`, `image_sha256`, `label_sha256`,
paths relative to the dataset root) recording exactly which (image, label) pairs exist,
so every training job reads the same frozen file list instead of re-scanning a
multi-gigabyte raw tree, and so the raw data's identity is provenance-tracked without
committing the raw data itself
(`../../docs/SHARED_INFRASTRUCTURE.md` §1 rules 3-4).

```bash
uv run imavis-edge-seg data manifest \
    --dataset cityscapes --data-root /home/ubuntu/datasets/cityscapes --split train \
    -o data/manifests/cityscapes_train.csv

uv run imavis-edge-seg data manifest \
    --dataset acdc --data-root /home/ubuntu/datasets/acdc --split train \
    -o data/manifests/acdc_train.csv
```

Add `--condition fog --condition night` (repeatable) to restrict ACDC to specific
conditions; default is all four. Add `--no-checksums` for a faster manifest when
provenance hashing isn't needed yet (e.g. a quick local sanity check) — but the
manifest committed to git for an actual experiment should keep checksums.

Only run this against a **complete** download — a manifest built from a
still-extracting tree silently freezes a partial file list.

## Loading in code

`imavis_edge_seg.data.CityscapesDataset` / `ACDCDataset` (both `torch.utils.data.Dataset`)
take `root`, `split`, and a `SegmentationResizeToTensor(height, width)` transform — use
one of `SupernetConfig.input_resolutions[level]` from `config.py` so the loaded tensors
match the elasticity level being trained. Both raise `FileNotFoundError` immediately if
no samples are found for the given split/root, rather than silently training on an
empty dataset.

## Policy

- Evaluation-split labels are never used for tuning (Cityscapes `test`, ACDC `test`,
  Dark Zurich) — only for the final reported numbers, and only via each dataset's own
  official evaluation path where one exists (Cityscapes/ACDC benchmark servers).
- Calibration/routing thresholds (`RESEARCH_PLAN.md` §5.1 contribution 3) are fit on
  `val` splits only.
- Raw images/labels are never committed — `.gitignore` blocks common image extensions
  under `data/`; only manifests and this documentation live in git.
- Dark Zurich is zero-shot/adaptation-free external validation — do not tune anything
  against it, per `RESEARCH_PLAN.md` §6.
