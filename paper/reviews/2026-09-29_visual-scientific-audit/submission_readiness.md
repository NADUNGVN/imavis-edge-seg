# Submission-readiness statement: visual-scientific v3

## Verified

- Figure 1 explains the complete offline/runtime PACE-Seg path from code-backed nodes.
- Figure 2 shows the four related capacities and both primary static deployment paths.
- Figure 3 parses canonical FLOPs and measured mean-latency artifacts.
- Figure 4 parses all three supernet and three Fast-SCNN evaluation files without
  inventing seed pairing; its whiskers are explicitly descriptive.
- The 31-page review PDF compiles and was inspected page by page. Figure scale,
  captions, table overflow, author encoding, and float order were checked.

## Unresolved blocker

The final RQ3 quality--latency figure and headline remain blocked by the fit/deployment
probe-feature mismatch in `scripts/evaluate_router.py`: fitting uses a ground-truth
ignore mask while deployment cannot. The manuscript keeps the current router numbers
as conditional diagnostic evidence.

## Acceptable limitations

- No qualitative panel is included because the frozen repository lacks the licensed
  RGB, mask, and prediction raster assets required to construct it honestly.
- E1 and E3 reuse predictions; the operating cells are correlated; QAT is fake-quant;
  no energy result is claimed.

## Canonical conditional router arithmetic

120 total cells; 69 fair cells; D versus A 51 wins, 16 ties, 2 losses; macro
`+0.0241` mIoU; D `0/120` and A `51/120` violating cells. These numbers are not
presented as independent trials or a frozen deployment-matched headline.
