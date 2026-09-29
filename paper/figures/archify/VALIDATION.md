# Archify validation record

Both diagrams were authored and finalized with Archify 3.0.1 at showcase quality.
The editable JSON and final SVG/PDF vectors are committed; large standalone HTML
viewers and browser screenshots are local QA artifacts and are intentionally ignored.

| Figure | Candidate SHA-256 | Final artifact SHA-256 | Gates | Visual review |
|---|---|---|---|---|
| Fig. 1 PACE-Seg system | `440f7cd95fb487f2a84d1300ac33e3082953a9b8a18bfa507a512ef93d77e3fe` | `45a56e4085153c4b7a5d18b2c091b9afc2491fcae5de88bfdfeb8d947a17971d` | validate, deliver, check, browser-check: pass | Final light export inspected; no crossing or clipping remained |
| Fig. 2 elastic deployment | `b3d8ad0f4b9bb0d77b8b867f5abf36b39273f2b46f60fd37ba0c9ffbd89a3d62` | `fe0a8c352cb117b27ddef998dd3e2b883ab76dcef3b994a7d34e5b5742434565` | validate, deliver, check, browser-check: pass | Final light export inspected; one three-bend shared-weight route was accepted because it remained unambiguous at print size |

The artifact hashes refer to the validated standalone HTML outputs. Committed SVG/PDF
hashes can change with the browser or PDF writer without changing diagram semantics;
source JSON is the canonical editable representation.
