# Release checks and bounded interpretation

This record describes engineering checks on the 2026-10-28 release. It is not
clinical validation, a security certificate or a new mechanism result.

## Code and viewer

- Thirteen data-independent core/privacy tests passed, including numerical input
  contracts, a small synthetic PointNet++ forward and five privacy-binding
  contract checks. No real-patient model forward or fit was required to package
  or publish the code.
- Seventeen viewer/input tests passed, including both aggregate-cohort wrappers,
  four-arm display, missing-data handling, archive/array validation and the
  default-disabled optional upload interface.
- The isolated viewer has no PyTorch dependency. Dependency consistency checks
  passed. A real local launch returned HTTP 200 and `ok` on its health endpoint.
  Local reachability is not a public deployment. Browser visual preview through
  the desktop connector was unavailable in this run, so no visual browser test
  is claimed.
- Input checks are bounded tests, not a guarantee against every malicious file.
  The optional uploader transfers bytes to its server. It requires two opt-ins
  and RAM-backed temporary buffering under the supplied local launcher.

## Artwork

- Five main, four Extended Data and three supplementary figures are provided
  as SVG/PDF/PNG. The combined PDF has twelve pages.
- Every PDF page was rasterized and visually inspected. A text-bound check found
  no spans extending beyond the page. SVG files retain editable text, axes,
  equations and diagram components; real-data rendering layers remain raster.
- The figure output manifest excludes itself and each listed SHA256 was checked.
- Real illustrations use the first four pre-fixed Lausanne DEV cases. Their
  deterministic manifest order, common camera and point correspondence avoid
  performance-based case selection. Reference identity colors are explanatory,
  not input features to the deployed Gate interface.
- The source MRA illustration supplies clinical-image context. The historical
  Gate conditions on a given binary vessel foreground and geometry; it does not
  read the displayed source intensity as an additional input channel.
- The displayed SA1 query has 32 neighborhood slots and 21 unique source points,
  including repeated self-padding. Proximity neighborhoods do not certify
  anatomical vascular connectivity.

## Scientific and release boundaries

Protected internal-cohort raw data, closed FINAL tuning, original Y/Z states,
scientific gates and STOP are unchanged. Previously saved aggregate results
may be displayed without reopening raw protected data. Negative or failed
mechanism versions are not recast as successes. Sampled-point Dice is not
full-volume segmentation Dice or candidate identity performance.

No original patient arrays, private path-bound runners, credentials, model
weights or authorization to redistribute datasets are included in the portable
bundle. The owner approved GitHub publication on 2026-10-28, without selecting
an open-source code license or approving hosted deployment. The public
reconstruction helpers replace an embedded historical source-ID list with its
ordered-list SHA256 commitment; the corresponding privacy adaptation is
recorded and leaves numerical model functions unchanged. The locally preserved
unredacted original-code archive is explicitly not a public upload package.
