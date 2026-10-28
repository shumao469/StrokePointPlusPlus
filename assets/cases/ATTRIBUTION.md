# Real external-DEV illustrations

These are real Lausanne MRA external-development cases, not generated anatomy.
CaseA–CaseD are the first four cases in the already-frozen NC03 visible-manifest
order, selected before inspecting their displayed results. They are illustrative
development examples, not additional validation cases or a favorable-case sample.

All truth and model panels for one case use exactly the same reviewed 2048
sampled points, physical geometry, class palette, camera and scale. The surface
is the supplied vessel-foreground mask, not a predicted segmentation. The MRA
panel shows maximum-intensity projections cropped using that given foreground.
Set-abstraction illustrations use the saved128-point and32-point hierarchy.
No new model inference or training was performed to make these illustrations.

`CaseA_grouping.png` shows the saved first anchor and its Euclidean grouping,
not vessel connectivity:2048gray background points,128blue anchors, a red fixed
query and gold saved-neighbor relations. The32stored neighborhood slots contain
21unique points for this query, including repeated self-query padding. No new
nearest-neighbor search or topology estimation was used.

Original Lausanne images are CC0. The derived CoW annotations are separate works
provided by the TopCoW organizers under their documented source-attribution and
noncommercial-use terms; commercial use requires permission. The study's supplied
creator clarification supports noncommercial research and aggregate publication,
not an unrestricted relicensing or redistribution of annotation volumes.

Sources to cite:

- Yang K, Musio F, Ma Y, et al. *The TopCoW Challenge—Topology-Aware Circle of
  Willis Segmentation for CT and MR Angiography*. NEJM AI (2026).
  [doi:10.1056/AIdbp2500994](https://doi.org/10.1056/AIdbp2500994).
- TopCoW Challenge Organizers. *TopCoW Training Data and External Testsets* (2025).
  [doi:10.5281/zenodo.15692630](https://doi.org/10.5281/zenodo.15692630).
- Di Noto T, Marie G, Tourbier S, et al. *Towards Automated Brain Aneurysm
  Detection in TOF-MRA: Open Data, Weak Labels, and Anatomical Knowledge*.
  Neuroinformatics21,21–34(2023; online2022).
  [doi:10.1007/s12021-022-09597-0](https://doi.org/10.1007/s12021-022-09597-0).

Source identifiers and exact input/output hashes are recorded only in the
excluded private-build provenance. Public image filenames and captions do not
contain source identifiers. These illustrations confer no clinical safety claim.
