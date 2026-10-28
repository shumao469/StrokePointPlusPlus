# Manuscript and figure outline

## Working title

**Candidate evidence for anatomical identity control in cerebral vessel point clouds**

Software name: **StrokePoint++**. Application: **StrokePoint++-Gate**.
This name describes a research direction, not a validated stroke diagnosis or prognosis system.

## Central, supportable contribution

The implemented application aggregates frozen PointNet++-derived point features
over predicted vessel candidates and estimates whether selected small-vessel
classes are present. It can reduce observed absent-class false detections, but
the effect must be evaluated alongside point identity performance, rare true
positives and patient-event errors. Raw-image strong3D is a necessary comparator.

The current evidence does **not** establish a new PointNet++ architecture, OT
mechanism, superior joint-learning algorithm or formal clinical safety guarantee.
The framework figure emphasizes the actual computational decomposition, not a
claim of architectural novelty that failed its controls.

## Results narrative

1. **Conditional vessel identity task and evidence-aware application** (Main1).
   Describe a supplied binary foreground, shared coordinate/sampling contract,
   frozen representation and separately fitted presence classifiers. Keep the
   original paired input and reconstructed single-MRA version separate.
2. **Hierarchical representation and candidate decision** (Main2).
   Define sampled neighborhoods, saved point features, 256-dimensional mean/max
   pooling, logistic probabilities and reassignment. Explain that proximity is
   not connectivity, and presence classification is not point-action risk.
3. **Original application evidence** (Main3).
   Describe development as exploratory and the original internal holdout as
   internal, with effective patient denominators. Correct the relative FPR
   reduction to43.52%. Preserve the historical audit caveats and oracle status.
4. **External single-MRA comparison** (Main4).
   Report both external cohorts without pooling or retuning. Strong3D has
   higher observed overall pointDice than the point-cloud application, while
   gating has lower observed FPR in FINAL. This is a tradeoff, not dominance.
5. **Rare-vessel limits and correction safety** (Main5).
   Identify Acom and third-A2 failure, the actual1rareTP loss, corrected versus
   new point errors, and the unpassed patient-event risk boundary. Keep
   incomplete bootstrap intervals and sparse negative supports visible.

## Editorial positioning

A polished figure package does not establish NC/NMI readiness. Current results
support an honest application/reproducibility manuscript and a careful
comparative benchmark. Any new architecture study requires a separately frozen
DEV-only repair, full same-input strong controls and a future genuinely
unexposed test source. Do not use the now-opened FINAL to choose such a repair.

## Reference adaptation

Dai et al., Nature Methods23:760-771(2026), doi:10.1038/s41592-026-03034-9,
provides a useful example of explicit spatial inputs, encoder detail, modular
tasks and real-data validation. Its spatial-omics SCOT/Sinkhorn objectives are
not transferred into this manuscript as if implemented. The supplied RetinaFuse
deck informs typography, white space, semantic colors and real-data placement;
all StrokePoint++ scientific artwork is newly drawn from this project's evidence.
