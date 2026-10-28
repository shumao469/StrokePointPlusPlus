# StrokePoint++

[GitHub repository](https://github.com/shumao469/StrokePointPlusPlus)

**StrokePoint++-Gate** is the application line: a frozen PointNet++-derived
representation, predicted-segment pooling and saved presence classifiers.
The package also contains the stronger raw-MRA 3D comparator, a simple fusion
control, and explicitly named research versions. Real vessel images and point
clouds below come from a deterministic public-data development series, not AI
generated anatomy.

Research engineering release, version `0.1.0.dev0`. The name is a project label:
it **does not mean stroke diagnosis, infarct prediction, prognosis or clinical
safety has been validated**. This package addresses anatomical labeling of
cerebral vessels **given an existing binary vessel foreground**.

| Version name | What it contains | Interpretation |
| --- | --- | --- |
| Gate-AppYZ-v1 | Original paired CTA/MRA application | Historical sealed application, not overwritten |
| Gate-MRA-R0v2 | Rebuilt fixed-recipe heads, single MRA | Separately labeled external-transfer reproduction |
| StrokePoint-3D | Raw-MRA 3D comparator | Higher observed external point Dice, short pilot |
| StrokePoint-Fuse | Simple image/point feature concatenation | Matched-input comparator, no superiority claim |
| Research archive | Joint/presence-feedback and earlier implementations | Versioned exploration, negative findings retained |

## What you can operate

1. **Inspect evidence and versions**: list Gate-AppYZ-v1, Gate-MRA-R0v2,
   StrokePoint-3D and StrokePoint-Fuse, check source/model hashes, and inspect
   reviewed external aggregate results without opening subject data.
2. **Run a parameterized model interface**: bind your separately authorized
   weights and saved scalers by SHA256, supply the documented label-free NPZ
   arrays, and produce anatomical point labels. Missing weights fail clearly;
   this package never fills them with random states or starts fitting.
3. **Explore actual vessels and point clouds**: view the MRA/surface/point-cloud
   examples, compare the fixed model outputs, and optionally visualize a local
   uploaded NPZ in server memory. This last feature is off by default and does
   not infer anatomy, diagnose stroke or save the uploaded file.
4. **Rebuild publication artwork**: reproduce five main figures, four Extended
   Data figures and three supplementary figures from locked aggregate source
   tables and attributed real-data renderings. SVG text, equations and plot axes
   remain editable; true image/3D renderings are raster layers.

## Interactive viewer / Streamlit

The app is a **CPU-only results viewer**, not a prediction endpoint. It needs
no dataset, model weights, GPU, API key or account login to run locally.
The owner approved this public repository on 2026-10-28; no Streamlit Cloud
deployment was requested or performed.

Clone the repository first:

```bash
git clone https://github.com/shumao469/StrokePointPlusPlus.git
cd StrokePointPlusPlus
```

Windows PowerShell：

```powershell
py -3.10 -m venv .venv-app
.\.venv-app\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv-app\Scripts\python.exe -m streamlit run apps/streamlit_app.py --server.address 127.0.0.1 --server.port 8851 --browser.gatherUsageStats false
```

Linux / WSL：

```bash
python3 -m venv .venv-app
.venv-app/bin/python -m pip install -r requirements-app.txt
.venv-app/bin/python -m streamlit run apps/streamlit_app.py --server.address 127.0.0.1 --server.port 8851 --browser.gatherUsageStats false
```

Open **http://127.0.0.1:8851/** after startup. Stop your foreground app with
`Ctrl+C`. If the port is occupied, choose another port (for example `--server.port
8852`) without stopping unrelated research services. Missing figures or aggregate
files should be repaired from the repository, not replaced by invented results.

The viewer was tested with Python 3.10; other Python environments require their
own dependency/tests check. For optional future Cloud deployment, see the
platform's current supported Python versions in [DEPLOYMENT.md](DEPLOYMENT.md).

The local URL is `http://127.0.0.1:8851/` while that process is running. On the
construction workstation, `deployment/start_local.ps1` manages the separate
environment and records the actual launch receipt. See [deployment instructions](DEPLOYMENT.md)
for Streamlit Community Cloud. A `.streamlit.app` URL exists only after an
authorized repository and Cloud deployment succeed; no public deployment is
claimed by including these instructions. No raw subject data or weights should
be placed in a public repository.

Viewer pages expose cohort-specific reviewed metrics, confidence intervals,
rare-branch support and the figure gallery. They keep Lausanne DEV and IXI-HH
FINAL separate. Clinical use and automated retraining are disabled.

```bash
python -m pip install -r requirements-figures.txt
python scripts/make_publication_figures.py
```

The real-data rendering scripts require separately authorized, locally bound
development data and private configuration; the default figure builder uses
only the already rendered public illustrations and aggregate source tables.
It never opens the protected 65-patient raw data, fits a model or tunes FINAL.
If Arial is available outside the package, optionally set `STROKEPOINT_FONT_DIR`
to that authorized font directory before rebuilding; otherwise DejaVu Sans is
used. No proprietary font is bundled.

## Scientific status / 目前结论

- `Gate-AppYZ-v1`: sealed historical application with paired CTA/MRA inputs.
  In its internal cohort, native-absence FPR decreased approximately
  `0.125651 -> 0.070964` (43.52% relative reduction, not55%). Dice has65 patient
  units; absence FPR has64 valid patient units. This is not independent external
  validation. Non-significant rare-branch changes do not establish noninferiority
  or safety; some earlier rare-branch sample counts were scans, not patients.
- `Gate-MRA-R0v2`: newly reconstructed fixed-recipe LR/scaler heads, deployed
  using **one MRA** and the fixed four-source latent average. It is not recovered
  original fitted state and not input-equivalent to paired CTA/MRA-max inference.
- `StrokePoint-3D`: enhanced3D U-Net comparator (official DNA architecture,
  local physical patches, coordinates and point readout), a short2000-update
  pilot.
  One inner training fold had no native third-A2 positives; keeping13 outputs
  does not create missing rare-class training support.
- `StrokePoint-Fuse`: simple image/PointNet++ concatenation comparator.
  Joint FiLM/presence-feedback variants remain research/negative-result records;
  they are not presented as a successful new mechanism.
- Lausanne20 is externalDEV; IXI-HH20 is a model/source-record-qualified fixed
  externalFINAL cohort in the current study, not proof of globally never-used
  data. 

## Installation

Python3.10 or later. Metadata commands use only the standard library:

```bash
python -m pip install -e .
python -m strokepointpp inspect
python -m strokepointpp summary
python -m strokepointpp verify
```

For inference/synthetic tests, install the optional dependencies in a separate
environment. Install the appropriate official PyTorch CPU/CUDA build for your
machine first; this bundle does not modify drivers or CUDA installation.

```bash
python -m pip install -e ".[inference]"
python -m unittest discover -s tests -v
```

The study used Python3.10, PyTorch2.9.1+cu128,
dynamic-network-architectures0.4.2, nnU-Net2.6.2 and scikit-learn1.7.2.
The precise recorded versions/configurations are retained in
`configs/spatial_fixed_pilot_v1.json`. nnU-Net is contextual provenance; the
adapter uses DNA's `PlainConvUNet`, not the entire nnU-Net training pipeline.
Newer software/device combinations require separate numerical validation.

### Library roles / 依赖库用途

| Library | Role | Needed for |
| --- | --- | --- |
| Python standard library | CLI, JSON manifests, SHA256 and allowlisted packaging | Basic inspection and verification |
| PyTorch | Frozen PointNet++ / 3D model numerical operations | Optional explicit inference |
| dynamic-network-architectures | `PlainConvUNet` implementation | Strong-3D / fusion adapter |
| NumPy, SciPy | Array geometry, interpolation and numerical statistics | Inference and figure building |
| scikit-learn | Archived LR, scaling and DBSCAN provenance | Research reproduction, no default refitting |
| nibabel | Authorized NIfTI orientation and spatial coordinates | Separate data preparation / rendering |
| Streamlit, Plotly | CPU-only aggregate and 3D-point viewer | Local app or separately approved Cloud deployment |
| Matplotlib, PyMuPDF | Editable scientific artwork and PDF assembly | Figure reproduction |
| PyVista / VTK, scikit-image | Real point-cloud/surface rendering | Optional local real-case illustration scripts |

Keep the application, inference and figure environments separate. The viewer
does not require PyTorch or CUDA and does not occupy the training GPUs.

## Components and package boundary

```text
strokepointpp/              installable metadata CLI and new engineering adapters
strokepointpp/_frozen/      original algorithm functions, import namespaces adapted
archive/algorithms/        selected sources, with documented privacy adaptations
configs/                   fixed settings and local asset mapping example
results/aggregate/         independently reviewed aggregate statistics only
manifests/                 source hashes, dependency/import audit, public file hashes
tests/                     synthetic/data-independent tests
scripts/build_release.py   allowlisted reproducible ZIP builder
```

The local construction folder also contains `private_archive/source/`: exact
path-bound historical runners and their import closure. It is deliberately
**excluded from the public ZIP**, because those scripts embed investigator-local
paths. No original patient images, labels, per-patient predictions, caches or
fitted model weights/scalers are distributed. Public source adaptations do not
silently alter frozen numerical algorithms. The two public reconstruction helper
copies replace the historical training-source ID list with an ordered-list
SHA256 commitment. They preserve the same private-binding check and do not
change model arithmetic. Original unredacted scripts stay local, not on GitHub.
Import and privacy transformation records are in
`manifests/import_adaptations_v1.json` and
`manifests/privacy_transformations_v1.json`.

## Assets and explicit inference

Obtain model assets from the project owner through an approved channel. Data
permissions do not automatically grant model redistribution rights. The bundle
provides SHA256 and logical IDs, not download URLs, private paths or weights.
Copy `configs/model_assets.local.example.json` outside the bundle and replace
each placeholder with your authorized local asset path; all mapped assets are
checked against the fixed manifest before deserialization.

```bash
python -m strokepointpp verify --assets my_assets.json
python -m strokepointpp infer --method gate --assets my_assets.json --input visible_gate.npz --output gate_prediction.npz --device cpu
python -m strokepointpp infer --method 3d --assets my_assets.json --input visible_spatial.npz --output three_d_prediction.npz --device cuda:0 --fold 0
python -m strokepointpp infer --method fuse --assets my_assets.json --input visible_spatial.npz --output fusion_prediction.npz --device cuda:0 --fold 0
```

Inference deliberately fails on missing assets, wrong SHA, unexpected keys or an
existing output. It never substitutes random weights, downloads data, fits heads,
searches thresholds or executes training automatically. Individual spatial-fold
commands are single-model research outputs; the archived external comparison
averaged both locked folds' logits, so a single-fold command is not its exact
replication. Gate uses all four fixed fold/seed sources.

## Prepared visible input contracts

The adapter consumes caller-prepared **label-free NPZ**, not arbitrary NIfTI.
The preparation/training/evaluator source is locally archived for audit; a
turnkey raw-image clinical pipeline has not been validated by this release.

Gate input requires exactly these14 keys:

```text
xyz float32[2048,3]               centered physical world coordinates /100
s1 int64[128], s2 int64[32]       original coordinate-only FPS
g1 int64[128,32], g2 int64[32,32]
i21 int64[128,3], w21 float32[128,3]
i10 int64[2048,3], w10 float32[2048,3]
native_gd float[32,128]          physical foreground-graph path distances; +inf allowed
path_features float32[32,128,2]  stack(where(finite,gd/120,1),finite_bit)
euclidean_ids int64[32,K], euclidean_mask bool[32,K], euclidean_active bool[32]
```

The frozen legacy route overrides i21/w21 using the three smallest finite native
path distances, deterministic index ties and inverse-distance weights. This is
not silently replaced by a new Euclidean interpolation or learned topology.
The four independently trained sources' hidden coordinates were not aligned;
their historical averaging is preserved, not claimed to be semantically aligned
or newly validated by this software packaging step.
Geometry, foreground and point order must be produced under a prospectively
bound protocol, never chosen using true anatomical identities or known errors.
Manual foreground/ROI, when used, is disclosed as a conditional input, not
automatic segmentation. No targets such as `y` or native class presence are
accepted by the Gate input contract.

Spatial input requires `volume_CDHW float32[1,96,96,96]`,
`local_NCDHW float32[2048,1,5,5,5]`, `point_grid_XYZ float32[2048,3]`,
`point_xyz float32[2048,3]`, and the eight SSG geometry keys s1/s2/g1/g2/
i21/w21/i10/w10. Volume axes are K,J,I; grid order is I,J,K normalized[-1,1],
align_corners=True. Local offsets are[-1.2,-0.6,0,0.6,1.2]mm, not native voxels.
The archived protocol uses foreground-bbox+5mm and that scan's foreground
intensity mean/std. Incorrect orientation, units or foreground sampling can
invalidate predictions even if array shapes match.

CPU Gate arithmetic is float32 feature extraction plus fixed NumPy/LR rules.
Spatial CUDA inference uses BF16 autocast with explicit FP32-sensitive kernels,
TF32 off. CPU/CUDA exact-label replay is not automatically certified here.
Returned gated probabilities are class-blocked diagnostic renormalizations,
**not validated event-risk probabilities**. Blocking reassigns labels; it is
not a true “unknown” state.

## Evaluation, reproduction and limitations

Original evaluation distinguishes sampled point Dice, complete-native-mask
absence FPR, candidate identity, corrected/new error points, patient-any-error
risk and rare-branch TP loss. Units and denominators must remain separate.
Patient-level bootstrap uncertainty is retained; zero errors or a CI crossing
zero does not establish a10% risk bound or noninferiority.

The package includes executable numerical components and provenance, not the
datasets or authorization to reopen protected data. Reproducing full training
needs authorized original data, original split/source bindings, sufficient
compute, target quarantine and new engineering verification. Historical Gate
and spatial branches used different training sources/counts; their external
comparison is not an isolated, training-budget-matched architecture ablation.
Do not use the protected65 cohort or closed FINAL cohort for model selection,
preprocessing tuning, risk calibration or threshold search.

```bash
python scripts/build_release.py --source-workspace YOUR_AUTHORIZED_STUDY_WORKSPACE
```

The builder copies only an explicit code/metadata allowlist, preserves local
private originals, checks reviewed-statistics hashes and creates a deterministic
content ZIP with fixed member timestamps. It never executes study scripts,
loads patient arrays/model weights, trains or sends messages.

## Licenses, citations and publication

See `LICENSE_NOTICE.md`: repository publication is owner-approved, but no
MIT/Apache or other open-source code license has been selected. Public visibility
is not a blanket reuse license. Third-party packages retain their upstream
licenses.

- TopCoW release: https://zenodo.org/records/15692630 — cite the TopCoW paper
  requested by the organizers and applicable source datasets.
- Lausanne original images: CC0; derived annotations are separately provided
  under the organizer-confirmed TopCoW/OpenData Swiss terms.
- IXI-HH original images: CC BY-SA3.0; comply with its attribution/ShareAlike
  conditions. Derived annotations are under the organizer-confirmed terms.
- Creator-provided annotation clarification permits noncommercial method
  development/training/evaluation and aggregate publication with source citation;
  commercial permission and image/model redistribution questions remain separate.
- PointNet++ method: Qi et al., “PointNet++: Deep Hierarchical Feature Learning
  on Point Sets in a Metric Space”, NeurIPS2017.
- Visual narrative reference, not an implemented OT module: Dai et al.,
  “3d-OT: a deep geometry-aware framework for heterogeneous slices alignment
  of spatial multi-omics”, Nature Methods23,760-771(2026),
  https://doi.org/10.1038/s41592-026-03034-9 .
- TopCoW benchmark: Yang et al., “TopCoW Challenge—Topology-Aware Circle of
  Willis Segmentation for CT and MR Angiography”, NEJM AI3(8),AIdbp2500994(2026),
  https://doi.org/10.1056/AIdbp2500994 .
- Official3D implementation: https://github.com/MIC-DKFZ/dynamic-network-architectures
  and https://github.com/MIC-DKFZ/nnUNet .

