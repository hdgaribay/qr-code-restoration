# Artifact B Stage 2 Scientific Repair

This package audits the prior Stage 1/2 experiments, reproduces stored Stage 2 validation results, diagnoses supervision alignment and domain shift, and evaluates a controlled Stage 2 repair. It does **not** access held-out test identities and does not run Stage 3.

## Contents

- `source/experiment/` — selected historical Stage 1/2 source data, logs, checkpoints, and figures. It contains training and validation material only; no held-out test manifest or test images are included.
- `source/prior_project/` — prior QR restoration methods and pilot U-Net checkpoints.
- `data/repair_train_*.npz` — new fully digital training pairs, generated from 37 QR matrix/capture units sourced from the 36 audited Stage 2 training identities.
- `data/known_geometry_validation.npz` — 2,400 fully digital validation-selection examples generated from 200 Stage 1 validation identities, with known shared geometric transforms.
- `models/` — three repaired checkpoints, each initialized from the same Stage 1 ALL standard checkpoint.
- `audit/` — checkpoint hashes, metadata reconciliation, domain-gap measurements, alignment-risk diagnostics, and `SHA256SUMS.csv` for all deliverable files except itself and temporary render QA files.
- `results/` — reproduced historical metrics, per-image predictions, bootstrap intervals, repair contrasts, training logs, and valid digital MER.
- `figures/` — readable actual-output comparison panels, module zooms, and metric plots.
- `ArtifactB_Stage2_Repair_Research_Addendum.docx` — editable report addendum.
- `src/` — audit, training, evaluation, visualization, and packaging scripts.

## Reproducibility

Use Python 3 with PyTorch, OpenCV, Pillow, NumPy, Matplotlib, and ZXing-C++ Python bindings. The evaluation decoder is ZXing-C++ 3.1.1; one fixed 256 x 256 ROI is passed to every method. Exact decode success is computed after decoding by hashing decoder text against the recorded payload hash. Payloads and module matrices are never supplied to the decoder.

From this directory:

```bash
python src/complete_audit_tables.py
python src/write_manifest.py
python src/train_aligned_repair.py
python src/evaluate_repair_models.py
python src/make_failure_visuals.py
```

Training uses base seed 7,253,325 and paired branch seed 7,253,425 for matched repaired models, 1,776 pairs per model, 240 optimizer updates, batch size 8, and learning rate 1e-4. The SINGLE model sees S-only examples; ALL sees balanced S/DS/SG/DSG examples. H2 uses the same ALL data and differs only by the loss. Checkpoint selection uses only known-geometry digital validation, not the existing 10-identity Stage 2 validation outcomes. Stage 2 validation is used for final model comparison and remains synthetic validation, not independent test evidence.

The QR module matrix is warped with the same recorded homography used to generate its degraded input. MER samples nine interior pixels per module after applying the inverse/forward geometry mapping. This is valid for the fully digital set only. Stage 2 photographic-derived ROIs have no independently verified pixel registration to their paired Clean matrix, so their MER is intentionally withheld.

## Key findings

The old Stage 2 summary lists step 120, but every Stage 2 checkpoint and its training log select step 240. The checkpoint metadata also carries the base data seed while the run log records the run seed. Historical validation decode counts reproduce exactly.

The historical Stage 2 pixel targets are idealized renderings of paired Clean matrices; their pixel-level registration to perspective-normalized degraded ROIs was not established. This is a supervision risk, not evidence that each individual target is misaligned. A detector-based geometry proxy finds larger residuals at severe levels, with shadow-related detection limitations. Stage 1 synthetic training also underrepresents severe shadows and differs in QR versions, blur, background texture, and luminance.

On the existing Stage 2 synthetic validation set, repaired ALL standard decoding reaches 41/55 at L3 and 21/55 at L4, versus RAW at 17/55 and 2/55. Original classical reaches 47/55 and 34/55; QR-aware classical reaches 45/55 and 41/55. Therefore, the repair improves over the old U-Nets and RAW at these levels but does not beat the classical methods. ALL exceeds SINGLE at L3/L4. Structure-weighted loss does not improve over standard loss at severe levels.

See the addendum and CSVs for per-condition results, identity-bootstrap intervals, latency, false decodes, and known-geometry MER. Results from the 200-identity digital validation are checkpoint-selection validation results, not independent test results.

## Proposal status

**Completed in this phase:** checkpoint and metric audit; historical validation reproduction; direct Stage 1/Stage 2/pilot/baseline comparison; alignment-risk and domain-gap audit; controlled matched digital H1/H2 repair training; existing Stage 2 validation comparison; known-geometry digital validation with MER; failure/recovery visual analysis; reproducibility package.

**Pending:** independent final synthetic test on an untouched identity set; any Stage 3 evaluation; independent real-world validation. Prior real-photo analysis had examined original test identities, so those identities retain that provenance and must not be presented as untouched independent evidence.
