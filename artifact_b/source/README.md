# ECE 253 Artifact B Stage 1 and Stage 2

This package contains the completed synthetic pretraining and synthetic fine-tuning experiments, the validation analyses, source code, and the complete prior Artifact B project in `prior_project/`. The held-out test identities were not opened during this phase. No Stage 3 or final held-out evaluation was run.

## Reproduction

The package was run with Python 3.12, PyTorch 2.9.1 CPU, OpenCV 5.0.0, ZXing-C++ 3.1.1, NumPy, Pillow, and Matplotlib. The runner imports the original four-scale Compact U-Net and restoration/decoder implementations from `prior_project/ArtifactB_Next_Research/src/`.

From this directory, use the workspace Python environment or install the compatible packages, then run:

```bash
python experiment/src/stage1_stage2.py --stage stage1_data  # regenerate deterministic Stage 1 pairs if removed
python experiment/src/stage1_stage2.py --stage train --stage1-steps 240 --stage2-steps 240
python experiment/src/stage1_stage2.py --stage stage1_evaluate
python experiment/src/stage1_stage2.py --stage evaluate
python experiment/src/stage1_stage2.py --stage samples
```

Stage 1 validation arrays, Stage 2 audited training/validation pairs, pair manifests, checkpoints, prediction-level CSVs, aggregate tables, and figures are included. The two large Stage 1 training arrays are omitted from this compact archive because the deterministic generator recreates them with `--stage stage1_data` from the recorded seed and manifest. Rebuilding Stage 2 image pairs from the original 940 training and 220 validation photos requires the six original `synthetic_shadow_pilot_images_*_of_6` ZIP volumes supplied with the project; the already prepared, audited Stage 2 pairs are included here so model training and evaluation do not depend on those external image volumes.

## Experiment design

- Stage 1 has 2,000 unique new QR identities for U-NET SINGLE and 2,000 for U-NET ALL, plus 200 disjoint synthetic validation identities. Each training arm has 6,000 aligned pairs; validation has 1,400 pairs.
- U-NET SINGLE uses S, D, and G only. U-NET ALL uses S, D, G, SD, SG, DG, and SDG. Degradations include spatial cast shadows with varied angles, blur, glare, sensor noise, and JPEG encoding. Generated module matrices are known.
- Stage 2 uses 693 valid aligned pairs from 36 of the 45 locked training identities. The remaining source images were excluded when their clean reference could not pass dual-decoder verification or the input-only geometry checks. Validation has 213 geometry-valid pairs from the locked validation identities; all 220 source validation images remain in the decode-rate denominator.
- The training branches use the same architecture, seed, initialization policy, sampling order, geometric augmentation RNG, batch size, optimizer, learning rate within each stage, validation cadence, and 240-step budget. The treatment is the impairment distribution for H1 and the loss for H2. H2 uses the same Stage 1 ALL-standard initialization and the same Stage 2 image pairs.
- Stage 2 matrix references are sampled from paired Clean photos and checked against the recorded payload using OpenCV and ZXing-C++. They are not independently annotated, pixel-aligned module ground truth; Stage 2 MER fields are therefore intentionally blank.
- The same fixed ROI policy and ZXing-C++ exact-payload decoder are used for all compared Stage 2 methods. The decoder receives no oracle geometry or ground-truth payload.
- The Stage 2 comparison is validation-only. The earlier real-photo test identities were examined in prior work, so any historical real benchmark number is not an independent test result.

## Contents

- `experiment/src/stage1_stage2.py`: data generation, training, validation, and visualization code.
- `experiment/configs/run_config.json`: seeds, model, data counts, and fixed training budget.
- `experiment/data/`: aligned Stage 1 and Stage 2 arrays and manifests.
- `experiment/models/`: best validation-selected Stage 1 and Stage 2 checkpoints.
- `experiment/logs/`: training histories, validation predictions, exact decode summaries, paired recovery/regression contrasts, and matrix-reference audit.
- `experiment/figures/` and `experiment/samples/`: actual output panels, zooms, curves, and examples.
- `experiment/diagnostics/`: preserved exploratory runs excluded from final H1/H2 claims, including a target-version mismatch run and a corrected-target but unmatched-seed run.
- `prior_project/`: unchanged source project and previous results, including the previous report and trained baseline checkpoints.

See `experiment/README_RESULTS.md` for the final measured findings and explicit proposal completion status.
