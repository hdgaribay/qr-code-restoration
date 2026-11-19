# Stage 1 and Stage 2 validation findings

## Scope

All reported model comparisons are from synthetic validation data. Stage 1 evaluates 1,400 generated pairs from 200 new QR identities. Stage 2 evaluates the existing 220 synthetic validation images, representing 10 locked identities; 55 images are evaluated at each of L1–L4. All intervals resample QR identities. No held-out test image bytes were read, and no Stage 3 test was run.

## Stage 1

On the generated validation set, RAW decoded 1,388/1,400 (99.14%). U-NET SINGLE decoded 1,398/1,400 (99.86%), while U-NET ALL and U-NET ALL with structure-weighted loss each decoded 1,399/1,400 (99.93%). The paired SINGLE-to-ALL difference was +0.07 percentage points (identity-bootstrap 95% CI 0.00 to 0.21; 1 recovery, 0 regressions). This near-ceiling result offers little power to distinguish the training regimes.

The independently generated module matrices support valid Stage 1 MER. RAW MER was 4.821%; U-NET SINGLE was 0.0108%; U-NET ALL was 0.000679%; and structure-weighted U-NET ALL was 0.0000849%. The two ALL models both had zero measured errors in fixed, finder, timing, and alignment regions at the reported aggregate precision. Stage 1 inference plus decode latency averaged 24.3–24.6 ms for U-Nets and 0.38 ms for RAW on CPU.

## Stage 2 by severity

The most informative calibration point is L3 (illumination factor 0.30): RAW decoded 17/55 (30.9%), giving both successful and failed RAW cases. L4 (0.20) is near a RAW floor at 2/55 (3.6%). At L1 and L2, RAW decoded 96.4% and 87.3%, respectively.

| Method | L3 exact / 55 (identity 95% CI) | L3 recovery / regression | L4 exact / 55 (identity 95% CI) | L4 recovery / regression |
|---|---:|---:|---:|---:|
| RAW | 17 (17–47%) | 0 / 0 | 2 (0–10%) | 0 / 0 |
| Original classical | 47 (78–95%) | 30 / 0 | 34 (39–81%) | 32 / 0 |
| QR-aware classical | 45 (73–92%) | 28 / 0 | 41 (54–88%) | 39 / 0 |
| Fine-tuned U-NET SINGLE | 2 (0–10%) | 0 / 15 | 0 (0–0%) | 0 / 2 |
| Fine-tuned U-NET ALL | 5 (4–16%) | 1 / 13 | 2 (0–10%) | 2 / 2 |
| Fine-tuned U-NET ALL + QR loss | 5 (3–15%) | 1 / 13 | 2 (0–10%) | 2 / 2 |
| Prior standard U-Net pilot | 11 (14–26%) | 2 / 8 | 5 (4–16%) | 3 / 0 |
| Prior structure-weighted U-Net pilot | 14 (16–32%) | 4 / 7 | 8 (8–20%) | 6 / 0 |

The new fine-tuned neural models do not improve on RAW at L3 or L4 in aggregate, and they trail both classical restorers and the preserved pilot checkpoints. All methods produced zero false decodes in this validation run. Mean Stage 2 CPU latency at L3 was 0.4 ms RAW, 9.6 ms original classical, 12.5 ms QR-aware classical, and 24.8–37.1 ms for the fine-tuned U-Nets. Stage 2 MER is intentionally blank because the paired Clean-photo matrices are not independent, pixel-aligned ground truth.

## H1 and H2

H1 provides limited, severity-dependent evidence. Across all L2 validation images, U-NET ALL decoded 52.7% versus 27.3% for U-NET SINGLE, a paired +25.5 percentage-point difference (95% identity-bootstrap CI +16 to +40; 19 recoveries, 5 regressions). Within L2, the difference was +72.7 points for DS (9.1% to 81.8%; CI +40 to +100) and +45.5 points for SG (18.2% to 63.6%; CI +15 to +75). At L3, the ALL-versus-SINGLE difference was +5.5 points (CI −4 to +16), and at L4 it was +3.6 points (CI 0 to +10); both models decoded 0/11 examples for each of S, DS, SG, and DSG at L3 and L4. Combined-artifact training helped some moderate-severity cases but did not show broad generalization under stronger shadows.

H2 is not supported by exact decode performance. At L3, standard and structure-weighted U-NET ALL each decoded 5/55, with 4 paired recoveries and 4 regressions between them (net zero; 95% CI −11 to +10 points). At L4 both decoded 2/55 (1 recovery and 1 regression; net zero; CI −6 to +6 points). On the Stage 1 generated set, both reached 99.93% exact DR; the weighted model's MER was numerically lower, but errors were already almost absent and fixed-pattern errors were zero in both. The Stage 2 MER endpoint cannot be evaluated independently with the available matrix references.

## Interpretation and limitations

The best validation severity for measuring recovery is L3. It retains a 30.9% RAW success rate while classical restoration recovers 28–30 of the 38 RAW failures without a measured regression. L4 is useful as an extreme stress condition but has little RAW headroom. The synthetic results do not establish performance on new physical photographs. If a later external-validity study is desired, include varied S, DS, SG, and DSG captures near factor 0.30, with factor 0.50 controls and factor 0.20 stress cases; no photographs were requested or collected here.

Two exploratory runs are kept under `diagnostics/` but excluded from final claims: an initial Stage 2 target matrix used a regenerated QR version that did not match the photographed code, and a corrected-target run used unmatched treatment seeds. The final run used the audited Clean-photo reference matrices and matched seeds, initializations, sampling order, augmentation RNG, architecture, and training budgets.

The previous Artifact B package, prior report, prior model checkpoints, manifests, and results are retained unchanged under `prior_project/`. Historical real-photo test outcomes in that prior analysis were already examined, so they are not independent evidence and were not re-evaluated here.

The carried-forward pilot report recorded L3 RAW 18/55 and QR-aware classical 51/55, with 52/55 for RAW-first fallback. The present controlled U-Net comparison uses a single input-detected, perspective-normalized 256×256 ROI policy for every method; the earlier pilot used its fixed-frame image path. The current L3 values (RAW 17/55; QR-aware 45/55) therefore belong to the new common-ROI rerun and should not be read as a paired change from the older fixed-frame result.

## Proposal status

**Complete for this phase:** Stage 1 synthetic pretraining; Stage 2 validation-only fine-tuning; H1/H2 comparisons with matched model/budget/validation procedures; exact DR, paired recovery/regression, false-decode and latency summaries; valid Stage 1 MER and region-error summaries; actual output panels for recovery, unchanged success, regression, and failure; reproducibility source and logs.

**Pending:** final held-out synthetic test (intentionally not run); an independent Stage 2 module-matrix annotation and trustworthy pixel alignment for MER; the rest of Proposal Table 1 systems outside the Artifact B U-Net and included classical comparisons; external-validity assessment on new physical captures if required. No Stage 3 work was performed.
