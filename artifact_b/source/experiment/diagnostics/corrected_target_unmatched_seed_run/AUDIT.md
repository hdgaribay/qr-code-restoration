# Corrected-target exploratory run excluded from matched H1/H2 claims

This run used verified paired-Clean module matrices and correct Stage 2 supervision. It is preserved as an exploratory result. The three model branches used different initialization and batch-sampling seeds, so it does not meet the final paired-randomness control. The final matched run resets the same seed before each branch, uses identical batch index/augmentation sequences, and trains each branch for the same step budget. The unmatched-seed run is excluded from the final H1/H2 conclusions.
