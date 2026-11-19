from pathlib import Path

import cv2
import numpy as np


input_dir = Path("images") / "clean"
output_dir = Path("images") / "glared"
output_dir.mkdir(parents=True, exist_ok=True)

radius_pcts = [20, 30, 40]        # bigger glare spots
intensities = [1.5, 2.5, 4.0]     # now a gain: >1 makes a clipped, fully white core

rng = np.random.default_rng(42)  # fixed seed so results are reproducible


def add_synthetic_glare(img, center, radius, intensity):
    h, w, _ = img.shape

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    glare_mask = np.exp(-(dist_from_center**2) / (2 * (radius**2)))
    glare_mask = np.clip(glare_mask * intensity, 0, 1)

    glare_layer = np.zeros_like(img)
    glare_layer[:, :, 0] = glare_mask * 0.85  # Blue
    glare_layer[:, :, 1] = glare_mask * 0.95  # Green
    glare_layer[:, :, 2] = glare_mask * 1.00  # Red

    # Screen blend
    result = 1.0 - (1.0 - img) * (1.0 - glare_layer)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


image_paths = sorted(input_dir.glob("qr_*/qr_[0-9][0-9][0-9][0-9].png"))

if not image_paths:
    raise FileNotFoundError(
        f"No clean QR images found in {input_dir.resolve()}"
    )

for image_path in image_paths:
    qr_id = image_path.stem

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    img = img.astype(np.float32) / 255.0
    h, w, _ = img.shape

    for r_pct in radius_pcts:
        radius = int(w * r_pct / 100)

        radius_dir = output_dir / qr_id / f"r{r_pct}"
        radius_dir.mkdir(parents=True, exist_ok=True)

        for intensity in intensities:
            # Random glare center in the middle half of the image
            center = (
                int(rng.integers(w // 4, 3 * w // 4)),
                int(rng.integers(h // 4, 3 * h // 4)),
            )

            glared = add_synthetic_glare(img, center, radius, intensity)

            output_path = radius_dir / f"{qr_id}_glared_r{r_pct}_s{intensity}.png"
            cv2.imwrite(str(output_path), glared)
            print(f"Saved: {output_path}  (center={center})")