import csv
from pathlib import Path

import cv2
import numpy as np


input_dir = Path("images") / "clean"
output_dir = Path("images") / "glared"
output_dir.mkdir(parents=True, exist_ok=True)

# Ellipse strength is a gain (>1 makes a clipped, fully white core).
# Rectangle strength is peak opacity, because a rectangle is already 1
# inside, so a gain above 1 would change nothing.
ELLIPSE_STRENGTHS = [1.1, 1.15, 1.2]
RECT_STRENGTHS = [1.0, 1.05, 1.1]

# Roundness = minor axis / major axis (1.0 is a circle). One random draw
# per bin so the 3 ellipses cover a spread from elongated to nearly round.
ROUNDNESS_BINS = [(0.25, 0.45), (0.45, 0.70), (0.70, 0.95)]
RECT_AREA_BINS = [(0.10, 0.20), (0.20, 0.35), (0.35, 0.55)]
RECT_ASPECT_RANGE = (1 / 3, 3.0)   # width/height, so boxes can be tall or wide

rng = np.random.default_rng(42)  # fixed seed so results are reproducible


def random_center(h, w):
    return (
        int(rng.integers(w // 4, 3 * w // 4)),
        int(rng.integers(h // 4, 3 * h // 4)),
    )


def ellipse_mask(h, w, center, major_sigma, roundness, angle_deg):
    """Rotated anisotropic Gaussian with minor = major * roundness."""
    minor_sigma = major_sigma * roundness
    Y, X = np.ogrid[:h, :w]
    t = np.deg2rad(angle_deg)
    dx, dy = X - center[0], Y - center[1]
    u = dx * np.cos(t) + dy * np.sin(t)
    v = -dx * np.sin(t) + dy * np.cos(t)
    return np.exp(-(u**2 / (2 * major_sigma**2) + v**2 / (2 * minor_sigma**2)))


def rect_mask(h, w, center, box_w, box_h, angle_deg):
    """Rotated rectangle with soft edges (window / softbox reflection)."""
    mask = np.zeros((h, w), np.float32)
    box = cv2.boxPoints((center, (box_w, box_h), angle_deg))
    cv2.fillPoly(mask, [box.astype(np.int32)], 1.0)
    blur_sigma = max(1.0, 0.1 * min(box_w, box_h))
    return cv2.GaussianBlur(mask, (0, 0), blur_sigma)


def apply_glare(img, mask, strength, gain=True):
    """Tint the mask and screen-blend it onto the image."""
    mask = mask * strength if gain else mask * strength
    
    glare_layer = np.zeros_like(img)
    glare_layer[:, :, 0] = mask * 0.85  # Blue
    glare_layer[:, :, 1] = mask * 0.95  # Green
    glare_layer[:, :, 2] = mask * 1.00  # Red

    result = 1.0 - (1.0 - img) * (1.0 - glare_layer)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)


image_paths = sorted(input_dir.glob("qr_*/qr_[0-9][0-9][0-9][0-9].png"))

if not image_paths:
    raise FileNotFoundError(
        f"No clean QR images found in {input_dir.resolve()}"
    )

log_path = output_dir / "glare_log.csv"
with log_path.open("w", newline="", encoding="utf-8") as log_file:
    log = csv.writer(log_file)
    log.writerow([
        "qr_id", "shape", "variant", "strength",
        "center_x", "center_y", "angle", "param_a", "param_b",
    ])

    for image_path in image_paths:
        qr_id = image_path.stem

        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Could not read {image_path}")
        img = img.astype(np.float32) / 255.0
        h, w, _ = img.shape

          # ---------- Ellipses: 3 random roundness values ----------
        for v, (lo, hi) in enumerate(ROUNDNESS_BINS, start=1):
            roundness = float(rng.uniform(lo, hi))
            major_sigma = w * float(rng.uniform(0.20, 0.35))

            variant_dir = output_dir / qr_id / "ellipse" / f"v{v}"
            variant_dir.mkdir(parents=True, exist_ok=True)

            for strength in ELLIPSE_STRENGTHS:
                center = random_center(h, w)           # new center per image
                angle = float(rng.uniform(0, 180))     # new angle per image
                mask = ellipse_mask(h, w, center, major_sigma, roundness, angle)

                glared = apply_glare(img, mask, strength)
                out = variant_dir / f"{qr_id}_glared_ellipse_v{v}_s{strength}.png"
                cv2.imwrite(str(out), glared)
                log.writerow([qr_id, "ellipse", f"v{v}", strength,
                              center[0], center[1], round(angle, 1),
                              round(major_sigma, 1), round(roundness, 3)])
                print(f"Saved: {out}")

         # ---------- Rectangles: 3 random area coverages ----------
        for v, (lo, hi) in enumerate(RECT_AREA_BINS, start=1):
            area_frac = float(rng.uniform(lo, hi))
            aspect = float(np.exp(rng.uniform(np.log(RECT_ASPECT_RANGE[0]),
                                              np.log(RECT_ASPECT_RANGE[1]))))
            area = area_frac * w * h
            box_w = float(np.sqrt(area * aspect))
            box_h = float(np.sqrt(area / aspect))

            variant_dir = output_dir / qr_id / "rect" / f"v{v}"
            variant_dir.mkdir(parents=True, exist_ok=True)

            for strength in RECT_STRENGTHS:
                center = random_center(h, w)           # new center per image
                angle = float(rng.uniform(0, 180))     # new angle per image
                mask = rect_mask(h, w, center, box_w, box_h, angle)

                glared = apply_glare(img, mask, strength)
                out = variant_dir / f"{qr_id}_glared_rect_v{v}_s{strength}.png"
                cv2.imwrite(str(out), glared)
                log.writerow([qr_id, "rect", f"v{v}", strength,
                              center[0], center[1], round(angle, 1),
                              round(box_w, 1), round(box_h, 1)])
                print(f"Saved: {out}")