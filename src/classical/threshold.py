from pathlib import Path

import cv2
import numpy as np


input_dir = Path("images") / "glared"
output_dir = Path("images") / "glare_restored"

# --- Tunables ---
WHITE_TOL = 0   # white only if every channel >= 255 - WHITE_TOL.
                # Raise to 1-3 if antialiasing or rounding makes modules look fat.


def restore(img):
    # A white module stays exactly (255, 255, 255) under glare, while a dark
    # module is lifted but has at least one channel below 255.
    # So "any channel below 255" means dark.
    min_channel = img.min(axis=2)
    return np.where(min_channel >= 255 - WHITE_TOL, 255, 0).astype(np.uint8)


image_paths = sorted(input_dir.glob("qr_*/*/v*/qr_*_glared_*_v*_s*.png"))

if not image_paths:
    raise FileNotFoundError(f"No glared images found in {input_dir.resolve()}")

print(f"Images found: {len(image_paths)}")

for image_path in image_paths:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read {image_path}")
        continue

    restored = restore(img)

    out_path = output_dir / image_path.relative_to(input_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), restored)
    print(f"Saved: {out_path}")