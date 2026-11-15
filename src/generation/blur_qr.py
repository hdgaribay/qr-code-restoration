from pathlib import Path

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

from src.generation.defocus import disk_psf


input_dir = Path("images") / "clean"
output_dir = Path("images") / "blurred"
output_dir.mkdir(parents=True, exist_ok=True)

image_paths = sorted(input_dir.glob("qr_*/qr_[0-9][0-9][0-9][0-9].png"))
radii = np.arange(7, 13)

if not image_paths:
    raise FileNotFoundError(
        f"No clean QR images found in {input_dir.resolve()}"
    )

for image_path in image_paths:
    qr_id = image_path.stem

    image = np.asarray(
        Image.open(image_path).convert("L"),
        dtype=np.float64
    ) / 255.0

    for radius in radii:
        psf = disk_psf(radius)

        blurred = convolve2d(
            image,
            psf,
            mode="same",
            boundary="fill",
            fillvalue=1.0
        )

        blurred_uint8 = (
            np.clip(blurred, 0.0, 1.0) * 255
        ).round().astype(np.uint8)

        radius_dir = output_dir / qr_id / f"r{radius}"
        radius_dir.mkdir(parents=True, exist_ok=True)
        output_path = radius_dir / f"{qr_id}_blur_r{radius}.png"
        Image.fromarray(blurred_uint8).save(output_path)

        print(f"Saved: {output_path}")
