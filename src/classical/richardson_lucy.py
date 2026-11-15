from pathlib import Path

import numpy as np
from PIL import Image
from skimage.restoration import richardson_lucy

from src.generation.defocus import disk_psf


iterations = [5,10,20,40]  

input_dir = Path("images") / "blurred"
image_paths = sorted(input_dir.glob("qr_00**_blur_r*.png"))

print("Searching in:", input_dir.resolve())
print("Images found:", len(image_paths))
for num_iterations in iterations:
    for image_path in image_paths:
        output_dir = Path("images") / "rl_deconv" / f"iter_{num_iterations}"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Example: qr_0001_blur_r9.png -> radius 9
        radius = int(image_path.stem.rsplit("_r", 1)[1])

        blurred = np.asarray(
            Image.open(image_path).convert("L"),
            dtype=np.float64
        ) / 255.0

        psf = disk_psf(radius)

        # Add white padding to reduce artificial image-border effects.
        padding = 2 * radius
        padded = np.pad(
            blurred,
            pad_width=padding,
            mode="constant",
            constant_values=1.0
        )

        restored_padded = richardson_lucy(
            padded,
            psf,
            num_iter=num_iterations,
            clip=False
        )

        # Remove the added padding.
        restored = restored_padded[
            padding:-padding,
            padding:-padding
        ]

        restored_uint8 = (
            np.clip(restored, 0.0, 1.0) * 255
        ).round().astype(np.uint8)

        output_path = output_dir / f"{image_path.stem}_rl.png"
        Image.fromarray(restored_uint8).save(output_path)

        print(f"Saved: {output_path}")