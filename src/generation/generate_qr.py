import csv
from pathlib import Path

import numpy as np
import qrcode

output_dir = Path("images") / "clean"
output_dir.mkdir(parents=True, exist_ok=True)

metadata = []

for number in range(1, 21):
    qr_id = f"qr_{number:04d}"
    payload = f"ECE253-QR-{number:04d}"
    identity_dir = output_dir / qr_id
    identity_dir.mkdir(parents=True, exist_ok=True)

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )

    qr.add_data(payload)
    qr.make(fit=False)  # Keep version 1 fixed.

    img = qr.make_image(fill_color="black", back_color="white")
    image_path = identity_dir / f"{qr_id}.png"
    img.save(image_path)

    # Includes the border; True means black.
    modules = np.asarray(qr.get_matrix(), dtype=bool)
    module_path = identity_dir / f"{qr_id}_modules.npy"
    np.save(module_path, modules)

    metadata.append({
        "qr_id": qr_id,
        "payload": payload,
        "version": qr.version,
        "error_correction": "L",
        "pixels_per_module": 10,
        "border_modules": 4,
        "image_path": image_path.as_posix(),
        "module_path": module_path.as_posix(),
    })

    print(f"Saved {qr_id}: {payload}")

with (output_dir / "metadata.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=metadata[0].keys())
    writer.writeheader()
    writer.writerows(metadata)
