import csv
from pathlib import Path

import cv2


input_dir = Path("images") / "rl_deconv"
metadata_path = Path("images") / "clean" / "metadata.csv"

# Map each QR identity to its correct payload.
with metadata_path.open(newline="", encoding="utf-8") as file:
    expected_payloads = {
        row["qr_id"]: row["payload"]
        for row in csv.DictReader(file)
    }

# Search only the iteration folders.
image_paths = sorted(
    input_dir.glob("iter_*/qr_*_blur_r*_rl.png")
)

decoder = cv2.QRCodeDetector()

print(f"Images found: {len(image_paths)}")

if not image_paths:
    print(f"No matching images found in: {input_dir.resolve()}")

for image_path in image_paths:
    iteration_folder = image_path.parent.name

    # Example: qr_0002_blur_r9_rl.png -> qr_0002
    qr_id = image_path.stem.split("_blur_r", 1)[0]

    if qr_id not in expected_payloads:
        raise ValueError(f"No expected payload recorded for {qr_id}")

    expected_payload = expected_payloads[qr_id]

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(
            f"{iteration_folder} | {image_path.name}: "
            "image could not be loaded"
        )
        continue

    decoded_payload, points, straight_qr = decoder.detectAndDecode(image)

    if decoded_payload == expected_payload:
        outcome = "correct"
    elif decoded_payload == "":
        outcome = "failed"
    else:
        outcome = "wrong payload"

    print(
        f"{iteration_folder} | {image_path.name}: {outcome} "
        f"| expected={expected_payload!r} "
        f"| decoded={decoded_payload!r}"
    )