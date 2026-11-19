import csv
from collections import defaultdict
from pathlib import Path

import cv2


input_dir = Path("images") / "glare_restored"
metadata_path = Path("images") / "clean" / "metadata.csv"

with metadata_path.open(newline="", encoding="utf-8") as file:
    expected_payloads = {
        row["qr_id"]: row["payload"]
        for row in csv.DictReader(file)
    }

# e.g. images/glare_restored/qr_0001/ellipse/v1/qr_0001_glared_ellipse_v1_s1.5.png
image_paths = sorted(
    input_dir.glob("qr_*/*/v*/qr_*_glared_*_v*_s*.png")
)

decoder = cv2.QRCodeDetector()

print(f"Images found: {len(image_paths)}")

if not image_paths:
    print(f"No matching images found in: {input_dir.resolve()}")

# key -> [correct, total]
by_strength = defaultdict(lambda: [0, 0])  # (shape, strength)
by_variant = defaultdict(lambda: [0, 0])   # (shape, variant)
overall = [0, 0]
failed_files = []

for image_path in image_paths:
    # qr_0001_glared_ellipse_v1_s1.5 -> qr_0001, ellipse, v1, s1.5
    qr_id, rest = image_path.stem.split("_glared_", 1)
    shape, variant, strength = rest.split("_")

    if qr_id not in expected_payloads:
        raise ValueError(f"No expected payload recorded for {qr_id}")

    expected_payload = expected_payloads[qr_id]

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"{image_path.name}: image could not be loaded")
        continue

    decoded_payload, _, _ = decoder.detectAndDecode(image)

    correct = decoded_payload == expected_payload
    outcome = "correct" if correct else "failed"

    for tally, key in ((by_strength, (shape, strength)),
                       (by_variant, (shape, variant))):
        tally[key][1] += 1
        if correct:
            tally[key][0] += 1

    overall[1] += 1
    if correct:
        overall[0] += 1
    else:
        failed_files.append(image_path)

    print(
        f"{image_path.name}: {outcome} "
        f"| expected={expected_payload!r} | decoded={decoded_payload!r}"
    )


def print_table(title, tally):
    print(f"\n{title}")
    for key, (ok, total) in sorted(tally.items()):
        label = " ".join(key)
        print(f"  {label:<16} {ok}/{total} correct ({100 * ok / total:.1f}%)")


print_table("Decode rate by shape and strength:", by_strength)
print_table("Decode rate by shape and variant:", by_variant)

if overall[1]:
    print(f"\nOverall: {overall[0]}/{overall[1]} correct "
          f"({100 * overall[0] / overall[1]:.1f}%)")

if failed_files:
    print(f"\nFailed images ({len(failed_files)}):")
    for p in failed_files:
        print(f"  {p}")