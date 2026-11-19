"""Write a SHA-256 manifest for the deliverable, excluding this manifest itself."""
from pathlib import Path
import csv, hashlib
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'audit' / 'SHA256SUMS.csv'
rows = []
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p == OUT:
        continue
    rel = p.relative_to(ROOT)
    if 'qa_docx' in rel.parts or '__pycache__' in rel.parts or p.suffix == '.pyc':
        continue
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    rows.append((str(rel), p.stat().st_size, h.hexdigest()))
with OUT.open('w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['path','bytes','sha256'])
    w.writerows(rows)
print(f'{OUT}: {len(rows)} files')
