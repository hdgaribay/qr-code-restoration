"""Add timed classical preprocessing to the already completed digital decode rows."""
from pathlib import Path
import csv,time
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source/prior_project/ArtifactB_Next_Research/src'))
from methods import classical_orig,qr_aware
with (ROOT/'results/known_geometry_validation_predictions.csv').open(newline='',encoding='utf-8-sig') as f:
 rows=list(csv.DictReader(f));fields=list(rows[0])
with np.load(ROOT/'data/known_geometry_validation.npz') as z:x=z['x']
with (ROOT/'data/known_geometry_validation_manifest.csv').open(newline='',encoding='utf-8-sig') as f:manifest=list(csv.DictReader(f))
proc={'CLASSICAL_ORIG':[],'QR_AWARE':[]}
for im in x:
 t=time.perf_counter();classical_orig(im,kernel=81,sigma=9,window=51,k=.18);proc['CLASSICAL_ORIG'].append((time.perf_counter()-t)*1000)
 t=time.perf_counter();qr_aware(im,window_factor=5,k=.18,use_anchor=True,use_guided=True);proc['QR_AWARE'].append((time.perf_counter()-t)*1000)
for i in range(len(x)):
 for m in proc:
  pair=manifest[i]['pair_id'];row=next(r for r in rows if r['pair_id']==pair and r['method']==m)
  row['latency_ms']=str(float(row['latency_ms'])+proc[m][i])
with (ROOT/'results/known_geometry_validation_predictions.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
# Refresh only the latency summaries; all decoding, MER, CI, recovery and regression values remain unchanged.
with (ROOT/'results/known_geometry_validation_comparison.csv').open(newline='',encoding='utf-8-sig') as f:
 summaries=list(csv.DictReader(f));sfields=list(summaries[0])
for s in summaries:
 vals=[float(r['latency_ms']) for r in rows if r['method']==s['method'] and int(r['severity_level'])==int(s['severity_level']) and (s['condition']=='ALL' or r['condition']==s['condition'])]
 if vals:s['mean_latency_ms']=str(float(np.mean(vals)))
with (ROOT/'results/known_geometry_validation_comparison.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=sfields);w.writeheader();w.writerows(summaries)
print({m:{'mean_ms':float(np.mean(v)),'median_ms':float(np.median(v))} for m,v in proc.items()})
