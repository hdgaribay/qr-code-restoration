import csv,sys,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from audit_and_evaluate import readcsv,writecsv,geometry_row,bootstrap_ids
EXP=ROOT/'source/experiment'
# Extend geometric alignment audit to the audited training pairs (no test data opened).
with np.load(EXP/'data/stage2_pairs/train.npz') as z: tx=z['x']
tr=[r for r in readcsv(EXP/'data/stage2_pairs/stage2_pair_manifest.csv') if r['split']=='train' and r.get('pair_index')]
geos=[]
for r in tr:
 im=tx[int(r['pair_index'])]
 g={'split':'training','identity_id':r['identity_id'],'image_number':r['image_number'],'condition':r['condition'],'severity_level':r['severity_level'],'module_count':r['module_count'],'geometry_source':r['geometry_source'],'geometry_ok_at_cache':r['geometry_ok'],'pair_index':r['pair_index']}
 g.update(geometry_row(im,int(r['module_count'])));geos.append(g)
valrows=[r for r in readcsv(EXP/'data/stage2_pairs/stage2_pair_manifest.csv') if r['split']=='validation']
with np.load(EXP/'data/stage2_pairs/validation.npz') as z:vx=z['x']
valgeo=[]
for r in valrows:
 im=vx[int(r['pair_index'])] if r.get('pair_index') else None
 g={'split':'validation','identity_id':r['identity_id'],'image_number':r['image_number'],'condition':r['condition'],'severity_level':r['severity_level'],'module_count':r['module_count'],'geometry_source':r['geometry_source'],'geometry_ok_at_cache':r['geometry_ok'],'pair_index':r.get('pair_index','')}
 if im is not None:g.update(geometry_row(im,int(r['module_count'])))
 else:g.update({'roi_detected':False,'expected_module_pitch_px':'','corner_rms_px':'','corner_rms_module':'','edge_scale_error_pct':'','residual_rotation_deg':''})
 valgeo.append(g)
writecsv(ROOT/'audit/geometry_alignment.csv',geos+valgeo)
# Paired checkpoint comparison on same 220 images, clustered by identity.
preds=readcsv(ROOT/'results/reproduced_stage2_predictions.csv');out=[]
pairs=[('STAGE1_SINGLE_STANDARD','UNET_SINGLE_STANDARD'),('STAGE1_ALL_STANDARD','UNET_ALL_STANDARD'),('STAGE1_ALL_STRUCTURE_WEIGHTED','UNET_ALL_STRUCTURE_WEIGHTED')]
for a,b in pairs:
 for lev in range(1,5):
  for cond in ['ALL','S','DS','SG','DSG']:
   aa={(r['identity_id'],r['image_number']):int(r['exact_decode']) for r in preds if r['method']==a and int(r['severity_level'])==lev and (cond=='ALL' or r['condition']==cond)}
   bb={(r['identity_id'],r['image_number']):int(r['exact_decode']) for r in preds if r['method']==b and int(r['severity_level'])==lev and (cond=='ALL' or r['condition']==cond)}
   keys=sorted(set(aa)&set(bb)); rr=[{'identity_id':k[0],'delta':bb[k]-aa[k]} for k in keys]
   ids=sorted(set(x['identity_id'] for x in rr));means=np.array([np.mean([x['delta'] for x in rr if x['identity_id']==i]) for i in ids]);rng=np.random.default_rng(7711+lev+len(cond));boot=np.array([means[rng.integers(0,len(means),len(means))].mean() for _ in range(5000)])
   ci=np.quantile(boot,[.025,.975])
   out.append({'starting_checkpoint':a,'fine_tuned_checkpoint':b,'severity_level':lev,'condition':cond,'n_images':len(keys),'n_identities':len(ids),'stage1_exact':sum(aa.values()),'stage2_exact':sum(bb.values()),'recoveries_from_stage1':sum(x['delta']>0 for x in rr),'regressions_from_stage1':sum(x['delta']<0 for x in rr),'net_delta':sum(x['delta'] for x in rr),'paired_delta_ci_low':ci[0],'paired_delta_ci_high':ci[1]})
writecsv(ROOT/'results/pretrain_vs_finetune_paired.csv',out)
# Compare reproduced counts and false-decode counts to the historical aggregate results.
hist=readcsv(EXP/'logs/method_comparison_validation.csv');oldmap={(r['method'],int(r['severity_level']),r['condition']):r for r in hist}
new=readcsv(ROOT/'results/reproduced_stage2_comparison.csv');diff=[]
for r in new:
 k=(r['method'],int(r['severity_level']),r['condition']);p=oldmap.get(k)
 if p:
  diff.append({'method':r['method'],'level':r['severity_level'],'condition':r['condition'],'recomputed_n':r['n_images'],'historical_n':p['n_images'],'recomputed_exact':r['exact_decodes'],'historical_exact':p['exact_decodes'],'exact_count_match':int(int(r['exact_decodes'])==int(p['exact_decodes'])),'recomputed_false_decodes':r['false_decodes'],'historical_false_decodes':p['false_decodes'],'false_decode_count_match':int(int(r['false_decodes'])==int(p['false_decodes']))})
writecsv(ROOT/'audit/historical_reproduction_check.csv',diff)
# Version and severity evidence: only generation manifest and train/validation arrays.
rows=[]
for group,rr in [('Stage1 generated validation', [r for r in readcsv(EXP/'data/stage1_pairs_manifest.csv') if r['split']=='validation']),('Stage2 audited training',tr),('Stage2 validation',[r for r in readcsv(EXP/'data/stage2_pairs/stage2_pair_manifest.csv') if r['split']=='validation'])]:
 versions=Counter(r['module_count'] for r in rr if r.get('module_count'))
 for n,c in sorted(versions.items(),key=lambda x:int(x[0])):
  rows.append({'population':group,'module_count':n,'qr_version':(int(n)-17)//4,'n_records':c,'module_pitch_in_256_roi_px':256/(int(n)+8)})
 factors=Counter(r.get('factor',r.get('shadow_factor','')) for r in rr if r.get('factor',r.get('shadow_factor','')))
 for fac,c in sorted(factors.items()):
  rows.append({'population':group,'shadow_factor':fac,'n_records':c})
writecsv(ROOT/'audit/version_and_shadow_distribution.csv',rows)
print('training geometry records',len(geos),'validation records',len(valgeo),'detected training',sum(r['roi_detected']=='True' for r in geos),'detected val',sum(r['roi_detected']=='True' for r in valgeo),'historical matches',sum(int(r['exact_count_match']) for r in diff),'/',len(diff))
