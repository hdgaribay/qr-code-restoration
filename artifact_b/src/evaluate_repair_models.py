#!/usr/bin/env python3
"""Evaluate repaired models on the existing Stage 2 validation and aligned digital validation.
No original-photo archive, test manifest, payload oracle, or test identity is read.
"""
from pathlib import Path
import csv,hashlib,json,sys,time
from collections import defaultdict
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'source/experiment';sys.path.insert(0,str(ROOT/'source/prior_project/ArtifactB_Next_Research/src'));sys.path.insert(0,str(EXP/'src'))
from unet_model import CompactUNet
from methods import read_exact,classical_orig,qr_aware
import stage1_stage2 as legacy
from audit_and_evaluate import readcsv,writecsv,bootstrap_ids,sha_file

def model(path):
 ck=torch.load(path,map_location='cpu',weights_only=False);m=CompactUNet();m.load_state_dict(ck['state_dict']);m.eval();return m,ck
def htxt(s):return hashlib.sha256(s.encode()).hexdigest()
def warp_points(H,pts):return cv2.perspectiveTransform(np.asarray(pts,np.float32).reshape(1,-1,2),H)[0]
def mer_regions(img,mat,H):
 n=len(mat);p=256.0/(n+8)
 # Nine interior samples per module, evaluated as one vectorized gather.
 yy,xx=np.mgrid[0:n,0:n];ox,oy=np.meshgrid((-.16,0,.16),(-.16,0,.16))
 px=(xx[...,None]+4.5+ox.ravel())*p;py=(yy[...,None]+4.5+oy.ravel())*p
 points=np.stack([px,py],axis=-1).reshape(-1,2)
 pts=warp_points(H,points).reshape(n,n,9,2)
 xs=np.clip(np.rint(pts[...,0]).astype(int),0,255);ys=np.clip(np.rint(pts[...,1]).astype(int),0,255)
 vals=np.mean(img[ys,xs]<128,axis=2)>=.5
 errs=vals!=mat.astype(bool)
 masks=legacy.module_region_masks(n)
 ans={'module_errors':int(errs.sum()),'module_count':int(errs.size),'mer':float(errs.mean()),'fixed_pattern_errors':int(errs[masks['fixed']].sum()),'fixed_pattern_modules':int(masks['fixed'].sum())}
 for name in ('finder','timing','alignment'):
  mask=masks[name];ans[name+'_errors']=int(errs[mask].sum());ans[name+'_modules']=int(mask.sum());ans[name+'_mer']=float(errs[mask].mean()) if mask.any() else ''
 return ans
def identity_boot_ci(rows,key,seed=7781,B=5000):
 by=defaultdict(list)
 for r in rows:by[r['identity_id']].append(float(r[key]))
 ids=sorted(by);v=np.array([np.mean(by[i]) for i in ids]);rng=np.random.default_rng(seed);boot=np.array([v[rng.integers(0,len(v),len(v))].mean() for _ in range(B)])
 return np.quantile(boot,[.025,.975]).tolist()
def write_summary(rows,split):
 sums=[]
 for m in sorted({r['method'] for r in rows}):
  for lev in sorted({int(r['severity_level']) for r in rows}):
   for cond in ['ALL','S','DS','SG','DSG']:
    rr=[r for r in rows if r['method']==m and int(r['severity_level'])==lev and (cond=='ALL' or r['condition']==cond)]
    if not rr:continue
    dci=identity_boot_ci(rr,'exact_decode',7781+lev+len(cond));nci=identity_boot_ci(rr,'net_improvement',8791+lev+len(cond));merrows=[r for r in rr if int(r.get('mer_valid',0) or 0)==1]
    sums.append({'split':split,'method':m,'severity_level':lev,'condition':cond,'n_images':len(rr),'n_identities':len({r['identity_id'] for r in rr}),
      'exact_decodes':sum(int(r['exact_decode']) for r in rr),'exact_decode_rate':np.mean([int(r['exact_decode']) for r in rr]),'exact_ci_identity_low':dci[0],'exact_ci_identity_high':dci[1],
      'recoveries':sum(int(r['recovery']) for r in rr),'regressions':sum(int(r['regression']) for r in rr),'net_improvement':sum(int(r['net_improvement']) for r in rr),'net_ci_identity_low':nci[0],'net_ci_identity_high':nci[1],
      'false_decodes':sum(int(r['false_decode']) for r in rr),'mean_latency_ms':float(np.mean([float(r['latency_ms']) for r in rr])),
      'valid_mer_images':len(merrows),'module_error_rate':sum(int(r['module_errors']) for r in merrows)/sum(int(r['module_count']) for r in merrows) if merrows else '',
      'fixed_pattern_error_rate':sum(int(r['fixed_pattern_errors']) for r in merrows)/sum(int(r['fixed_pattern_modules']) for r in merrows) if merrows else '',
      'finder_error_rate':sum(int(r['finder_errors']) for r in merrows)/sum(int(r['finder_modules']) for r in merrows) if merrows else '',
      'timing_error_rate':sum(int(r['timing_errors']) for r in merrows)/sum(int(r['timing_modules']) for r in merrows) if merrows else '',
      'alignment_error_rate':sum(int(r['alignment_errors']) for r in merrows)/sum(int(r['alignment_modules']) for r in merrows) if merrows else ''})
 return sums
def main():
 torch.set_num_threads(min(4,torch.get_num_threads()));repaired=[];repair_specs=[]
 for label,name in [('REPAIR_SINGLE','repair_stage2_single_best.pt'),('REPAIR_ALL','repair_stage2_all_best.pt'),('REPAIR_ALL_WEIGHTED','repair_stage2_all_weighted_best.pt')]:
  p=ROOT/'models'/name;m,ck=model(p);repair_specs.append((label,m,p,ck))
 # Existing Stage 2 validation: use stored ROI bytes; seven missing-geometry examples remain failures in 220-image denominator.
 with np.load(EXP/'data/stage2_pairs/validation.npz') as z:x=z['x'];yarr=z['y']
 manifest=[r for r in readcsv(EXP/'data/stage2_pairs/stage2_pair_manifest.csv') if r['split']=='validation']
 old=readcsv(EXP/'logs/validation_predictions.csv');expected={}
 for r in old:
  if r['method']=='RAW':expected[(r['identity_id'],r['image_number'],int(r['severity_level']))]=r['payload_sha256']
 visual_records=[];outx=[];outnames=[];outy=[];outkeys=[]
 for r in manifest:
  key=(r['identity_id'],r['image_number'],int(r['severity_level']));ind=int(r['pair_index']) if r.get('pair_index') else None;im=x[ind] if ind is not None else None;payload=expected[key];rawok=False
  if im is not None:rawtxt=read_exact(im);rawok=bool(rawtxt and htxt(rawtxt)==payload)
  outputs={}
  if im is not None:
   tensor=torch.from_numpy((im.astype(np.float32)/255.)[None,None])
   for label,m,p,ck in repair_specs:
    t=time.perf_counter()
    with torch.inference_mode():prob=m(tensor).numpy()[0,0]
    out=np.where(prob>=.5,0,255).astype(np.uint8);proc=(time.perf_counter()-t)*1000
    td=time.perf_counter();text=read_exact(out);dt=(time.perf_counter()-td)*1000;exact=bool(text and htxt(text)==payload)
    outputs[label]=out
    visual_records.append({'identity_id':r['identity_id'],'image_number':r['image_number'],'condition':r['condition'],'severity_level':key[2],'method':label,'geometry_ok':bool(r['geometry_ok']),'payload_sha256':payload,'decoded_payload_hash':htxt(text) if text else '',
     'exact_decode':int(exact),'false_decode':int(bool(text and not exact)),'raw_exact':int(rawok),'recovery':int(not rawok and exact),'regression':int(rawok and not exact),'net_improvement':int(exact)-int(rawok),'latency_ms':proc+dt,'mer_valid':0,'module_errors':'','module_count':'','note':'Stage 2 MER withheld: paired-Clean matrix is not independently pixel-aligned.'})
  else:
   for label,_,_,_ in repair_specs:visual_records.append({'identity_id':r['identity_id'],'image_number':r['image_number'],'condition':r['condition'],'severity_level':key[2],'method':label,'geometry_ok':False,'payload_sha256':payload,'decoded_payload_hash':'','exact_decode':0,'false_decode':0,'raw_exact':0,'recovery':0,'regression':0,'net_improvement':0,'latency_ms':0.0,'mer_valid':0,'module_errors':'','module_count':'','note':'invalid ROI retained as failure'})
  outx.append(im if im is not None else np.full((256,256),255,np.uint8));outkeys.append(json.dumps([r['identity_id'],r['image_number'],r['condition'],key[2]]));outy.append(yarr[ind] if ind is not None else np.full((256,256),255,np.uint8))
  for label,out in outputs.items():pass
  # Preserve repaired outputs aligned to the manifest order, for figure selection.
  if im is None:
   for label,_,_,_ in repair_specs:outnames.append(label);outnames.append(label)
  else:
   for label,_,_,_ in repair_specs:outnames.append(label)
 # Reload array outputs in deterministic order to avoid storing 3*220 if geometry invalid; process once more only to save aligned outputs.
 model_outputs=[]
 for label,m,p,ck in repair_specs:
  arr=np.full((len(manifest),256,256),255,np.uint8)
  for j,r in enumerate(manifest):
   if not r.get('pair_index'):continue
   im=x[int(r['pair_index'])];t=torch.from_numpy((im.astype(np.float32)/255.)[None,None])
   with torch.inference_mode():prob=m(t).numpy()[0,0]
   arr[j]=np.where(prob>=.5,0,255).astype(np.uint8)
  model_outputs.append(arr)
 np.savez_compressed(ROOT/'data/stage2_repair_outputs.npz',x=np.stack(outx),y=np.stack(outy),outputs=np.stack(model_outputs),keys=np.array(outkeys,dtype='U120'))
 writecsv(ROOT/'results/stage2_repair_predictions.csv',visual_records)
 oldpred=readcsv(ROOT/'results/reproduced_stage2_predictions.csv');allpred=oldpred+visual_records;writecsv(ROOT/'results/all_stage2_comparison_predictions.csv',allpred)
 stage2sum=write_summary(allpred,'existing_stage2_synthetic_validation')
 writecsv(ROOT/'results/all_stage2_method_comparison.csv',stage2sum)
 # Paired H1/H2 effects for repair models on Stage 2 synthetic validation.
 contrasts=[]
 for hyp,a,b in [('H1_REPAIR_SINGLE_vs_ALL','REPAIR_SINGLE','REPAIR_ALL'),('H2_REPAIR_ALL_standard_vs_weighted','REPAIR_ALL','REPAIR_ALL_WEIGHTED')]:
  for lev in range(1,5):
   for cond in ['ALL','S','DS','SG','DSG']:
    aa={(r['identity_id'],r['image_number']):int(r['exact_decode']) for r in visual_records if r['method']==a and int(r['severity_level'])==lev and (cond=='ALL' or r['condition']==cond)}
    bb={(r['identity_id'],r['image_number']):int(r['exact_decode']) for r in visual_records if r['method']==b and int(r['severity_level'])==lev and (cond=='ALL' or r['condition']==cond)}
    keys=sorted(set(aa)&set(bb));paired=[{'identity_id':k[0],'delta':bb[k]-aa[k]} for k in keys];ids=sorted({r['identity_id'] for r in paired});v=np.array([np.mean([r['delta'] for r in paired if r['identity_id']==i]) for i in ids]);rng=np.random.default_rng(8827+lev+len(cond));bt=np.array([v[rng.integers(0,len(v),len(v))].mean() for _ in range(5000)]);lo,hi=np.quantile(bt,[.025,.975])
    contrasts.append({'dataset':'existing Stage2 validation','hypothesis':hyp,'baseline':a,'comparison':b,'severity_level':lev,'condition':cond,'n_images':len(keys),'n_identities':len(ids),'baseline_exact':sum(aa.values()),'comparison_exact':sum(bb.values()),'recoveries':sum(r['delta']>0 for r in paired),'regressions':sum(r['delta']<0 for r in paired),'net':sum(r['delta'] for r in paired),'paired_ci_low':lo,'paired_ci_high':hi})
 writecsv(ROOT/'results/repair_hypothesis_contrasts_stage2.csv',contrasts)
 # Aligned digital validation with known transform H. This is the checkpoint-selection set, not independent test data.
 d=np.load(ROOT/'data/known_geometry_validation.npz');dx=d['x'];dy=d['y'];dm=d['mask'];dman=readcsv(ROOT/'data/known_geometry_validation_manifest.csv')
 # Ground-truth module matrices are recoverable from the original Stage1 generated-ID validation target arrays, which are independently known.
 with np.load(EXP/'data/stage1_all_validation.npz') as z:s1y=z['y']
 s1m=[r for r in readcsv(EXP/'data/stage1_pairs_manifest.csv') if r['split']=='validation'];mat_by={}
 for i,r in enumerate(s1m):
  n=int(r['module_count']);pitch=256/(n+8);mat=np.zeros((n,n),np.uint8)
  for yy in range(n):
   for xx in range(n):mat[yy,xx]=1 if s1y[i,int(round((yy+4.5)*pitch)),int(round((xx+4.5)*pitch))]<128 else 0
  mat_by[r['synthetic_id']]=mat
 eval_specs=[]
 for label,p in [('STAGE1_ALL','stage1_all_standard_best.pt'),('STAGE1_WEIGHTED','stage1_all_structure_weighted_best.pt'),('STAGE2_ALL','stage2_all_standard_best.pt'),('STAGE2_WEIGHTED','stage2_all_structure_weighted_best.pt')]:
  m,ck=model(EXP/'models'/p);eval_specs.append((label,m))
 for label,m,_,_ in repair_specs:eval_specs.append((label,m))
 dmethods=['RAW','CLASSICAL_ORIG','QR_AWARE']+[x[0] for x in eval_specs];drows=[];torch.set_num_threads(min(4,torch.get_num_threads()))
 # Cache network outputs in batches, while keeping exact decoding identical for every method.
 network_outputs={}
 for label,m in eval_specs:
  outs=[];latper=[]
  for st in range(0,len(dx),16):
   batch=torch.from_numpy(dx[st:st+16].astype(np.float32)/255.)[:,None];t=time.perf_counter()
   with torch.inference_mode():probs=m(batch).numpy()[:,0]
   elapsed=(time.perf_counter()-t)*1000/len(batch);latper.extend([elapsed]*len(batch));outs.extend([np.where(a>=.5,0,255).astype(np.uint8) for a in probs])
  network_outputs[label]=(outs,latper)
 for i,(im,clean,rec) in enumerate(zip(dx,dy,dman)):
  ident=rec['synthetic_id'];matrix=mat_by[ident];H=np.asarray(json.loads(rec['transform_matrix']),np.float32);payloadhash=rec['payload_sha256'];rawtxt=read_exact(im);rawok=bool(rawtxt and htxt(rawtxt)==payloadhash);cache={}
  cache['RAW']=(im,0.0);cache['CLASSICAL_ORIG']=(classical_orig(im,kernel=81,sigma=9,window=51,k=.18),0.0);cache['QR_AWARE']=(qr_aware(im,window_factor=5,k=.18,use_anchor=True,use_guided=True)[0],0.0)
  for label,(outs,lats) in network_outputs.items():cache[label]=(outs[i],lats[i])
  for method,(out,proc) in cache.items():
   t=time.perf_counter();decoded=read_exact(out);decms=(time.perf_counter()-t)*1000;exact=bool(decoded and htxt(decoded)==payloadhash);mer=mer_regions(out,matrix,H)
   drows.append({'identity_id':ident,'pair_id':rec['pair_id'],'condition':rec['condition'],'severity_level':int(rec['severity_level']),'method':method,'exact_decode':int(exact),'false_decode':int(bool(decoded and not exact)),'raw_exact':int(rawok),'recovery':int(not rawok and exact),'regression':int(rawok and not exact),'net_improvement':int(exact)-int(rawok),'latency_ms':proc+decms,'mer_valid':1,**mer,'shadow_factor':rec['illumination_factor'],'shadow_coverage_fraction':rec['shadow_coverage_fraction'],'geometry_source':'known digital shared homography'})
 writecsv(ROOT/'results/known_geometry_validation_predictions.csv',drows)
 dsum=write_summary(drows,'known_geometry_validation_selection_set');writecsv(ROOT/'results/known_geometry_validation_comparison.csv',dsum)
 # Save checkpoint hashes and the complete checkpoint run manifest.
 ckrows=[]
 for label,p in [('REPAIR_SINGLE','repair_stage2_single_best.pt'),('REPAIR_ALL','repair_stage2_all_best.pt'),('REPAIR_ALL_WEIGHTED','repair_stage2_all_weighted_best.pt')]:
  path=ROOT/'models'/p;ck=torch.load(path,map_location='cpu',weights_only=False)
  log=readcsv(ROOT/'results'/f'{ck["model"]}_training_log.csv')
  ckrows.append({'label':label,'path':str(path.relative_to(ROOT)),'sha256':sha_file(path),'model':ck['model'],'selected_step':ck['step'],'validation_loss':ck['validation_loss'],'seed':ck['seed'],'initial_checkpoint':ck['initial_checkpoint'],'log_best_step':min(log,key=lambda r:float(r['validation_loss']))['step'],'training_pairs':1776,'validation_pairs':2400,'architecture':ck['architecture'],'mer_alignment':'known digital target transform'})
 writecsv(ROOT/'audit/repair_checkpoint_manifest.csv',ckrows)
 print(json.dumps({'stage2_rows':len(visual_records),'stage2_summaries':len(stage2sum),'digital_validation_pairs':len(drows),'digital_summaries':len(dsum),'repair_checkpoint_steps':[(r['label'],r['selected_step']) for r in ckrows]}),flush=True)
if __name__=='__main__':main()
