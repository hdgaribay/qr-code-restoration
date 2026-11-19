#!/usr/bin/env python3
"""Read-only Stage 2 audit and common-input model evaluation.
Uses only included train/validation NPZ arrays and validation payload hashes.
No image archives, payload text, split manifest, or test identities are opened.
"""
from pathlib import Path
import csv, hashlib, json, sys, time
from collections import defaultdict
import numpy as np
import cv2, torch

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'source'; EXP=SRC/'experiment'; OUT=ROOT
sys.path.insert(0,str(SRC/'prior_project/ArtifactB_Next_Research/src'))
from unet_model import CompactUNet
from methods import read_exact, classical_orig, qr_aware

def readcsv(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:p.write_text('',encoding='utf-8');return
    with p.open('w',newline='',encoding='utf-8') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def bootstrap_ids(rows,key,seed=253112026,B=5000):
    by=defaultdict(list)
    for r in rows:by[r['identity_id']].append(float(r[key]))
    if not by:return (float('nan'),float('nan'))
    ids=sorted(by); vals=np.array([np.mean(by[i]) for i in ids]);rng=np.random.default_rng(seed)
    boot=np.array([vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(B)])
    return tuple(np.quantile(boot,[.025,.975]))
def qstats(v):
    a=np.asarray(v,dtype=float)
    return {'n':int(len(a)),'min':float(np.min(a)),'p10':float(np.percentile(a,10)),'median':float(np.median(a)),'p90':float(np.percentile(a,90)),'max':float(np.max(a))}
def blockiness(im):
    a=im.astype(np.float32)
    dx=np.abs(np.diff(a,axis=1));dy=np.abs(np.diff(a,axis=0))
    vx=np.mean(dx[:,7::8]) if dx.shape[1]>7 else np.nan;vy=np.mean(dy[7::8,:]) if dy.shape[0]>7 else np.nan
    ix=np.mean(dx[:,np.arange(dx.shape[1])%8!=7]);iy=np.mean(dy[np.arange(dy.shape[0])%8!=7,:])
    return float((vx+vy)/2),float((ix+iy)/2)
def illumination_coverage(im):
    # Proxy only: broad morphological closing estimates a slowly varying field.
    a=im.astype(np.uint8);L=cv2.morphologyEx(a,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    L=cv2.GaussianBlur(L,(0,0),9).astype(np.float32);base=max(float(np.median(L)),1.)
    ratio=L/base
    return {f'illum_proxy_fraction_below_{t:.1f}':float(np.mean(ratio<t)) for t in (.9,.8,.7,.6)}
def geometry_row(im,n):
    gray=im
    try:ok,pts=cv2.QRCodeDetector().detect(gray)
    except cv2.error:ok,pts=False,None
    pitch=256.0/(n+8)
    base=np.array([[4*pitch,4*pitch],[(n+4)*pitch,4*pitch],[(n+4)*pitch,(n+4)*pitch],[4*pitch,(n+4)*pitch]],np.float32)
    row={'roi_detected':bool(ok and pts is not None),'expected_module_pitch_px':pitch,'corner_rms_px':'','corner_rms_module':'','edge_scale_error_pct':'','residual_rotation_deg':''}
    if not ok or pts is None:return row
    q=pts.reshape(4,2).astype(np.float32)
    candidates=[]
    for arr in (q,q[::-1]):
        for k in range(4):candidates.append(np.roll(arr,k,axis=0))
    aligned=min(candidates,key=lambda z:float(np.mean(np.sum((z-base)**2,axis=1))))
    err=aligned-base;rms=float(np.sqrt(np.mean(np.sum(err**2,axis=1))))
    edges=np.linalg.norm(np.roll(aligned,-1,axis=0)-aligned,axis=1)
    edge_scale=float(np.mean(edges)/(n*pitch))
    vec=aligned[1]-aligned[0];ang=float(np.degrees(np.arctan2(vec[1],vec[0])))
    row.update({'corner_rms_px':rms,'corner_rms_module':rms/pitch,'edge_scale_error_pct':(edge_scale-1)*100,'residual_rotation_deg':ang})
    return row
def load_models():
    specs=[]
    names=[('STAGE1_SINGLE_STANDARD','stage1_single_standard_best.pt'),('STAGE1_ALL_STANDARD','stage1_all_standard_best.pt'),('STAGE1_ALL_STRUCTURE_WEIGHTED','stage1_all_structure_weighted_best.pt'),
           ('UNET_SINGLE_STANDARD','stage2_single_standard_best.pt'),('UNET_ALL_STANDARD','stage2_all_standard_best.pt'),('UNET_ALL_STRUCTURE_WEIGHTED','stage2_all_structure_weighted_best.pt')]
    for label,name in names:
        p=EXP/'models'/name;ck=torch.load(p,map_location='cpu',weights_only=False);m=CompactUNet();m.load_state_dict(ck['state_dict']);m.eval();specs.append((label,m,p,ck))
    for label,name in [('UNET_PRIOR_STANDARD_PILOT','unet_standard.pt'),('UNET_PRIOR_STRUCTURE_WEIGHTED_PILOT','unet_structure_weighted.pt')]:
        p=SRC/'prior_project/ArtifactB_Next_Research/research_next/models'/name;ck=torch.load(p,map_location='cpu',weights_only=False);m=CompactUNet();m.load_state_dict(ck['state_dict']);m.eval();specs.append((label,m,p,ck))
    return specs

def main():
    torch.set_num_threads(min(4,torch.get_num_threads()))
    logs=EXP/'logs';data=EXP/'data'
    # Checkpoint hash / metadata reconciliation.
    specs=load_models();manifest=[]
    for label,_,p,ck in specs:
        stem=ck.get('model',''); logp=logs/f'{stem}_training_log.csv'
        lrrows=readcsv(logp) if logp.exists() else []
        chosen=min(lrrows,key=lambda x:float(x['validation_loss'])) if lrrows else {}
        manifest.append({'label':label,'relative_path':str(p.relative_to(ROOT)),'sha256':sha_file(p),'checkpoint_model':stem,
         'checkpoint_step':ck.get('step',''),'checkpoint_validation_loss':ck.get('validation_loss',''),'checkpoint_seed_field':ck.get('seed',''),
         'training_log_best_step':chosen.get('step',''),'training_log_best_validation_loss':chosen.get('validation_loss',''),
         'training_log_run_seed':chosen.get('seed',''),'training_log_learning_rate':chosen.get('learning_rate',''),
         'initial_checkpoint':('stage1_'+stem.removeprefix('stage2_')+'_best.pt') if stem.startswith('stage2_') else '',
         'objective_comparable_with':'binary/gradient validation objective; weighted and unweighted objectives are not directly comparable' if 'WEIGHTED' in label else 'binary/gradient validation objective'})
    writecsv(OUT/'audit/checkpoint_manifest.csv',manifest)
    stale=json.loads((logs/'stage2_corrected_training_summary.json').read_text())
    final=json.loads((logs/'model_training_summary.json').read_text())
    reconciliation={'historical_corrected_summary':stale,'final_summary':final,'actual_checkpoint_metadata':{r['label']:{'step':r['checkpoint_step'],'validation_loss':r['checkpoint_validation_loss'],'seed_field':r['checkpoint_seed_field'],'training_log_run_seed':r['training_log_run_seed'],'training_log_best_step':r['training_log_best_step']} for r in manifest if r['checkpoint_model'].startswith('stage2_')},
     'finding':'stage2_corrected_training_summary.json is stale: it records 120 steps while the final stage2 checkpoint metadata and final training logs show step 240. The checkpoint seed field stores the base data seed 253112026; the training log records actual Stage 2 run seed 253112228. The final Stage 2 all-standard checkpoint was selected at step 240 with validation loss 0.5011355; weighted branch uses a different loss objective and starts from the weighted Stage 1 checkpoint, so the historical H2 is not an isolated loss ablation.'}
    (OUT/'audit/metadata_reconciliation.json').write_text(json.dumps(reconciliation,indent=2))
    # Load validation pairs only. Array order is assigned by pair_index in train/validation manifest.
    with np.load(data/'stage2_pairs/validation.npz') as z:xarr=z['x']
    rows=readcsv(data/'stage2_pairs/stage2_pair_manifest.csv')
    val=[r for r in rows if r['split']=='validation']
    assert len(val)==220 and int(sum(r.get('pair_index','')!='' for r in val))==len(xarr)==213
    refs=readcsv(logs/'validation_predictions.csv');refmap={}
    for r in refs:
        if r['method']=='RAW':refmap[(r['identity_id'],r['image_number'],r['severity_level'])]=r['payload_sha256']
    records=[];geometry=[];images={}
    for r in val:
        key=(r['identity_id'],r['image_number'],r['severity_level']); exp_hash=refmap[key]
        im=xarr[int(r['pair_index'])] if r.get('pair_index') else None
        images[key]=im
        gr={'identity_id':r['identity_id'],'image_number':r['image_number'],'condition':r['condition'],'severity_level':r['severity_level'],'module_count':r['module_count'],'geometry_source':r['geometry_source'],'geometry_ok_at_cache':r['geometry_ok'],'pair_index':r.get('pair_index','')}
        if im is not None:gr.update(geometry_row(im,int(r['module_count'])))
        geometry.append(gr)
        methods={}
        if im is None:
            methods={'RAW':(None,0.0),'CLASSICAL_ORIG':(None,0.0),'QR_AWARE':(None,0.0)}
            for label,_,_,_ in specs:methods[label]=(None,0.0)
        else:
            for nm,fn in [('RAW',lambda a:a),('CLASSICAL_ORIG',lambda a:classical_orig(a,kernel=81,sigma=9,window=51,k=.18)),('QR_AWARE',lambda a:qr_aware(a,window_factor=5,k=.18,use_anchor=True,use_guided=True)[0])]:
                t=time.perf_counter();out=fn(im);dt=(time.perf_counter()-t)*1000;methods[nm]=(out,dt)
            tensor=torch.from_numpy((im.astype(np.float32)/255.)[None,None])
            for label,model,_,_ in specs:
                t=time.perf_counter()
                with torch.inference_mode():prob=model(tensor).numpy()[0,0]
                out=np.where(prob>=.5,0,255).astype(np.uint8);dt=(time.perf_counter()-t)*1000
                methods[label]=(out,dt)
        rawtxt=''
        if methods['RAW'][0] is not None:rawtxt=read_exact(methods['RAW'][0])
        rawok=bool(rawtxt and hashlib.sha256(rawtxt.encode()).hexdigest()==exp_hash)
        for method,(out,proc_ms) in methods.items():
            dec='';dms=0.
            if out is not None:
                t=time.perf_counter();dec=read_exact(out);dms=(time.perf_counter()-t)*1000
            exact=bool(dec and hashlib.sha256(dec.encode()).hexdigest()==exp_hash)
            false=bool(dec and not exact)
            records.append({'identity_id':r['identity_id'],'image_number':r['image_number'],'condition':r['condition'],'severity_level':int(r['severity_level']),
             'method':method,'geometry_ok':bool(r['geometry_ok']),'payload_sha256':exp_hash,'decoded_payload':dec,'exact_decode':int(exact),'false_decode':int(false),
             'raw_exact':int(rawok),'recovery':int(not rawok and exact),'regression':int(rawok and not exact),'net_improvement':int(exact)-int(rawok),
             'latency_ms':proc_ms+dms,'mer_valid':0,'mer':'','target_alignment_valid':0,
             'decoder_policy':'single ZXing-C++ QR read on ROI/output; output digest compared after decode; decoder saw image only'} )
    writecsv(OUT/'audit/geometry_alignment.csv',geometry)
    writecsv(OUT/'results/reproduced_stage2_predictions.csv',records)
    # Summaries, confidence intervals clustered by QR identity.
    sums=[]
    for method in sorted({r['method'] for r in records}):
      for level in sorted({r['severity_level'] for r in records}):
       for cond in ['ALL','S','DS','SG','DSG']:
        rr=[r for r in records if r['method']==method and r['severity_level']==level and (cond=='ALL' or r['condition']==cond)]
        if not rr:continue
        ci=bootstrap_ids(rr,'exact_decode',seed=253112026+level+len(cond));ni=bootstrap_ids(rr,'net_improvement',seed=253112126+level+len(cond))
        sums.append({'split':'existing_synthetic_validation_only','method':method,'severity_level':level,'condition':cond,'n_images':len(rr),'n_identities':len({r['identity_id'] for r in rr}),
          'exact_decodes':sum(int(r['exact_decode']) for r in rr),'exact_decode_rate':np.mean([r['exact_decode'] for r in rr]),'exact_ci_identity_low':ci[0],'exact_ci_identity_high':ci[1],
          'recoveries':sum(int(r['recovery']) for r in rr),'regressions':sum(int(r['regression']) for r in rr),'net_improvement':sum(int(r['net_improvement']) for r in rr),
          'net_ci_identity_low':ni[0],'net_ci_identity_high':ni[1],'false_decodes':sum(int(r['false_decode']) for r in rr),'mean_latency_ms':np.mean([r['latency_ms'] for r in rr]),
          'valid_mer_images':0,'module_error_rate':'','finder_error_rate':'','timing_error_rate':'','alignment_error_rate':'','note':'Stage 2 visual target is a paired-Clean sampled matrix, not independent pixel-aligned ground truth; MER withheld.'})
    writecsv(OUT/'results/reproduced_stage2_comparison.csv',sums)
    # Verify reproducibility against the historical aggregate table for same methods/ROIs/decoder.
    old=readcsv(logs/'method_comparison_validation.csv');diff=[]
    oldmap={(r['method'],int(r['severity_level']),r['condition']):r for r in old if r['split']=='validation_synthetic_only'}
    for r in sums:
        if r['method'] not in oldmap and r['method'] not in ('STAGE1_SINGLE','STAGE1_ALL','STAGE1_ALL_WEIGHTED'):continue
        prior=oldmap.get((r['method'],int(r['severity_level']),r['condition']))
        if prior:
            diff.append({'method':r['method'],'level':r['severity_level'],'condition':r['condition'],'recomputed_n':r['n_images'],'historical_n':prior['n_images'],
             'recomputed_exact':r['exact_decodes'],'historical_exact':prior['exact_decodes'],'exact_count_match':int(r['exact_decodes']==int(prior['exact_decodes'])),'recomputed_false_decodes':r['false_decodes'],'historical_false_decodes':prior['false_decodes']})
    writecsv(OUT/'audit/historical_reproduction_check.csv',diff)
    # Domain summaries from Stage 1 validation, Stage 2 train and validation arrays.
    domain=[]
    def add_domain(label,x,meta):
        mean=[];std=[];span=[];hf=[];block=[];inside=[];pitch=[];factor=[]
        for im,row in zip(x,meta):
            a=im.astype(np.float32);mean.append(a.mean());std.append(a.std());span.append(np.percentile(a,95)-np.percentile(a,5))
            hf.append(np.std(a-cv2.GaussianBlur(a,(0,0),2)))
            b,i=blockiness(im);block.append(b/max(i,1e-6))
            inside.append(illumination_coverage(im)['illum_proxy_fraction_below_0.8'])
            if row.get('module_count'):pitch.append(256/(int(row['module_count'])+8))
            if row.get('shadow_factor') not in ('',None):factor.append(float(row['shadow_factor']))
            elif row.get('factor') not in ('',None):factor.append(float(row['factor']))
        for name,vals in [('luminance_mean',mean),('pixel_std',std),('robust_contrast_p95_minus_p5',span),('high_frequency_residual_std',hf),('jpeg_block_boundary_ratio',block),('illumination_proxy_coverage_below_0.8',inside),('expected_module_pitch_px',pitch),('recorded_shadow_factor',factor)]:
            if vals:domain.append({'population':label,'metric':name,**qstats(vals)})
    s1man=readcsv(data/'stage1_pairs_manifest.csv');s1val=[r for r in s1man if r['split']=='validation']
    with np.load(data/'stage1_all_validation.npz') as z:s1x=z['x']
    add_domain('Stage1 generated validation (proxy for training-generation distribution)',s1x,s1val)
    trainman=[r for r in rows if r['split']=='train' and r.get('pair_index')]
    with np.load(data/'stage2_pairs/train.npz') as z:trX=z['x']
    add_domain('Stage2 audited training pairs',trX,trainman)
    valman=[r for r in val if r.get('pair_index')]
    add_domain('Stage2 geometry-valid validation pairs',xarr,valman)
    writecsv(OUT/'audit/domain_gap_summary.csv',domain)
    # Outcome visual data saved as NPZ of actual arrays for downstream figure builder.
    np.savez_compressed(OUT/'data/validation_inputs_and_predictions.npz',x=np.stack([images[k] if images[k] is not None else np.full((256,256),255,np.uint8) for k in images]),
        keys=np.array([json.dumps(k) for k in images],dtype='U200'))
    print(json.dumps({'validation_images':len(val),'geometry_valid':len(xarr),'methods':sorted({r['method'] for r in records}),'prediction_rows':len(records),'domain_rows':len(domain),'geometry_rows':len(geometry),'historical_comparisons':len(diff)}))
if __name__=='__main__':main()
