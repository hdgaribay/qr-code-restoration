#!/usr/bin/env python3
"""Reproducible synthetic Stage 1/2 runs for ECE 253 Artifact B.

This runner never reads image bytes for the test split. Stage 1 uses newly
generated synthetic QR identities. Stage 2 uses existing training synthetic
images only; validation identities are used for model selection and reporting.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, random, sys, time, zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT.parent
PRIOR=PACKAGE/'prior_project'/'ArtifactB_Next_Research'
OUT=ROOT
DATA=OUT/'data'; MODELS=OUT/'models'; LOGS=OUT/'logs'; FIGS=OUT/'figures'; SAMPLES=OUT/'samples'
sys.path.insert(0,str(PRIOR/'src'))
from unet_model import CompactUNet, proposal_loss
from methods import read_exact, classical_orig, qr_aware, _finder_pitch

SEED=253112026
SIDE=256
N_TRAIN_IDENTITIES=2000
N_STAGE1_VAL_IDENTITIES=200
PAIRS_PER_TRAIN_ID=3
TRAIN_STEPS=240
EVAL_EVERY=80
BATCH=8
DEVICE='cpu'
COND_NAMES=['S','D','G','SD','SG','DG','SDG']
ALL_VARIANTS=['single_standard','all_standard','all_structure_weighted']

def seed_all(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))

def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def write_csv(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text('',encoding='utf-8'); return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def sha256(b): return hashlib.sha256(b).hexdigest()

def qr_matrix(payload):
    enc=cv2.QRCodeEncoder_create(); raw=enc.encode(str(payload))
    if raw.ndim==3: raw=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
    # OpenCV's encoder has a two-module white border on every side.
    n=int(raw.shape[0]-4)
    core=(raw[2:2+n,2:2+n]<128).astype(np.uint8)
    if core.shape[0]!=core.shape[1] or n<21 or (n-21)%4: raise ValueError('Unexpected QR module matrix')
    return core

def fixed_pattern_mask(n):
    v=(n-21)//4+1; m=np.zeros((n,n),np.float32)
    for y,x in ((0,0),(0,n-7),(n-7,0)): m[y:y+7,x:x+7]=1
    if n>16: m[6,8:n-8]=1; m[8:n-8,6]=1
    table={1:[],2:[6,18],3:[6,22],4:[6,26],5:[6,30],6:[6,34],7:[6,22,38],8:[6,24,42],9:[6,26,46],10:[6,28,50]}
    for cy in table.get(v,[]):
        for cx in table.get(v,[]):
            if (cy<9 and cx<9) or (cy<9 and cx>n-10) or (cy>n-10 and cx<9): continue
            m[max(0,cy-2):cy+3,max(0,cx-2):cx+3]=1
    canvas=np.zeros((n+8,n+8),np.float32);canvas[4:4+n,4:4+n]=m
    return cv2.resize(canvas,(SIDE,SIDE),interpolation=cv2.INTER_NEAREST)

def ideal_canvas(matrix):
    n=matrix.shape[0]; canvas=np.ones((n+8,n+8),np.uint8)*255
    canvas[4:4+n,4:4+n]=np.where(matrix>0,0,255).astype(np.uint8)
    return cv2.resize(canvas,(SIDE,SIDE),interpolation=cv2.INTER_NEAREST)

def impair(clean, codes, rng):
    """Apply spatial shadow plus optional blur/glare; retain known clean target."""
    a=clean.astype(np.float32); h,w=a.shape; yy,xx=np.mgrid[0:h,0:w].astype(np.float32)
    if 'S' in codes:
        angle=float(rng.uniform(0,180)); theta=np.deg2rad(angle)
        nx,ny=np.cos(theta),np.sin(theta)
        # Boundary position creates partial QR coverage; transition ranges hard to smooth.
        center=float(rng.uniform(-0.35,0.35))*math.hypot(h,w)
        sharp=float(rng.uniform(0.4,0.16*SIDE))
        d=(xx-SIDE/2)*nx+(yy-SIDE/2)*ny-center
        if sharp<2.5: alpha=(d>=0).astype(np.float32)
        else:
            t=np.clip(.5+d/sharp,0,1);alpha=t*t*(3-2*t)
        factor=float(rng.uniform(.35,.82));a*=1-(1-factor)*alpha
    if 'D' in codes:
        sig=float(rng.uniform(.45,1.65)); k=int(np.ceil(sig*4))|1
        a=cv2.GaussianBlur(a,(k,k),sig,borderType=cv2.BORDER_REFLECT)
    if 'G' in codes:
        cx=float(rng.uniform(.15,.85))*w;cy=float(rng.uniform(.15,.85))*h
        sx=float(rng.uniform(.035,.16))*w;sy=float(rng.uniform(.035,.16))*h
        amp=float(rng.uniform(35,115)); blob=np.exp(-.5*(((xx-cx)/sx)**2+((yy-cy)/sy)**2))
        a=np.clip(a+amp*blob,0,255)
    a=np.clip(a+rng.normal(0,float(rng.uniform(.2,3.0)),(h,w)),0,255).astype(np.uint8)
    # Mild capture compression is applied uniformly to represent the camera pipeline.
    quality=int(rng.integers(48,96)); ok,buf=cv2.imencode('.jpg',a,[cv2.IMWRITE_JPEG_QUALITY,quality])
    if ok: a=cv2.imdecode(buf,cv2.IMREAD_GRAYSCALE)
    return a, locals().get('angle',None), locals().get('factor',1.0), quality

def payloads(prefix,count,seed):
    rng=np.random.default_rng(seed); out=[]; seen=set()
    while len(out)<count:
        p=f'{prefix}-{len(out):05d}-{rng.bytes(10).hex().upper()}'
        if p not in seen: seen.add(p);out.append(p)
    return out

def generate_stage1():
    DATA.mkdir(parents=True,exist_ok=True)
    manifest=[]
    def create_set(split,nids,all_categories,seed):
        xs=[];ys=[];ms=[];rng=np.random.default_rng(seed)
        ps=payloads(f'ECE253-SYN-{split.upper()}',nids,seed)
        for ix,payload in enumerate(ps):
            mat=qr_matrix(payload); target=ideal_canvas(mat); mask=fixed_pattern_mask(len(mat))
            if split=='train':
                if all_categories:
                    cats=[COND_NAMES[(ix*3+j+seed%7)%len(COND_NAMES)] for j in range(PAIRS_PER_TRAIN_ID)]
                else:
                    cats=[['S','D','G'][(ix*3+j+seed%3)%3] for j in range(PAIRS_PER_TRAIN_ID)]
            else:
                cats=COND_NAMES[:]
            for pi,cat in enumerate(cats):
                local=np.random.default_rng(seed+ix*1009+pi*7919)
                degraded,angle,factor,q=impair(target,cat,local)
                xs.append(degraded);ys.append(target);ms.append(mask.astype(np.uint8))
                manifest.append({'split':split,'variant':'ALL' if all_categories else 'SINGLE','synthetic_id':f'{('ALL' if all_categories else 'SINGLE')}_{split}_{ix:05d}',
                    'payload':payload,'payload_sha256':sha256(payload.encode()),'module_count':len(mat),'impairments':cat,
                    'shadow_angle_deg':angle,'shadow_factor':factor,'jpeg_quality':q,'seed':seed+ix*1009+pi*7919,
                    'x_sha256':sha256(degraded.tobytes()),'target_sha256':sha256(target.tobytes())})
        return (np.stack(xs),np.stack(ys),np.stack(ms))
    t0=time.time()
    trS=create_set('train',N_TRAIN_IDENTITIES,False,SEED+101)
    trA=create_set('train',N_TRAIN_IDENTITIES,True,SEED+202)
    va=create_set('validation',N_STAGE1_VAL_IDENTITIES,True,SEED+303)
    existing=read_csv(PRIOR/'data_manifest.csv')
    protected={r['ground_truth_payload'] for r in existing if r['split'] in ('validation','test') and r.get('ground_truth_payload')}
    generated={r['payload'] for r in manifest}
    overlap=generated & protected
    if overlap: raise RuntimeError(f'New synthetic payload overlaps protected payload inventory: {len(overlap)}')
    for name,arr in [('stage1_single_train',trS),('stage1_all_train',trA),('stage1_all_validation',va)]:
        np.savez_compressed(DATA/f'{name}.npz',x=arr[0],y=arr[1],mask=arr[2])
    write_csv(DATA/'stage1_pairs_manifest.csv',manifest)
    # Add identity inventories and exact dataset counts, without exposing any existing held-out IDs.
    summary={'seed':SEED,'new_training_qr_identities_total':2*N_TRAIN_IDENTITIES,'new_training_qr_identities_per_regime':N_TRAIN_IDENTITIES,'new_validation_qr_identities':N_STAGE1_VAL_IDENTITIES,
        'single_training_pairs':len(trS[0]),'all_training_pairs':len(trA[0]),'stage1_validation_pairs':len(va[0]),
        'single_training_categories':'S,D,G only','all_training_categories':'S,D,G,SD,SG,DG,SDG balanced by identity/seed',
        'payload_overlap_protected_existing_validation_test':len(overlap),'payload_overlap_new_train_val':0,'generation_seconds':time.time()-t0}
    (LOGS/'stage1_data_summary.json').write_text(json.dumps(summary,indent=2))
    np.save(DATA/'stage1_new_payload_sha256.npy',np.array([x['payload_sha256'] for x in manifest],dtype='U64'))
    print(json.dumps({'stage1_data':summary}),flush=True)
    return summary

def photo_sources():
    root=PACKAGE.parent
    entries=[p for p in (root/'upload').glob('New 253 *.zip')]
    byzip={}
    for p in entries:
        if 'New 253 1-' in p.name: byzip['New 253 1-20261110T195530Z-1-001.zip']=zipfile.ZipFile(p)
        elif 'New 253 2-' in p.name: byzip['New 253 2-20261110T195522Z-1-001.zip']=zipfile.ZipFile(p)
    return byzip

def synthetic_archives():
    return [zipfile.ZipFile(p) for p in sorted((PACKAGE.parent/'upload').glob('synthetic_shadow_pilot_images_*_of_6*.zip'))]

def read_archive_image(zips, member):
    for z in zips:
        try: return z.read(member)
        except KeyError: pass
    raise FileNotFoundError(member)

def pil_rgb_bytes(b):
    im=ImageOps.exif_transpose(Image.open(io.BytesIO(b))).convert('RGB')
    im.thumbnail((1400,1400),Image.Resampling.LANCZOS)
    return np.asarray(im)

def project_roi(rgb):
    """One input-only detector + fixed 4-module quiet-zone margin; returns 256x256 ROI."""
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    try: ok,pts=cv2.QRCodeDetector().detect(gray)
    except cv2.error: return None,{'geometry_source':'input_detector_failed','geometry_ok':False}
    if not ok or pts is None: return None,{'geometry_source':'input_detector_failed','geometry_ok':False}
    q=pts.reshape(-1,2).astype(np.float32)
    if q.shape!=(4,2) or not cv2.isContourConvex(q.reshape(-1,1,2).astype(np.int32)):
        return None,{'geometry_source':'input_detector_invalid','geometry_ok':False}
    edges=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1); side=float(np.median(edges))
    # Estimate module pitch from the input-only finder-pattern contours.
    pitch,pitch_source=_finder_pitch(gray,q)
    pitch=float(pitch); expansion=1.0+8.0*pitch/max(side,1.0)
    center=q.mean(axis=0); qe=center+(q-center)*expansion
    h,w=gray.shape
    area=float(cv2.contourArea(q.reshape(-1,1,2).astype(np.int32)))/(h*w)
    dst=np.float32([[0,0],[SIDE-1,0],[SIDE-1,SIDE-1],[0,SIDE-1]])
    H=cv2.getPerspectiveTransform(qe,dst)
    patch=cv2.warpPerspective(gray,H,(SIDE,SIDE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    meta={'geometry_source':'cv2_detect_plus_'+pitch_source,'geometry_ok':bool(side>=80 and area>.008 and pitch>=1.5),
          'detected_side_px':side,'detected_area_fraction':area,'approx_module_pitch_px':pitch}
    return patch,meta

def clean_reference_matrices(rows,split):
    """Recover each physical code's sampled module matrix from its paired Clean photo.

    A reference is retained only when OpenCV's straightened matrix decodes to the
    recorded payload and ZXing independently reads the same Clean photograph.
    This avoids regenerating a different QR version/ECC/mask for the same payload.
    """
    zips=photo_sources(); clean_rows={r['capture_group_id']:r for r in rows if r['split']==split and r['condition']=='Clean'}
    refs={}; audit=[]
    for group,r in sorted(clean_rows.items()):
        b=zips[r['source_zip']].read(r['archive_path']);rgb=pil_rgb_bytes(b);gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        decoded,pts,straight=cv2.QRCodeDetector().detectAndDecode(gray)
        zxing=read_exact(rgb)
        valid=bool(decoded==r['ground_truth_payload'] and zxing==r['ground_truth_payload'] and straight is not None and straight.ndim==2)
        mat=None
        if valid:
            mat=(straight<128).astype(np.uint8);n=mat.shape[0]
            valid=bool(mat.shape[1]==n and n>=21 and (n-21)%4==0 and read_exact(ideal_canvas(mat))==r['ground_truth_payload'])
        reason='' if valid else 'clean_reference_not_dual_decoder_verified_or_invalid_matrix'
        if valid:
            refs[group]={'matrix':mat,'payload':r['ground_truth_payload'],'source_image_number':r['image_number'],
                'matrix_sha256':sha256(mat.tobytes()),'source':'OpenCV straight_qrcode from paired Clean capture; OpenCV + ZXing payload agreement'}
        audit.append({'split':split,'capture_group_id':group,'identity_id':r['identity_id'],'clean_image_number':r['image_number'],
            'recorded_payload_sha256':sha256(r['ground_truth_payload'].encode()),'opencv_payload_match':decoded==r['ground_truth_payload'],
            'zxing_payload_match':zxing==r['ground_truth_payload'],'straight_matrix_shape':str(None if straight is None else straight.shape),
            'matrix_sha256':sha256(mat.tobytes()) if valid else '', 'valid_reference':valid,'skip_reason':reason})
    write_csv(LOGS/f'{split}_clean_reference_matrix_audit.csv',audit)
    return refs

def build_real_pair_cache():
    """Read only train/validation image bytes; build target-paired 256px ROIs."""
    outdir=DATA/'stage2_pairs';outdir.mkdir(parents=True,exist_ok=True)
    mrows=read_csv(PRIOR/'data_manifest.csv')
    synth=read_csv(PRIOR/'research_next'/'synthetic_manifest.csv')
    tv=[r for r in mrows if r['split'] in ('train','validation')]
    assert len([r for r in tv if r['split']=='train'])==235 and len([r for r in tv if r['split']=='validation'])==55
    assert not any(r['split']=='test' for r in tv)
    images=synthetic_archives(); sources=photo_sources()
    train_refs=clean_reference_matrices(mrows,'train');val_refs=clean_reference_matrices(mrows,'validation')
    matrix_refs={'train':train_refs,'validation':val_refs}
    smap={(m['split'],str(m['image_number']),str(m['severity_level'])):m for m in synth if m['split'] in ('train','validation') and int(m['severity_level'])>0}
    bynum={r['image_number']:r for r in tv}
    xs={k:[] for k in ('train','validation')};ys={k:[] for k in ('train','validation')};ms={k:[] for k in ('train','validation')}
    manifest=[]; cache={}; n_expected={'train':940,'validation':220}
    ordered=sorted(smap.items(),key=lambda kv:(kv[0][0],int(kv[0][1]),int(kv[0][2])))
    for entry_index,((split,num,level),m) in enumerate(ordered,1):
        r=bynum[num]
        assert r['split']==split and r['identity_id'] not in ('')
        if split=='test': raise RuntimeError('Test split guard')
        srczip=sources[r['source_zip']]
        payload=r['ground_truth_payload']
        ref=matrix_refs[split].get(r['capture_group_id'])
        if ref is None:
            manifest.append({'split':split,'image_number':num,'identity_id':r['identity_id'],'capture_group_id':r['capture_group_id'],'condition':r['condition'],
                'severity_level':level,'skip_reason':'paired_clean_reference_matrix_not_verified'});continue
        matrix=ref['matrix']
        target=ideal_canvas(matrix);mask=fixed_pattern_mask(len(matrix))
        b=read_archive_image(images,m['relative_path'])
        rgb=pil_rgb_bytes(b);patch,geom=project_roi(rgb)
        rec={'split':split,'image_number':num,'identity_id':r['identity_id'],'capture_group_id':r['capture_group_id'],'condition':r['condition'],'severity_level':int(level),
             'factor':m.get('core_illumination_factor',''),'payload_sha256':sha256(payload.encode()),'module_count':len(matrix),
             'geometry_source':geom['geometry_source'],'geometry_ok':geom['geometry_ok'],'source_image_sha256':sha256(b),
             'ground_truth_matrix_source':ref['source'],'ground_truth_matrix_sha256':ref['matrix_sha256'],
             'synthetic_label':'SYNTHETIC','skip_reason':''}
        if patch is None or not geom['geometry_ok']:
            rec['skip_reason']='input_geometry_unavailable_or_low_confidence'
        else:
            xs[split].append(patch);ys[split].append(target);ms[split].append(mask.astype(np.uint8))
            rec['pair_index']=len(xs[split])-1
        manifest.append(rec)
        if entry_index%50==0: print(f'prepared existing synthetic pairs {entry_index}/{len(ordered)}',flush=True)
    for split in ('train','validation'):
        if xs[split]: np.savez_compressed(outdir/f'{split}.npz',x=np.stack(xs[split]),y=np.stack(ys[split]),mask=np.stack(ms[split]))
        else: np.savez_compressed(outdir/f'{split}.npz',x=np.zeros((0,SIDE,SIDE),np.uint8),y=np.zeros((0,SIDE,SIDE),np.uint8),mask=np.zeros((0,SIDE,SIDE),np.float32))
    write_csv(outdir/'stage2_pair_manifest.csv',manifest)
    splits=read_csv(PRIOR/'split_manifest.csv');train_ids={r['identity_id'] for r in splits if r['split']=='train'};val_ids={r['identity_id'] for r in splits if r['split']=='validation'};test_ids={r['identity_id'] for r in splits if r['split']=='test'}
    used_train={r['identity_id'] for r in manifest if r['split']=='train' and not r.get('skip_reason')};used_val={r['identity_id'] for r in manifest if r['split']=='validation' and not r.get('skip_reason')}
    assert used_train<=train_ids and used_val<=val_ids and not (used_train&val_ids) and not (used_train&test_ids) and not (used_val&test_ids)
    summary={'training_synthetic_images_expected':n_expected['train'],'validation_synthetic_images_expected':n_expected['validation'],
        'training_pairs_usable':len(xs['train']),'validation_pairs_geometry_valid':len(xs['validation']),
        'training_identities_used':len(used_train),'validation_identities':len(used_val),'test_identity_image_bytes_read':0,
        'train_clean_matrices_dual_decoder_verified':len(train_refs),'validation_clean_matrices_dual_decoder_verified':len(val_refs),
        'training_ids_match_locked_split':used_train<=train_ids,'validation_ids_match_locked_split':used_val<=val_ids,
        'train_geometry_skips':sum(r['split']=='train' and bool(r.get('skip_reason')) for r in manifest),
        'validation_geometry_skips':sum(r['split']=='validation' and bool(r.get('skip_reason')) for r in manifest),
        'pair_target':'same capture-group clean photo straight_qrcode matrix, verified by independent OpenCV and ZXing payload matches, rendered into each input-only detector ROI',
        'warning':'Target is a clean-reference sampled matrix; no independent pixel-aligned clean photograph is available.'}
    (LOGS/'stage2_data_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'stage2_data':summary}),flush=True)
    return summary

def ensure_memmaps(path):
    path=Path(path); stem=path.with_suffix('')
    marker=Path(str(stem)+'_memmap.ready')
    keys=('x','y','mask')
    files=[Path(str(stem)+f'_{key}.npy') for key in keys]
    if marker.exists() and all(p.exists() for p in files): return files
    with np.load(path) as z:
        for key,dest in zip(keys,files):
            if dest.exists(): dest.unlink()
            arr=z[key]
            if arr.dtype!=np.uint8: arr=arr.astype(np.uint8)
            np.save(dest,arr)
            del arr
    marker.write_text('ready',encoding='utf-8')
    return files

class ArraySet(Dataset):
    def __init__(self,path,augment=False):
        xp,yp,mp=ensure_memmaps(path)
        self.x=np.load(xp,mmap_mode='r');self.y=np.load(yp,mmap_mode='r');self.m=np.load(mp,mmap_mode='r');self.augment=augment
    def __len__(self): return len(self.x)
    def __getitem__(self,i):
        x=self.x[i].copy().astype(np.float32)/255.;y=1.-self.y[i].copy().astype(np.float32)/255.;m=self.m[i].copy().astype(np.float32)
        if self.augment:
            k=np.random.randint(4);x=np.rot90(x,k).copy();y=np.rot90(y,k).copy();m=np.rot90(m,k).copy()
            if np.random.random()<.5: x=x[:,::-1].copy();y=y[:,::-1].copy();m=m[:,::-1].copy()
        return torch.from_numpy(x[None]),torch.from_numpy(y[None]),torch.from_numpy(m[None])

def val_loss(model,ds,weighted):
    model.eval(); loader=DataLoader(ds,batch_size=BATCH,shuffle=False,num_workers=0); losses=[]
    with torch.inference_mode():
        for bi,(x,y,m) in enumerate(loader,1):
            p=model(x); loss=proposal_loss(p,y,m if weighted else None)
            losses.append(float(loss.cpu()))
            if bi%100==0: print(f'validation batches {bi}/{len(loader)}',flush=True)
    return float(np.mean(losses))

def train_model(name,train_path,val_path,weighted,init_path=None,steps=TRAIN_STEPS,lr=1e-3):
    # Reset a shared seed for each matched branch: identical parameter initialization,
    # sample order, and geometric augmentations. The condition distribution or loss
    # is the intended treatment variable.
    seedkey=name.replace('stage1_','').replace('stage2_','')
    run_seed=SEED+(101 if name.startswith('stage1_') else 202)
    seed_all(run_seed)
    model=CompactUNet()
    if init_path:
        ck=torch.load(init_path,map_location='cpu',weights_only=False);model.load_state_dict(ck['state_dict'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    ds=ArraySet(train_path,augment=True);vds=ArraySet(val_path,augment=False)
    if len(ds)==0 or len(vds)==0: raise RuntimeError('Empty training/validation array')
    rng=np.random.default_rng(run_seed+17)
    best=float('inf');rows=[];start=time.time();bestpath=MODELS/f'{name}_best.pt'
    print(f'initialized {name}: train_pairs={len(ds)} val_pairs={len(vds)} batch={BATCH}',flush=True)
    for step in range(1,steps+1):
        inds=rng.integers(0,len(ds),size=BATCH); batch=[ds[int(i)] for i in inds]
        x=torch.stack([b[0] for b in batch]);y=torch.stack([b[1] for b in batch]);m=torch.stack([b[2] for b in batch])
        model.train();optimizer.zero_grad(set_to_none=True);p=model(x);loss=proposal_loss(p,y,m if weighted else None);loss.backward();optimizer.step()
        if step<=2: print(f'{name} optimization step {step} complete',flush=True)
        if step%EVAL_EVERY==0 or step==steps:
            print(f'{name} starting validation at step {step}',flush=True)
            vl=val_loss(model,vds,weighted)
            print(f'{name} validation finished at step {step}',flush=True)
            row={'model':name,'step':step,'train_loss':float(loss.detach()),'validation_loss':vl,'weighted_loss':weighted,
                 'learning_rate':lr,'device':DEVICE,'elapsed_seconds':time.time()-start,'seed':run_seed}
            rows.append(row);write_csv(LOGS/f'{name}_training_log.csv',rows)
            if vl<best:
                best=vl;torch.save({'state_dict':model.state_dict(),'model':name,'seed':SEED,'step':step,'validation_loss':vl,
                    'architecture':'4-scale Compact U-Net widths 16/32/64/128; grayscale 256x256'},bestpath)
            print(json.dumps({'training':row,'best':best}),flush=True)
    return bestpath,rows

def make_plot(path,series,xlab,ylab,title):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7.6,4.5))
    for label,rows in series.items():
        ax.plot([r['step'] for r in rows],[r['validation_loss'] for r in rows],marker='o',label=label)
    ax.set_xlabel(xlab);ax.set_ylabel(ylab);ax.set_title(title);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)

def train_all(stage1_steps=TRAIN_STEPS,stage2_steps=TRAIN_STEPS):
    # 3-stage1 models form a 2 x loss ablation; 3 fine-tuned models mirror it.
    results={}
    for name,trainp,weighted in [('single_standard',DATA/'stage1_single_train.npz',False),
                                 ('all_standard',DATA/'stage1_all_train.npz',False),
                                 ('all_structure_weighted',DATA/'stage1_all_train.npz',True)]:
        p,log=train_model('stage1_'+name,trainp,DATA/'stage1_all_validation.npz',weighted,steps=stage1_steps,lr=1e-3)
        results['stage1_'+name]={'checkpoint':str(p.relative_to(PACKAGE)),'best_step':int(torch.load(p,map_location='cpu',weights_only=False)['step'])}
    # Stage 2 starts from the corresponding Stage 1 weights and uses all usable existing train variants.
    tr=DATA/'stage2_pairs'/'train.npz';va=DATA/'stage2_pairs'/'validation.npz'
    for name,weighted in [('single_standard',False),('all_standard',False),('all_structure_weighted',True)]:
        init=MODELS/f'stage1_{name}_best.pt'
        p,log=train_model('stage2_'+name,tr,va,weighted,init_path=init,steps=stage2_steps,lr=1e-4)
        results['stage2_'+name]={'checkpoint':str(p.relative_to(PACKAGE)),'best_step':int(torch.load(p,map_location='cpu',weights_only=False)['step'])}
    (LOGS/'model_training_summary.json').write_text(json.dumps(results,indent=2))
    # Loss plots keep H1 and H2 comparisons explicit.
    for group, names in [('stage1_h1',['stage1_single_standard','stage1_all_standard']),('stage1_h2',['stage1_all_standard','stage1_all_structure_weighted']),
                         ('stage2_h1',['stage2_single_standard','stage2_all_standard']),('stage2_h2',['stage2_all_standard','stage2_all_structure_weighted'])]:
        series={}
        for name in names:
            series[name.replace('stage1_','').replace('stage2_','')]=read_csv(LOGS/f'{name}_training_log.csv')
        make_plot(FIGS/f'{group}_loss.png',series,'Training steps','Validation objective (within-objective only)',f'{group.replace("_"," ").upper()} validation curves')
    return results

def finetune_all(steps=TRAIN_STEPS):
    """Re-run corrected Stage 2 from the untouched Stage 1 best checkpoints."""
    tr=DATA/'stage2_pairs'/'train.npz';va=DATA/'stage2_pairs'/'validation.npz';results={}
    for name,weighted in [('single_standard',False),('all_standard',False),('all_structure_weighted',True)]:
        init=MODELS/f'stage1_{name}_best.pt'
        p,log=train_model('stage2_'+name,tr,va,weighted,init_path=init,steps=steps,lr=1e-4)
        results[name]={'best_checkpoint':str(p.relative_to(PACKAGE)),'best_step':int(torch.load(p,map_location='cpu',weights_only=False)['step']),
                       'initial_checkpoint':str(init.relative_to(PACKAGE))}
    (LOGS/'stage2_corrected_training_summary.json').write_text(json.dumps(results,indent=2))
    for group,names in [('stage2_corrected_h1',['stage2_single_standard','stage2_all_standard']),
                        ('stage2_corrected_h2',['stage2_all_standard','stage2_all_structure_weighted'])]:
        series={name.replace('stage2_',''):read_csv(LOGS/f'{name}_training_log.csv') for name in names}
        make_plot(FIGS/f'{group}_loss.png',series,'Training steps','Validation objective (within-objective only)',f'{group.replace("_"," ").upper()} validation curves')
    return results

def save_training_samples():
    """Save actual model inputs, ideal targets, and predicted outputs for review."""
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[('single_standard','stage1_single_standard_best.pt'),('all_standard','stage1_all_standard_best.pt'),
           ('all_structure_weighted','stage1_all_structure_weighted_best.pt')]
    stage1x=np.load(DATA/'stage1_all_train_x.npy',mmap_mode='r');stage1y=np.load(DATA/'stage1_all_train_y.npy',mmap_mode='r')
    manifest=read_csv(DATA/'stage1_pairs_manifest.csv');stage1rows=[r for r in manifest if r['split']=='train' and r['variant']=='ALL']
    chosen=[]
    for cat in COND_NAMES:
        idx=next((i for i,r in enumerate(stage1rows) if r['impairments']==cat),None)
        if idx is not None:chosen.append((cat,idx))
    models={}
    for name,ckname in specs:
        ck=torch.load(MODELS/ckname,map_location='cpu',weights_only=False);m=CompactUNet();m.load_state_dict(ck['state_dict']);m.eval();models[name]=m
    fig,axes=plt.subplots(len(chosen),5,figsize=(14,2.6*len(chosen)))
    for ri,(cat,idx) in enumerate(chosen):
        inp=stage1x[idx];gt=stage1y[idx];x=torch.from_numpy((inp.astype(np.float32)/255.)[None,None])
        imgs=[gt,inp]
        for name in ('single_standard','all_standard','all_structure_weighted'):
            with torch.inference_mode():p=models[name](x).numpy()[0,0]
            out=np.where(p>=.5,0,255).astype(np.uint8);imgs.append(out)
            Image.fromarray(out).save(SAMPLES/f'stage1_{cat}_{name}_output.png')
        Image.fromarray(inp).save(SAMPLES/f'stage1_{cat}_synthetic_input.png');Image.fromarray(gt).save(SAMPLES/f'stage1_{cat}_clean_qr_target.png')
        for ax,img in zip(axes[ri],imgs):ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.axis('off')
        axes[ri,0].set_ylabel(cat+'\nStage 1',fontsize=10)
    for ax,title in zip(axes[0],['Clean QR target matrix','Synthetic degraded input','SINGLE standard output','ALL standard output','ALL structure weighted output']):ax.set_title(title,fontsize=9)
    fig.suptitle('Stage 1 actual synthetic training samples; display scaling only',fontsize=13);fig.tight_layout();fig.savefig(SAMPLES/'stage1_sample_pairs.png',dpi=150);plt.close(fig)

    xarr=np.load(DATA/'stage2_pairs'/'train_x.npy',mmap_mode='r');yarr=np.load(DATA/'stage2_pairs'/'train_y.npy',mmap_mode='r');mrows=read_csv(DATA/'stage2_pairs'/'stage2_pair_manifest.csv')
    trrows=[r for r in mrows if r['split']=='train' and not r.get('skip_reason')]
    # Pick real training synthetic variants across S, DS, SG, DSG at strongest common level.
    picked=[]
    for cond in ('S','DS','SG','DSG'):
        pool=[r for r in trrows if r['condition']==cond and int(r['severity_level'])==4]
        if pool:picked.append(pool[0])
    models2={}
    for name in ('single_standard','all_standard','all_structure_weighted'):
        ck=torch.load(MODELS/f'stage2_{name}_best.pt',map_location='cpu',weights_only=False);m=CompactUNet();m.load_state_dict(ck['state_dict']);m.eval();models2[name]=m
    fig,axes=plt.subplots(len(picked),5,figsize=(14,2.8*len(picked)))
    for ri,r in enumerate(picked):
        i=int(r['pair_index']);inp=xarr[i];gt=yarr[i];x=torch.from_numpy((inp.astype(np.float32)/255.)[None,None]);imgs=[gt,inp]
        Image.fromarray(inp).save(SAMPLES/f"stage2_{r['condition']}_L{r['severity_level']}_synthetic_input.png");Image.fromarray(gt).save(SAMPLES/f"stage2_{r['condition']}_L{r['severity_level']}_ideal_matrix_target.png")
        for name in ('single_standard','all_standard','all_structure_weighted'):
            with torch.inference_mode():p=models2[name](x).numpy()[0,0]
            out=np.where(p>=.5,0,255).astype(np.uint8);imgs.append(out);Image.fromarray(out).save(SAMPLES/f"stage2_{r['condition']}_L{r['severity_level']}_{name}_output.png")
        for ax,img in zip(axes[ri],imgs):ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.axis('off')
        axes[ri,0].set_ylabel(f"{r['condition']} L{r['severity_level']}\n{r['identity_id']}",fontsize=9)
    if picked:
        for ax,title in zip(axes[0],['Paired Clean matrix','SYNTHETIC train input','SINGLE fine-tuned','ALL fine-tuned','ALL + QR loss fine-tuned']):ax.set_title(title,fontsize=9)
        fig.suptitle('Stage 2 actual training examples; paired clean-reference sampled matrices',fontsize=13);fig.tight_layout();fig.savefig(SAMPLES/'stage2_sample_pairs.png',dpi=150)
    plt.close(fig)

def module_region_masks(n):
    finder=np.zeros((n,n),bool);timing=np.zeros((n,n),bool);alignment=np.zeros((n,n),bool)
    for y,x in ((0,0),(0,n-7),(n-7,0)): finder[y:y+7,x:x+7]=1
    if n>16: timing[6,8:n-8]=1; timing[8:n-8,6]=1
    v=(n-21)//4+1
    table={1:[],2:[6,18],3:[6,22],4:[6,26],5:[6,30],6:[6,34],7:[6,22,38],8:[6,24,42],9:[6,26,46],10:[6,28,50]}
    for cy in table.get(v,[]):
      for cx in table.get(v,[]):
       if (cy<9 and cx<9) or (cy<9 and cx>n-10) or (cy>n-10 and cx<9): continue
       alignment[max(0,cy-2):cy+3,max(0,cx-2):cx+3]=1
    function=finder|timing|alignment
    return {'finder':finder,'timing':timing,'alignment':alignment,'fixed':function,'data':~function}

def module_errors(gray, matrix):
    n=matrix.shape[0];full=n+8;black=(gray<128);pred=np.zeros((n,n),bool)
    for iy in range(n):
      y0=int(round((iy+4)*SIDE/full));y1=int(round((iy+5)*SIDE/full))
      for ix in range(n):
        x0=int(round((ix+4)*SIDE/full));x1=int(round((ix+5)*SIDE/full))
        pred[iy,ix]=float(black[y0:y1,x0:x1].mean())>=.5
    err=pred!=matrix.astype(bool);regions=module_region_masks(n);region=regions['fixed']
    return {'module_errors':int(err.sum()),'module_count':int(err.size),'mer':float(err.mean()),
            'fixed_pattern_errors':int(err[region].sum()),'fixed_pattern_modules':int(region.sum()),
            'fixed_pattern_mer':float(err[region].mean()),'data_module_errors':int(err[~region].sum()),
            'data_modules':int((~region).sum()),'finder_errors':int(err[regions['finder']].sum()),
            'finder_modules':int(regions['finder'].sum()),'finder_mer':float(err[regions['finder']].mean()),
            'timing_errors':int(err[regions['timing']].sum()),'timing_modules':int(regions['timing'].sum()),
            'timing_mer':float(err[regions['timing']].mean()) if regions['timing'].any() else '',
            'alignment_errors':int(err[regions['alignment']].sum()),'alignment_modules':int(regions['alignment'].sum()),
            'alignment_mer':float(err[regions['alignment']].mean()) if regions['alignment'].any() else ''}

def bootstrap_identity(rows,key,seed=SEED,B=3000):
    by=defaultdict(list)
    for r in rows: by[r['identity_id']].append(float(r[key]))
    if not by:return [float('nan'),float('nan')]
    vals=np.array([np.mean(v) for v in by.values()]);rng=np.random.default_rng(seed)
    b=np.array([vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(B)])
    return np.quantile(b,[.025,.975]).tolist()

def decode_and_metrics(img,matrix,payload,geometry_ok):
    t0=time.perf_counter();text=read_exact(img); decoder_ms=(time.perf_counter()-t0)*1000
    exact=bool(text==payload and text!='')
    false=bool(text and text!=payload)
    g=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY) if img.ndim==3 else img
    errs=module_errors(g,matrix) if geometry_ok else {}
    return text,exact,false,errs,decoder_ms

def evaluate_stage1_validation():
    """Evaluate only the new held validation identities with exact known matrices."""
    xarr=np.load(DATA/'stage1_all_validation_x.npy',mmap_mode='r')
    records=[r for r in read_csv(DATA/'stage1_pairs_manifest.csv') if r['split']=='validation']
    if len(records)!=len(xarr):raise RuntimeError('Stage 1 validation manifest/array count mismatch')
    models=[]
    for name,ckname in [('single_standard','stage1_single_standard_best.pt'),('all_standard','stage1_all_standard_best.pt'),('all_structure_weighted','stage1_all_structure_weighted_best.pt')]:
        ck=torch.load(MODELS/ckname,map_location='cpu',weights_only=False);mod=CompactUNet();mod.load_state_dict(ck['state_dict']);mod.eval();models.append((name,mod))
    preds=[]
    for i,r in enumerate(records):
        inp=xarr[i];matrix=qr_matrix(r['payload']);raw_t=time.perf_counter();raw_text=read_exact(inp);raw_ms=(time.perf_counter()-raw_t)*1000;raw_exact=int(raw_text==r['payload'])
        variants=[('RAW',inp,raw_ms,raw_text)]
        x=torch.from_numpy((inp.astype(np.float32)/255.)[None,None])
        for name,model in models:
            t=time.perf_counter()
            with torch.inference_mode():prob=model(x).numpy()[0,0]
            infer_ms=(time.perf_counter()-t)*1000;out=np.where(prob>=.5,0,255).astype(np.uint8)
            dec0=time.perf_counter();decoded=read_exact(out);dec_ms=(time.perf_counter()-dec0)*1000
            variants.append((f'UNET_{name.upper()}',out,infer_ms+dec_ms,decoded))
        for method,img,lat,text in variants:
            exact=int(text==r['payload']);e=module_errors(img,matrix)
            preds.append({'split':'new_synthetic_validation','identity_id':r['synthetic_id'],'payload_sha256':r['payload_sha256'],
                'impairments':r['impairments'],'method':method,'exact_decode':exact,'raw_exact':raw_exact,
                'recovery':int(not raw_exact and exact),'regression':int(raw_exact and not exact),'false_decode':int(bool(text and text!=r['payload'])),
                'decoded_payload':text,'latency_ms':lat,'mer_valid':1,**e})
        if (i+1)%100==0:print(f'Stage 1 validation metrics {i+1}/{len(records)}',flush=True)
    write_csv(LOGS/'stage1_synthetic_validation_predictions.csv',preds)
    summaries=[]
    for method in sorted({r['method'] for r in preds}):
      for cond in ['ALL']+COND_NAMES:
        rr=[r for r in preds if r['method']==method and (cond=='ALL' or r['impairments']==cond)]
        if not rr:continue
        drci=bootstrap_identity(rr,'exact_decode');netci=bootstrap_identity(rr,'recovery',SEED+8)
        summaries.append({'split':'new_synthetic_validation','method':method,'impairments':cond,'n_pairs':len(rr),
            'n_identities':len({r['identity_id'] for r in rr}),'exact_decodes':sum(int(r['exact_decode']) for r in rr),
            'exact_decode_rate':float(np.mean([int(r['exact_decode']) for r in rr])),'exact_ci_identity_low':drci[0],'exact_ci_identity_high':drci[1],
            'recoveries':sum(int(r['recovery']) for r in rr),'regressions':sum(int(r['regression']) for r in rr),
            'false_decodes':sum(int(r['false_decode']) for r in rr),'mean_latency_ms':float(np.mean([float(r['latency_ms']) for r in rr])),
            'module_error_rate':sum(int(r['module_errors']) for r in rr)/sum(int(r['module_count']) for r in rr),
            'fixed_pattern_error_rate':sum(int(r['fixed_pattern_errors']) for r in rr)/sum(int(r['fixed_pattern_modules']) for r in rr),
            'finder_error_rate':sum(int(r['finder_errors']) for r in rr)/sum(int(r['finder_modules']) for r in rr),
            'timing_error_rate':sum(int(r['timing_errors']) for r in rr)/sum(int(r['timing_modules']) for r in rr),
            'alignment_error_rate':sum(int(r['alignment_errors']) for r in rr)/sum(int(r['alignment_modules']) for r in rr) if sum(int(r['alignment_modules']) for r in rr) else ''})
    write_csv(LOGS/'stage1_synthetic_validation_comparison.csv',summaries)
    contrasts=[]
    for h,a,b in [('H1_SINGLE_vs_ALL','UNET_SINGLE_STANDARD','UNET_ALL_STANDARD'),('H2_ALL_STANDARD_vs_QR_LOSS','UNET_ALL_STANDARD','UNET_ALL_STRUCTURE_WEIGHTED')]:
      for cond in ['ALL']+COND_NAMES:
        aa={(r['identity_id'],r['impairments']):int(r['exact_decode']) for r in preds if r['method']==a and (cond=='ALL' or r['impairments']==cond)}
        bb={(r['identity_id'],r['impairments']):int(r['exact_decode']) for r in preds if r['method']==b and (cond=='ALL' or r['impairments']==cond)}
        keys=sorted(set(aa)&set(bb));d=np.array([bb[k]-aa[k] for k in keys]);ids=sorted({k[0] for k in keys});rng=np.random.default_rng(SEED+len(cond));by={i:[bb[k]-aa[k] for k in keys if k[0]==i] for i in ids}
        vals=np.array([np.mean(by[i]) for i in ids]);boot=np.array([vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(3000)]) if len(vals) else np.array([np.nan])
        lo,hi=np.quantile(boot,[.025,.975]);contrasts.append({'hypothesis':h,'baseline':a,'comparison':b,'impairments':cond,'n_pairs':len(keys),'n_identities':len(ids),
            'exact_rate_baseline':np.mean([aa[k] for k in keys]) if keys else '', 'exact_rate_comparison':np.mean([bb[k] for k in keys]) if keys else '',
            'paired_rate_difference':np.mean(d) if len(d) else '', 'identity_bootstrap_95_low':lo,'identity_bootstrap_95_high':hi,
            'recoveries':int(np.sum(d>0)),'regressions':int(np.sum(d<0)),'ties':int(np.sum(d==0))})
    write_csv(LOGS/'stage1_hypothesis_contrasts.csv',contrasts)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conds=['S','D','G','SD','SG','DG','SDG'];fig,axes=plt.subplots(1,2,figsize=(13,5))
    for method in ['RAW','UNET_SINGLE_STANDARD','UNET_ALL_STANDARD','UNET_ALL_STRUCTURE_WEIGHTED']:
        rr=[r for r in summaries if r['method']==method and r['impairments'] in conds]
        by={r['impairments']:r for r in rr}
        axes[0].plot(conds,[float(by[c]['exact_decode_rate']) for c in conds],marker='o',label=method)
        axes[1].plot(conds,[float(by[c]['module_error_rate']) for c in conds],marker='o',label=method)
    axes[0].set_ylabel('Exact decode rate');axes[1].set_ylabel('Module error rate (MER)')
    for ax,title in zip(axes,['Stage 1 new-ID validation DR','Stage 1 new-ID validation MER']):ax.set_title(title);ax.grid(alpha=.25);ax.tick_params(axis='x',rotation=30)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(FIGS/'stage1_validation_dr_mer.png',dpi=170);plt.close(fig)
    return preds,summaries,contrasts

def evaluate_validation():
    # Read only locked validation rows and their 220 existing synthetic variants.
    mrows=read_csv(PRIOR/'data_manifest.csv');synth=read_csv(PRIOR/'research_next'/'synthetic_manifest.csv')
    valmap={(r['image_number'],r['condition']):r for r in mrows if r['split']=='validation'}
    valrefs=clean_reference_matrices(mrows,'validation')
    images=synthetic_archives();rows=[];visual={}
    models=[]
    for name in ('single_standard','all_standard','all_structure_weighted'):
        ck=torch.load(MODELS/f'stage2_{name}_best.pt',map_location='cpu',weights_only=False)
        mod=CompactUNet();mod.load_state_dict(ck['state_dict']);mod.eval();models.append((name,mod))
    # Existing pilot checkpoints are kept intact and evaluated as additional baselines.
    for source,name in [('unet_standard.pt','prior_standard_pilot'),('unet_structure_weighted.pt','prior_structure_weighted_pilot')]:
        ck=torch.load(PRIOR/'research_next'/'models'/source,map_location='cpu',weights_only=False)
        mod=CompactUNet();mod.load_state_dict(ck['state_dict']);mod.eval();models.append((name,mod))
    for m in synth:
        if m['split']!='validation' or int(m['severity_level'])==0: continue
        r=valmap[(m['image_number'],m['condition'])];payload=r['ground_truth_payload'];rawbytes=read_archive_image(images,m['relative_path']);rgb=pil_rgb_bytes(rawbytes)
        roi,geo=project_roi(rgb);ref=valrefs.get(r['capture_group_id']);matrix=ref['matrix'] if ref else None
        target=ideal_canvas(matrix) if matrix is not None else np.full((SIDE,SIDE),255,np.uint8)
        mask=fixed_pattern_mask(len(matrix)) if matrix is not None else np.zeros((SIDE,SIDE),np.float32)
        if roi is None:
            base={'identity_id':r['identity_id'],'image_number':r['image_number'],'condition':r['condition'],'severity_level':int(m['severity_level']),
                  'payload_sha256':sha256(payload.encode()),'geometry_ok':False,'roi':None,'target':target,'mask':mask,'raw':None}
            methods={}
            for method in ['RAW','CLASSICAL_ORIG','QR_AWARE']+[f'UNET_{x.upper()}' for x in ('single_standard','all_standard','all_structure_weighted','prior_standard_pilot','prior_structure_weighted_pilot')]:
                methods[method]={'decoded_payload':'','exact_decode':0,'false_decode':0,'latency_ms':0.,'module_errors':'','module_count':'','mer':'','fixed_pattern_errors':'','fixed_pattern_modules':'','fixed_pattern_mer':'','data_module_errors':'','data_modules':'','finder_errors':'','finder_modules':'','finder_mer':'','timing_errors':'','timing_modules':'','timing_mer':'','alignment_errors':'','alignment_modules':'','alignment_mer':''}
        else:
            # Direct baselines share the same rectified 256px ROI.
            method_imgs={};lat={}
            t=time.perf_counter();method_imgs['RAW']=roi;lat['RAW']=(time.perf_counter()-t)*1000
            t=time.perf_counter();method_imgs['CLASSICAL_ORIG']=classical_orig(roi,kernel=81,sigma=9,window=51,k=.18);lat['CLASSICAL_ORIG']=(time.perf_counter()-t)*1000
            t=time.perf_counter();method_imgs['QR_AWARE']=qr_aware(roi,window_factor=5,k=.18,use_anchor=True,use_guided=True)[0];lat['QR_AWARE']=(time.perf_counter()-t)*1000
            x=torch.from_numpy((roi.astype(np.float32)/255.)[None,None])
            for name,model in models:
                t=time.perf_counter()
                with torch.inference_mode():prob=model(x).numpy()[0,0]
                out=np.where(prob>=.5,0,255).astype(np.uint8);mname=f'UNET_{name.upper()}';method_imgs[mname]=out;lat[mname]=(time.perf_counter()-t)*1000
            methods={}
            for method,img in method_imgs.items():
                # The paired clean image provides a training target but not an independent matrix annotation.
                text,exact,false,errs,dt=decode_and_metrics(img,matrix,payload,False)
                methods[method]={'decoded_payload':text,'exact_decode':int(exact),'false_decode':int(false),'latency_ms':lat[method]+dt,
                    **{k:errs.get(k,'') for k in ('module_errors','module_count','mer','fixed_pattern_errors','fixed_pattern_modules','fixed_pattern_mer','data_module_errors','data_modules','finder_errors','finder_modules','finder_mer','timing_errors','timing_modules','timing_mer','alignment_errors','alignment_modules','alignment_mer')}}
            base={'identity_id':r['identity_id'],'image_number':r['image_number'],'condition':r['condition'],'severity_level':int(m['severity_level']),
                  'payload_sha256':sha256(payload.encode()),'geometry_ok':bool(geo['geometry_ok']),'roi':roi,'target':target,'mask':mask,'raw':method_imgs['RAW']}
            if base['geometry_ok']: base['methods']=method_imgs
        # RAW paired recovery/regression is computed on identical images and decoder policy.
        rawexact=int(methods['RAW']['exact_decode'])
        for method,vals in methods.items():
            exact=int(vals['exact_decode']); vals['recovery']=int(not rawexact and exact);vals['regression']=int(rawexact and not exact);vals['net_improvement']=vals['recovery']-vals['regression']
            rows.append({'split':'validation','identity_id':base['identity_id'],'image_number':base['image_number'],'condition':base['condition'],
                'severity_level':base['severity_level'],'illumination_factor':m.get('core_illumination_factor',''),'method':method,
                'payload_sha256':base['payload_sha256'],'geometry_ok':base['geometry_ok'],'exact_decode':exact,'raw_exact':rawexact,
                'matrix_reference_type':'paired clean-photo sampled matrix; not independent MER ground truth','mer_valid':0,
                'recovery':vals['recovery'],'regression':vals['regression'],'net_improvement':vals['net_improvement'],
                'false_decode':vals['false_decode'],'decoded_payload':vals['decoded_payload'],'latency_ms':vals['latency_ms'],
                **{k:vals.get(k,'') for k in ('module_errors','module_count','mer','fixed_pattern_errors','fixed_pattern_modules','fixed_pattern_mer','data_module_errors','data_modules','finder_errors','finder_modules','finder_mer','timing_errors','timing_modules','timing_mer','alignment_errors','alignment_modules','alignment_mer')}})
        visual[(base['identity_id'],base['image_number'],base['severity_level'])]=base
    write_csv(LOGS/'validation_predictions.csv',rows)
    summarize_eval(rows)
    make_visual_panels(visual,rows)
    return rows

def summarize_eval(rows):
    summaries=[]
    methods=sorted({r['method'] for r in rows})
    for method in methods:
      for sev in sorted({int(r['severity_level']) for r in rows}):
       for cond in ['ALL','S','DS','SG','DSG']:
        rr=[r for r in rows if r['method']==method and int(r['severity_level'])==sev and (cond=='ALL' or r['condition']==cond)]
        if not rr:continue
        ci=bootstrap_identity(rr,'exact_decode');netci=bootstrap_identity(rr,'net_improvement',SEED+1)
        valid=[r for r in rr if r['mer'] not in ('',None)]
        summaries.append({'split':'validation_synthetic_only','method':method,'severity_level':sev,'condition':cond,'n_images':len(rr),
          'n_identities':len({r['identity_id'] for r in rr}),'exact_decodes':sum(int(r['exact_decode']) for r in rr),
          'exact_decode_rate':np.mean([float(r['exact_decode']) for r in rr]),'exact_ci_identity_low':ci[0],'exact_ci_identity_high':ci[1],
          'recoveries':sum(int(r['recovery']) for r in rr),'regressions':sum(int(r['regression']) for r in rr),
          'net_improvement':sum(int(r['net_improvement']) for r in rr),'net_ci_identity_low':netci[0],'net_ci_identity_high':netci[1],
          'false_decodes':sum(int(r['false_decode']) for r in rr),'mean_latency_ms':np.mean([float(r['latency_ms']) for r in rr]),
          'p95_latency_ms':np.percentile([float(r['latency_ms']) for r in rr],95),
          'mer_valid_images':len(valid),'module_error_rate':(sum(int(x['module_errors']) for x in valid)/sum(int(x['module_count']) for x in valid)) if valid else '',
          'fixed_pattern_error_rate':(sum(int(x['fixed_pattern_errors']) for x in valid)/sum(int(x['fixed_pattern_modules']) for x in valid)) if valid else ''})
    write_csv(LOGS/'method_comparison_validation.csv',summaries)
    # Factorial contrasts by identity, paired within the same exact inputs.
    contrasts=[]
    for hname,a,b in [('H1_SINGLE_vs_ALL','UNET_SINGLE_STANDARD','UNET_ALL_STANDARD'),('H2_STANDARD_vs_STRUCTURE_WEIGHTED','UNET_ALL_STANDARD','UNET_ALL_STRUCTURE_WEIGHTED')]:
      for sev in sorted({int(r['severity_level']) for r in rows}):
       for cond in ['ALL','S','DS','SG','DSG']:
        aa={(r['identity_id'],r['image_number']):int(r['exact_decode']) for r in rows if r['method']==a and int(r['severity_level'])==sev and (cond=='ALL' or r['condition']==cond)}
        bb={(r['identity_id'],r['image_number']):int(r['exact_decode']) for r in rows if r['method']==b and int(r['severity_level'])==sev and (cond=='ALL' or r['condition']==cond)}
        common=sorted(set(aa)&set(bb)); d=np.array([bb[k]-aa[k] for k in common],float)
        ids=sorted({k[0] for k in common});rng=np.random.default_rng(SEED+sev+len(cond));bs=[]
        if ids:
          by={i:[bb[k]-aa[k] for k in common if k[0]==i] for i in ids}
          means=np.array([np.mean(by[i]) for i in ids]);bs=np.array([means[rng.integers(0,len(means),len(means))].mean() for _ in range(3000)])
          lo,hi=np.quantile(bs,[.025,.975])
        else: lo=hi=float('nan')
        contrasts.append({'hypothesis':hname,'baseline':a,'comparison':b,'severity_level':sev,'condition':cond,'n_images':len(common),'n_identities':len(ids),
          'exact_rate_baseline':np.mean([aa[k] for k in common]) if common else '', 'exact_rate_comparison':np.mean([bb[k] for k in common]) if common else '',
          'paired_rate_difference':np.mean(d) if len(d) else '', 'identity_bootstrap_95_low':lo,'identity_bootstrap_95_high':hi,
          'recoveries':int(np.sum(d>0)),'regressions':int(np.sum(d<0)),'ties':int(np.sum(d==0))})
    write_csv(LOGS/'hypothesis_contrasts_validation.csv',contrasts)

def make_visual_panels(visual,rows):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Select one deterministic, actual validation image per observed paired outcome category.
    outcomes={'successful_recovery':lambda a,b:not a and b,'unchanged_success':lambda a,b:a and b,
              'regression':lambda a,b:a and not b,'failure':lambda a,b:not a and not b}
    selected={}
    for outcome,fn in outcomes.items():
      for key,base in sorted(visual.items()):
        if not base.get('geometry_ok') or 'methods' not in base: continue
        # Visual selection uses saved exact flags directly; it does not call the decoder again.
        rawr=next((r for r in rows if r['identity_id']==base['identity_id'] and r['image_number']==base['image_number'] and int(r['severity_level'])==key[2] and r['method']=='RAW'),None)
        if rawr is None: continue
        raw=bool(int(rawr['exact_decode']))
        # For each method use U-NET ALL standard as representative restored output.
        comp=next((r for r in rows if r['identity_id']==base['identity_id'] and r['image_number']==base['image_number'] and int(r['severity_level'])==key[2] and r['method']=='UNET_ALL_STANDARD'),None)
        if comp is None: continue
        if fn(raw,bool(int(comp['exact_decode']))): selected[outcome]=(key,base,rawr,comp);break
    if not selected:
        (FIGS/'validation_outcome_panels.png').write_bytes(b'');return
    outcome_rows=[]
    methods_for_csv=['RAW','CLASSICAL_ORIG','QR_AWARE','UNET_SINGLE_STANDARD','UNET_ALL_STANDARD','UNET_ALL_STRUCTURE_WEIGHTED']
    for outcome,(key,base,rawr,com) in selected.items():
        matched={r['method']:r for r in rows if r['identity_id']==base['identity_id'] and r['image_number']==base['image_number'] and int(r['severity_level'])==key[2]}
        record={'outcome_vs_unet_all':outcome,'identity_id':key[0],'image_number':key[1],
                'severity_level':key[2],'condition':base['condition'],'matrix_reference_type':'paired Clean photo matrix; not independent MER ground truth'}
        for method in methods_for_csv:
            record[method+'_exact_decode']=int(matched[method]['exact_decode']) if method in matched else ''
            record[method+'_decoded_payload']=matched[method]['decoded_payload'] if method in matched else ''
        outcome_rows.append(record)
    write_csv(LOGS/'validation_example_outcomes.csv',outcome_rows)
    cols=['Paired Clean reference matrix','SYNTHETIC input ROI','Original classical','QR-aware classical','U-Net SINGLE','U-Net ALL','U-Net ALL + QR loss']
    fig,axes=plt.subplots(len(outcomes),len(cols),figsize=(21,12))
    for ri,(outcome,_) in enumerate(outcomes.items()):
      if outcome not in selected:
        for ax in axes[ri]:ax.axis('off')
        axes[ri,0].text(0,0.5,f'{outcome}: no validation example observed',fontsize=11);continue
      key,base,rawr,com=selected[outcome];arrs=base['methods']; imgs=[base['target'],base['roi'],arrs['CLASSICAL_ORIG'],arrs['QR_AWARE'],arrs['UNET_SINGLE_STANDARD'],arrs['UNET_ALL_STANDARD'],arrs['UNET_ALL_STRUCTURE_WEIGHTED']]
      for ci,(ax,img) in enumerate(zip(axes[ri],imgs)):
        ax.imshow(img,cmap='gray',vmin=0,vmax=255);ax.set_xticks([]);ax.set_yticks([])
        if ri==0:ax.set_title(cols[ci],fontsize=10)
      cond=base['condition']
      row_y=.865-ri*.235
      fig.text(.012,row_y,f'{outcome}\n{key[0]}\nimage {key[1]} {cond} L{key[2]}',ha='left',va='center',fontsize=8)
      outcome_label={'RAW':'RAW','CLASSICAL_ORIG':'classical','QR_AWARE':'QR-aware','UNET_SINGLE_STANDARD':'SINGLE','UNET_ALL_STANDARD':'ALL','UNET_ALL_STRUCTURE_WEIGHTED':'ALL + QR loss'}
      flags='; '.join(f"{outcome_label[m]}={'success' if int(next(r for r in rows if r['identity_id']==key[0] and r['image_number']==key[1] and int(r['severity_level'])==key[2] and r['method']==m)['exact_decode']) else 'fail'}" for m in methods_for_csv)
      caption=f"{flags}. Paired Clean matrix shown for reference; no independent Stage 2 MER."
      axes[ri,-1].text(.5,-.14,caption,transform=axes[ri,-1].transAxes,ha='center',va='top',fontsize=8,wrap=True)
    fig.suptitle('Synthetic validation examples — resized only for display; no pixel retouching',fontsize=14)
    fig.tight_layout(rect=[.14,.02,1,.97]);fig.savefig(FIGS/'validation_outcome_panels.png',dpi=150);plt.close(fig)
    zoom_outcomes=[x for x in outcomes if x in selected]
    if zoom_outcomes:
      zfig,zaxes=plt.subplots(len(zoom_outcomes),5,figsize=(15,3.1*len(zoom_outcomes)))
      if len(zoom_outcomes)==1:zaxes=np.array([zaxes])
      ztitles=['Clean-reference finder/timing','Synthetic input zoom','Classical QR-aware','U-Net ALL','U-Net ALL + QR loss']
      for ri,outcome in enumerate(zoom_outcomes):
        key,base,rawr,comp=selected[outcome];arrs=base['methods']
        ims=[base['target'],base['roi'],arrs['QR_AWARE'],arrs['UNET_ALL_STANDARD'],arrs['UNET_ALL_STRUCTURE_WEIGHTED']]
        for ci,(ax,img) in enumerate(zip(zaxes[ri],ims)):
          crop=img[:100,:100]
          zoom=cv2.resize(crop,(300,300),interpolation=cv2.INTER_NEAREST)
          ax.imshow(zoom,cmap='gray',vmin=0,vmax=255);ax.axis('off')
          if ri==0:ax.set_title(ztitles[ci],fontsize=9)
        zaxes[ri,0].set_ylabel(f'{outcome}\n{key[0]} L{key[2]}',fontsize=9)
      zfig.suptitle('Zoomed finder and timing comparison from the same validation examples',fontsize=13)
      zfig.tight_layout();zfig.savefig(FIGS/'validation_module_zoom.png',dpi=150);plt.close(zfig)

def main():
    import argparse
    global N_TRAIN_IDENTITIES,TRAIN_STEPS
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','data','stage1_data','real_pairs','train','finetune','evaluate','stage1_evaluate','samples'],default='all');ap.add_argument('--train-identities',type=int,default=N_TRAIN_IDENTITIES);ap.add_argument('--stage1-steps',type=int,default=TRAIN_STEPS);ap.add_argument('--stage2-steps',type=int,default=TRAIN_STEPS);args=ap.parse_args()
    N_TRAIN_IDENTITIES=args.train_identities;TRAIN_STEPS=args.stage1_steps
    for p in (DATA,MODELS,LOGS,FIGS,SAMPLES):p.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'python_torch':torch.__version__,'opencv':cv2.__version__,'seed':SEED,'test_images_read':0}),flush=True)
    if args.stage in ('all','data','stage1_data'):
        generate_stage1()
        if args.stage != 'stage1_data': build_real_pair_cache()
    if args.stage=='real_pairs': build_real_pair_cache()
    if args.stage=='finetune': finetune_all(args.stage2_steps)
    if args.stage in ('all','train'):
        if args.stage=='all': TRAIN_STEPS=args.stage1_steps
        train_all(args.stage1_steps,args.stage2_steps)
    if args.stage in ('all','evaluate'):evaluate_validation()
    if args.stage=='stage1_evaluate':evaluate_stage1_validation()
    if args.stage=='samples':save_training_samples()
    # Assert the code's locked split boundary remained satisfied.
    split=read_csv(PRIOR/'split_manifest.csv');testids={r['identity_id'] for r in split if r['split']=='test'}
    for p in (DATA/'stage2_pairs'/'stage2_pair_manifest.csv',):
        if p.exists():
            rr=read_csv(p);used={r['identity_id'] for r in rr if r['split'] in ('train','validation') and not r.get('skip_reason')}
            assert not used&testids
    print('COMPLETE: Stage 1/2 validation-only workflow finished; held-out test images were not opened.',flush=True)

if __name__=='__main__': main()
