"""Train/validate the proposal compact U-Net on real training identities only.

Shadow pairs are formed from training-split Clean captures and their pilot
synthetic variants. Validation uses only validation identities. Held-out test
files are never opened by this script.
"""
import argparse, csv, hashlib, json, random, sys, time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'research_next'; MODEL_DIR=OUT/'models'
sys.path.insert(0,str(ROOT/'src'))
import synthetic_shadow_pilot as pilot
from methods import load_fixed_frame,read_exact,gray8,_qr_quad,_finder_pitch
from unet_model import CompactUNet,proposal_loss
from unet_synthetic import fixed_pattern_mask

SEED=253112026; BATCH=8; STEPS=600; EVAL_EVERY=200; INPUT_SIDE=256

def readcsv(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def geometry(gray):
    q=_qr_quad(gray)
    if q is None:
        h,w=gray.shape;side=.52*min(h,w);cx=.57*w;cy=.55*h
        q=np.float32([[cx-side/2,cy-side/2],[cx+side/2,cy-side/2],[cx+side/2,cy+side/2],[cx-side/2,cy+side/2]])
        pitch=max(3.,side/49.);source='fixed_modules49_fallback'
    else:pitch,source=_finder_pitch(gray,q)
    side=float(np.median([np.linalg.norm(q[(j+1)%4]-q[j]) for j in range(4)]))
    modules=int(np.clip(round(side/max(pitch,1e-3)),21,177));modules=21+4*round((modules-21)/4)
    expansion=1.+8./modules # four-module quiet zone on each side
    # Keep the derived expansion independent of payload; it is the standard 4-module margin.
    center=q.mean(axis=0);qe=center+(q-center)*expansion
    dst=np.float32([[0,0],[INPUT_SIDE-1,0],[INPUT_SIDE-1,INPUT_SIDE-1],[0,INPUT_SIDE-1]])
    H=cv2.getPerspectiveTransform(qe.astype(np.float32),dst)
    return qe,H,modules,pitch,source

def extract_patch(frame,geo):
    qe,H,modules,pitch,source=geo
    g=gray8(frame)
    patch=cv2.warpPerspective(g,H,(INPUT_SIDE,INPUT_SIDE),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    # Geometry-only fixed pattern mask: no payload or test-label information.
    mask=fixed_pattern_mask(modules,quiet=4)
    mask=cv2.resize(mask,(INPUT_SIDE,INPUT_SIDE),interpolation=cv2.INTER_NEAREST)
    return patch.astype(np.uint8),mask.astype(np.float32)

def build_arrays(rows,manifest,split,levels):
    bynum={r['image_number']:r for r in rows};mm={}
    for m in manifest:
        if m['split']==split and int(m['severity_level']) in levels:mm.setdefault(m['image_number'],{})[int(m['severity_level'])]=m
    samples=[]
    chosen=[r for r in rows if r['split']==split and (split!='train' or r['condition']=='Clean')]
    for idx,r in enumerate(chosen,1):
        ims=mm.get(r['image_number'],{})
        if split=='train' and set(levels)-set(ims):continue
        src=pilot.rgb(pilot.source_path(r));clean_frame=pilot.fixed(src);geo=geometry(gray8(clean_frame));target,_=extract_patch(clean_frame,geo)
        for level in levels:
            m=ims.get(level)
            if m is None:continue
            p=pilot.source_path(r) if int(level)==0 else pilot.ROOT/m['relative_path']
            degraded=pilot.fixed(pilot.rgb(p));x,_=extract_patch(degraded,geo)
            samples.append({'input':x.astype(np.float32)/255.,'target':1.-target.astype(np.float32)/255.,
                'mask':extract_patch(clean_frame,geo)[1], 'identity_id':r['identity_id'],'image_number':r['image_number'],
                'condition':r['condition'],'severity_level':level,'factor':float(m['core_illumination_factor']),
                'geometry_source':geo[-1],'module_count':geo[2]})
        if idx%20==0:print(f'prepared {split} sources {idx}/{len(chosen)}',flush=True)
    return samples

class PairDataset(Dataset):
    def __init__(self,samples,augment=False):self.samples=samples;self.augment=augment
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i];x=s['input'].copy();y=s['target'].copy();m=s['mask'].copy()
        if self.augment:
            k=np.random.randint(4);x=np.rot90(x,k).copy();y=np.rot90(y,k).copy();m=np.rot90(m,k).copy()
            if np.random.rand()<.5:x=x[:,::-1].copy();y=y[:,::-1].copy();m=m[:,::-1].copy()
            if np.random.rand()<.5:x=x[::-1,:].copy();y=y[::-1,:].copy();m=m[::-1,:].copy()
        return tuple(torch.from_numpy(a[None].astype(np.float32)) for a in (x,y,m))

def validation_loss(model,loader,weighted,device):
    model.eval();losses=[]
    with torch.inference_mode():
        for x,y,m in loader:
            x=x.to(device);y=y.to(device);m=m.to(device);p=model(x);loss=proposal_loss(p,y,m if weighted else None)
            losses.extend([float(loss.detach().cpu())]*len(x))
    return float(np.mean(losses))

def train_variant(name,train_samples,val_samples,steps=STEPS,seed=SEED,eval_every=EVAL_EVERY):
    weighted=name=='structure_weighted';torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    device='cuda' if torch.cuda.is_available() else 'cpu';model=CompactUNet().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    val_loader=DataLoader(PairDataset(val_samples),batch_size=8,shuffle=False,num_workers=0)
    sampler=np.random.default_rng(seed+1);best=float('inf');best_step=0;log=[];ttrain=time.perf_counter()
    # Fixed-size random batches from an identity-disjoint training pair pool.
    for step in range(1,steps+1):
        inds=sampler.integers(0,len(train_samples),size=BATCH)
        batch=[train_samples[j] for j in inds]
        x=[];y=[];mask=[]
        for s in batch:
            a,b,c=PairDataset([s],augment=True)[0];x.append(a);y.append(b);mask.append(c)
        x=torch.stack(x).to(device);y=torch.stack(y).to(device);m=torch.stack(mask).to(device)
        model.train();opt.zero_grad(set_to_none=True);pred=model(x);loss=proposal_loss(pred,y,m if weighted else None);loss.backward();opt.step()
        if step<=3 or step%25==0:print(json.dumps({"variant":name,"step":step,"loss":float(loss.detach().cpu())}),flush=True)
        if step%eval_every==0 or step==steps:
            vl=validation_loss(model,val_loader,weighted,device)
            row={'variant':name,'step':step,'train_loss':float(loss.detach().cpu()),'validation_loss':vl,
                 'weighted_loss':weighted,'device':device,'elapsed_s':time.perf_counter()-ttrain}
            log.append(row);writecsv(OUT/f'unet_{name}_training_log.csv',log);print(json.dumps(row),flush=True)
            if vl<best:
                best=vl;best_step=step
                torch.save({'state_dict':model.state_dict(),'variant':name,'seed':seed,'best_step':step,
                    'validation_loss':vl,'training_identity_count':len({s['identity_id'] for s in train_samples}),
                    'training_pair_count':len(train_samples),'split_sha256':hashlib.sha256((ROOT/'split_manifest.csv').read_bytes()).hexdigest(),
                    'architecture':'CompactUNet widths 16/32/64/128','input':'rectified grayscale 256x256',
                    'output':'black-module probability','loss':'weighted BCE+gradient' if weighted else 'BCE+gradient'},
                    MODEL_DIR/f'unet_{name}.pt')
    writecsv(OUT/f'unet_{name}_training_log.csv',log)
    info={'variant':name,'device':device,'steps':steps,'best_step':best_step,'best_validation_loss':best,
        'training_pairs':len(train_samples),'training_identities':len({s['identity_id'] for s in train_samples}),
        'validation_pairs':len(val_samples),'validation_identities':len({s['identity_id'] for s in val_samples}),'seed':seed,'pytorch':torch.__version__}
    return info

def unwarp_prediction(frame,geo,black_probability):
    qe,H,modules,pitch,source=geo
    gray=gray8(frame);predgray=np.clip(255*(1.-black_probability),0,255).astype(np.uint8)
    dst=np.float32([[0,0],[INPUT_SIDE-1,0],[INPUT_SIDE-1,INPUT_SIDE-1],[0,INPUT_SIDE-1]])
    Hback=cv2.getPerspectiveTransform(dst,qe.astype(np.float32));h,w=gray.shape
    patch=cv2.warpPerspective(predgray,Hback,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    mask=cv2.warpPerspective(np.full((INPUT_SIDE,INPUT_SIDE),255,np.uint8),Hback,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    alpha=cv2.GaussianBlur(mask.astype(np.float32)/255.,(0,0),2.)[...,None]
    out=gray.astype(np.float32)*(1-alpha[...,0])+patch.astype(np.float32)*alpha[...,0]
    return np.clip(out,0,255).astype(np.uint8)

def load_model(path,device):
    obj=torch.load(path,map_location=device,weights_only=False);net=CompactUNet().to(device);net.load_state_dict(obj['state_dict']);net.eval();return net

def eval_models(rows,manifest,model_names=('standard','structure_weighted')):
    device='cuda' if torch.cuda.is_available() else 'cpu';samples=build_arrays(rows,manifest,'validation',[0,1,2,3,4])
    # Convert validation metadata back to the source image and matching synthetic path.
    bynum={r['image_number']:r for r in rows};mmap={(m['image_number'],int(m['severity_level'])):m for m in manifest if m['split']=='validation'}
    rawmap={(x['image_number'],int(x['severity_level'])):x for x in readcsv(OUT/'method_predictions_validation.csv') if x['method']=='RAW'}
    preds=[]
    for name in model_names:
        model=load_model(MODEL_DIR/f'unet_{name}.pt',device)
        # Warm up outside measured sample latency.
        with torch.inference_mode():model(torch.zeros((1,1,INPUT_SIDE,INPUT_SIDE),device=device))
        for i,s in enumerate(samples,1):
            r=bynum[s['image_number']];m=mmap[(r['image_number'],s['severity_level'])]
            path=pilot.source_path(r) if s['severity_level']==0 else pilot.ROOT/m['relative_path']
            frame=pilot.fixed(pilot.rgb(path));t0=time.perf_counter();geo=geometry(gray8(frame));patch,_=extract_patch(frame,geo)
            x=torch.from_numpy((patch.astype(np.float32)/255.)[None,None]).to(device)
            with torch.inference_mode():black=model(x)[0,0].detach().cpu().numpy()
            restored=unwarp_prediction(frame,geo,black);text=read_exact(restored);elapsed=time.perf_counter()-t0
            gt=(r.get('ground_truth_payload') or '').strip();exact=int(bool(gt) and text==gt);raw=rawmap[(r['image_number'],s['severity_level'])];rawok=int(raw['exact_decode'])
            preds.append({'split':'validation','identity_id':r['identity_id'],'capture_group_id':r['capture_group_id'],
                'image_number':r['image_number'],'condition':r['condition'],'severity_level':s['severity_level'],'factor':s['factor'],
                'method':'UNET_'+name.upper(),'exact_decode':exact,'decoded_nonempty':int(bool(text)),
                'false_decode':int(bool(text) and text!=gt),'raw_exact':rawok,'recovery':int(not rawok and exact),
                'regression':int(rawok and not exact),'net_improvement':exact-rawok,'decoder_calls':1,
                'end_to_end_latency_s':elapsed,'decoded_payload':text,'patch_geometry_source':s['geometry_source']})
            if i%50==0:print(f'{name} validation {i}/{len(samples)}',flush=True)
    writecsv(OUT/'unet_predictions_validation.csv',preds)
    summary=[]
    for name in ('UNET_STANDARD','UNET_STRUCTURE_WEIGHTED'):
      for level in range(5):
       for cond in ('ALL','Clean','S','DS','SG','DSG'):
        rr=[x for x in preds if x['method']==name and x['severity_level']==level and (cond=='ALL' or x['condition']==cond)]
        if not rr:continue
        def ci(key):
            by=defaultdict(list)
            for x in rr:by[x['identity_id']].append(float(x[key]))
            vals=np.array([np.mean(by[k]) for k in sorted(by)]);rng=np.random.default_rng(SEED);b=np.array([vals[rng.integers(0,len(vals),len(vals))].mean() for _ in range(10000)])
            return np.quantile(b,[.025,.975])
        ci0=ci('exact_decode');cin=ci('net_improvement')
        summary.append({'split':'validation','method':name,'severity_level':level,'factor':{0:1,1:.7,2:.5,3:.3,4:.2}[level],
            'condition':cond,'n_images':len(rr),'n_identities':len({x['identity_id'] for x in rr}),
            'exact_decodes':sum(x['exact_decode'] for x in rr),'exact_decode_rate':np.mean([x['exact_decode'] for x in rr]),
            'exact_ci_identity_low':ci0[0],'exact_ci_identity_high':ci0[1],
            'recoveries':sum(x['recovery'] for x in rr),'regressions':sum(x['regression'] for x in rr),
            'net_improvement':sum(x['net_improvement'] for x in rr),'net_ci_identity_low':cin[0],'net_ci_identity_high':cin[1],
            'false_decodes':sum(x['false_decode'] for x in rr),'mean_end_to_end_ms':1000*np.mean([x['end_to_end_latency_s'] for x in rr]),
            'median_end_to_end_ms':1000*np.median([x['end_to_end_latency_s'] for x in rr]),'p95_end_to_end_ms':1000*np.percentile([x['end_to_end_latency_s'] for x in rr],95)})
    writecsv(OUT/'unet_comparison_validation.csv',summary)
    return preds,summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=STEPS);ap.add_argument('--eval-every',type=int,default=EVAL_EVERY);ap.add_argument('--variant',choices=['both','standard','structure_weighted'],default='both');args=ap.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    OUT.mkdir(parents=True,exist_ok=True);MODEL_DIR.mkdir(parents=True,exist_ok=True)
    rows=pilot.load_rows();manifest=readcsv(ROOT/'synthetic_pilot/synthetic_manifest.csv')
    train=build_arrays(rows,manifest,'train',[1,2,3,4])
    validation=build_arrays(rows,manifest,'validation',[1,2,3,4])
    if {s['identity_id'] for s in train}&{s['identity_id'] for s in validation}:raise RuntimeError('Identity leakage')
    if any(s['identity_id'] in {r['identity_id'] for r in rows if r['split']=='test'} for s in train+validation):raise RuntimeError('Held-out identity leakage')
    print(json.dumps({'train_pairs':len(train),'train_identities':len({s['identity_id'] for s in train}),'validation_pairs':len(validation),'validation_identities':len({s['identity_id'] for s in validation}),'steps':args.steps}),flush=True)
    common={'seed':SEED,'steps':args.steps,'eval_every':args.eval_every}
    variants=('standard','structure_weighted') if args.variant=='both' else (args.variant,)
    infos=[train_variant(n,train,validation,**common) for n in variants]
    (OUT/'unet_training_summary.json').write_text(json.dumps({'variants':infos,'pytorch':torch.__version__,'test_identities_opened':0,
        'pair_source':'train Clean captures shadowed using pilot factors; validation pair selection uses val identities only',
        'loss_definition':'proposal weighted BCE plus gradient L1; structure mask from fixed finder/timing/alignment geometry only'},indent=2))
    eval_models(rows,manifest)
    print(json.dumps(infos,indent=2))

if __name__=='__main__':main()
