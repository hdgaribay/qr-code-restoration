#!/usr/bin/env python3
"""Controlled Stage 2 repair ablation with exact digital pixel alignment."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,math,random,time,sys
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'source/experiment'
sys.path.insert(0,str(EXP/'src'));sys.path.insert(0,str(ROOT/'source/prior_project/ArtifactB_Next_Research/src'))
from unet_model import CompactUNet,proposal_loss
from methods import read_exact
import stage1_stage2 as legacy

def readcsv(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);fields=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def matrix_from_canvas(canvas,n):
 full=n+8;bits=np.zeros((n,n),np.uint8)
 for y in range(n):
  cy=int(round((y+4.5)*256/full))
  for x in range(n):
   cx=int(round((x+4.5)*256/full));bits[y,x]=1 if canvas[cy,cx]<128 else 0
 return bits
def render_matrix(mat):
 n=len(mat);c=np.ones((n+8,n+8),np.uint8)*255;c[4:4+n,4:4+n]=np.where(mat>0,0,255).astype(np.uint8)
 return cv2.resize(c,(256,256),interpolation=cv2.INTER_NEAREST)
def warp_geometry(clean,mask,rng):
 side=256;src=np.array([[0,0],[255,0],[255,255],[0,255]],np.float32);center=np.array([127.5,127.5])
 angle=float(rng.uniform(-4.0,4.0));scale=float(rng.uniform(.965,1.025));rad=np.deg2rad(angle);R=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
 rot=(src-center)@R.T*scale+center
 jitter=rng.uniform(-2.5,2.5,(4,2)).astype(np.float32);dst=rot+jitter
 H=cv2.getPerspectiveTransform(src,dst.astype(np.float32))
 y=cv2.warpPerspective(clean,H,(side,side),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
 m=cv2.warpPerspective(mask.astype(np.uint8),H,(side,side),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
 return y,m,H,angle,scale

def shadow_params(rng,shape,level):
 h,w=shape;theta=float(rng.uniform(0,180));coverage=float(rng.uniform(.28,.72));pen=float(rng.uniform(3.0,14.0));factor={2:.50,3:.30,4:.20}[level]
 yy,xx=np.mgrid[0:h,0:w].astype(np.float32);nx,ny=np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta));proj=(xx-(w-1)/2)*nx+(yy-(h-1)/2)*ny
 threshold=float(np.quantile(proj,1-coverage));d=proj-threshold
 alpha=np.clip(.5+d/(2*pen),0,1);alpha=alpha*alpha*(3-2*alpha)
 illum=1-(1-factor)*alpha
 return illum,theta,factor,float(np.mean(alpha>=.5)),pen

def degrade(clean,illum,condition,rng):
 # Weak smooth surface illumination, followed by the controlled multiplicative cast shadow.
 h,w=clean.shape;yy,xx=np.mgrid[0:h,0:w].astype(np.float32);gx=(xx-w/2)/w;gy=(yy-h/2)/h
 surface=1+float(rng.uniform(-.06,.06))*gx+float(rng.uniform(-.06,.06))*gy
 a=clean.astype(np.float32)*surface*illum
 if 'D' in condition:
  sigma=float(rng.uniform(.45,1.15));k=int(np.ceil(sigma*4))|1;a=cv2.GaussianBlur(a,(k,k),sigma,borderType=cv2.BORDER_REFLECT)
 else:sigma=0.0
 if 'G' in condition:
  cx=float(rng.uniform(.18,.82))*w;cy=float(rng.uniform(.18,.82))*h;sx=float(rng.uniform(.04,.14))*w;sy=float(rng.uniform(.04,.14))*h;amp=float(rng.uniform(18,55))
  a+=amp*np.exp(-.5*(((xx-cx)/sx)**2+((yy-cy)/sy)**2))
 else:amp=0.0
 noise=float(rng.uniform(.4,2.0));a+=rng.normal(0,noise,(h,w));a=np.clip(a,0,255).astype(np.uint8)
 quality=int(rng.integers(48,96));ok,buf=cv2.imencode('.jpg',a,[cv2.IMWRITE_JPEG_QUALITY,quality]);x=cv2.imdecode(buf,cv2.IMREAD_GRAYSCALE) if ok else a
 return x,{'defocus_sigma':sigma,'glare_amplitude':amp,'sensor_noise_std':noise,'jpeg_quality':quality}

def get_identity_matrices():
 # Reconstruct QR module bits only from audited Stage 2 train-pair targets; no photographed pixels enter new training.
 train=np.load(EXP/'data/stage2_pairs/train.npz');y=train['y'];rows=[r for r in readcsv(EXP/'data/stage2_pairs/stage2_pair_manifest.csv') if r['split']=='train' and r.get('pair_index')]
 idm={}
 for r in rows:
  ident=f"{r['identity_id']}|{r['capture_group_id']}";n=int(r['module_count']);mat=matrix_from_canvas(y[int(r['pair_index'])],n)
  if ident in idm and not np.array_equal(idm[ident][0],mat):raise RuntimeError('inconsistent training QR matrix within capture group')
  idm[ident]=(mat,r['payload_sha256'],r['identity_id'],r['capture_group_id'])
 val=np.load(EXP/'data/stage1_all_validation.npz');vy=val['y'];vrows=[r for r in readcsv(EXP/'data/stage1_pairs_manifest.csv') if r['split']=='validation']
 vmap={}
 for i,r in enumerate(vrows):
  ident=r['synthetic_id'];n=int(r['module_count']);mat=matrix_from_canvas(vy[i],n)
  if ident not in vmap:vmap[ident]=(mat,r['payload'],r['payload_sha256'])
  elif not np.array_equal(vmap[ident][0],mat):raise RuntimeError('inconsistent validation QR matrix')
 if len(idm)!=37 or len(vmap)!=200:raise RuntimeError(f'expected 37 train matrix units/200 validation IDs, got {len(idm)}/{len(vmap)}')
 return idm,vmap

def generate_pool(pool,prefix,seed,validation=False):
 rng=np.random.default_rng(seed);conditions=['S','DS','SG','DSG'];levels=[2,3,4];xlist=[];ylist=[];mlist=[];manifest=[]
 for identity_idx,(ident,entry) in enumerate(sorted(pool.items())):
  matrix=entry[0];n=matrix.shape[0];clean=render_matrix(matrix);mask=legacy.fixed_pattern_mask(n)
  for level in levels:
   geom_count=1 if validation else 4
   for geom in range(geom_count):
    geom_seed=int(rng.integers(0,2**32-1));gr=np.random.default_rng(geom_seed);y,m,H,angle,scale=warp_geometry(clean,mask,gr)
    sr=np.random.default_rng(geom_seed+431);illum,theta,factor,coverage,pen=shadow_params(sr,(256,256),level)
    for rep,condition in enumerate(conditions):
     if not validation and prefix=='SINGLE': condition='S'
     dr=np.random.default_rng(geom_seed+rep*10007+991)
     x,aug=degrade(y,illum,condition,dr)
     mid=f'{prefix}_{ident}_L{level}_G{geom}_R{rep}'
     xlist.append(x);ylist.append(y);mlist.append(m)
     manifest.append({'split':'validation' if validation else 'train','branch':prefix,'synthetic_id':ident,'source_identity_id':(entry[2] if not validation else ident),'capture_group_id':(entry[3] if not validation else ''),'payload_sha256':(entry[2] if validation else entry[1]),'pair_id':mid,'module_count':n,'qr_version':(n-17)//4,'severity_level':level,'illumination_factor':factor,'shadow_angle_deg':theta,'shadow_coverage_fraction':coverage,'penumbra_px':pen,'condition':condition,'geometry_rotation_deg':angle,'geometry_scale':scale,'transform_matrix':json.dumps(H.round(7).tolist()),'defocus_sigma':aug['defocus_sigma'],'glare_amplitude':aug['glare_amplitude'],'sensor_noise_std':aug['sensor_noise_std'],'jpeg_quality':aug['jpeg_quality'],'x_sha256':digest(x),'y_sha256':digest(y),'target_alignment':'same known digital homography applied to clean target and degraded input'})
 return np.stack(xlist),np.stack(ylist),np.stack(mlist),manifest

def save_npz(path,x,y,m):np.savez_compressed(path,x=x,y=y,mask=m.astype(np.uint8))
def val_loss(model,x,y,m,weighted):
 model.eval();vals=[]
 with torch.inference_mode():
  for i in range(0,len(x),16):
   a=torch.from_numpy(x[i:i+16].astype(np.float32)/255.)[:,None];b=1-torch.from_numpy(y[i:i+16].astype(np.float32)/255.)[:,None];c=torch.from_numpy(m[i:i+16].astype(np.float32))[:,None]
   vals.append(float(proposal_loss(model(a),b,c if weighted else None)))
 return float(np.mean(vals))
def train_branch(name,trainpath,valpath,initpath,weighted,run_seed,steps=240):
 x,y,mask=[np.load(trainpath)[k] for k in ['x','y','mask']];vx,vy,vm=[np.load(valpath)[k] for k in ['x','y','mask']]
 ck=torch.load(initpath,map_location='cpu',weights_only=False);model=CompactUNet();model.load_state_dict(ck['state_dict']);model.train();torch.set_num_threads(min(4,torch.get_num_threads()));torch.manual_seed(run_seed);np.random.seed(run_seed);random.seed(run_seed)
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4);rng=np.random.default_rng(run_seed+17);aug_rng=np.random.default_rng(run_seed+31);logs=[];best=1e9;outp=ROOT/'models'/f'{name}_best.pt';outp.parent.mkdir(parents=True,exist_ok=True);started=time.time()
 for step in range(1,steps+1):
  ids=rng.integers(0,len(x),size=8);xb=[];yb=[];mb=[]
  for idx in ids:
   a=x[idx].astype(np.float32)/255.;b=1-y[idx].astype(np.float32)/255.;c=mask[idx].astype(np.float32);k=int(aug_rng.integers(0,4));a=np.rot90(a,k).copy();b=np.rot90(b,k).copy();c=np.rot90(c,k).copy()
   if aug_rng.random()<.5:a=a[:,::-1].copy();b=b[:,::-1].copy();c=c[:,::-1].copy()
   xb.append(a);yb.append(b);mb.append(c)
  xt=torch.from_numpy(np.stack(xb)[:,None]);yt=torch.from_numpy(np.stack(yb)[:,None]);mt=torch.from_numpy(np.stack(mb)[:,None]);opt.zero_grad(set_to_none=True);prob=model(xt);loss=proposal_loss(prob,yt,mt if weighted else None);loss.backward();opt.step()
  if step%80==0 or step==steps:
   vl=val_loss(model,vx,vy,vm,weighted);rec={'model':name,'step':step,'train_loss':float(loss.detach()),'validation_loss':vl,'weighted_loss':weighted,'learning_rate':1e-4,'seed':run_seed,'initial_checkpoint':str(initpath.relative_to(ROOT)),'pairs':len(x),'validation_pairs':len(vx),'elapsed_seconds':time.time()-started};logs.append(rec)
   if vl<best:
    best=vl;torch.save({'state_dict':model.state_dict(),'model':name,'step':step,'validation_loss':vl,'seed':run_seed,'architecture':'4-scale Compact U-Net widths 16/32/64/128; grayscale 256x256','initial_checkpoint':str(initpath.relative_to(ROOT)),'train_pair_alignment':'fully digital; exact shared warp applied to input and target','validation_identities':200},outp)
   print(json.dumps({'event':'validation','record':rec,'best_loss':best}),flush=True)
  if step%40==0:print(json.dumps({'event':'progress','model':name,'step':step,'steps':steps}),flush=True)
 writecsv(ROOT/'results'/f'{name}_training_log.csv',logs)
 return outp,logs

def main():
 torch.set_num_threads(min(4,torch.get_num_threads()));base_seed=7253325
 idm,valm=get_identity_matrices();train_s=generate_pool(idm,'SINGLE',base_seed);train_a=generate_pool(idm,'ALL',base_seed);val=generate_pool(valm,'VALIDATION',base_seed+1,validation=True)
 assert len(train_s[0])==len(train_a[0])==1776 and len(val[0])==2400
 save_npz(ROOT/'data/repair_train_single.npz',*train_s[:3]);save_npz(ROOT/'data/repair_train_all.npz',*train_a[:3]);save_npz(ROOT/'data/known_geometry_validation.npz',*val[:3])
 writecsv(ROOT/'data/repair_train_manifest.csv',train_s[3]+train_a[3]);writecsv(ROOT/'data/known_geometry_validation_manifest.csv',val[3])
 meta={'base_seed':base_seed,'train_matrix_target_units':len(idm),'source_train_identity_count':36,'train_pairs_per_arm':len(train_s[0]),'validation_identity_count':len(valm),'validation_pairs':len(val[0]),'severity_levels':{'2':0.50,'3':0.30,'4':0.20},'conditions':['S','DS','SG','DSG'],'matched_identity_pool':True,'matched_pairs_per_arm':True,'matched_steps':240,'architecture':'4-scale Compact U-Net 16/32/64/128','batch_size':8,'learning_rate':0.0001,'validation_selection':'Stage 1 pre-existing 200 synthetic validation identities, regenerated with known digital geometry; no Stage2 pixels used for pixel loss selection','train_source':'37 unique capture-group/matrix targets derived only from the 36 audited existing Stage2 training identities; original photo bytes not read','target_alignment':'known shared digital transform matrix applied to target and degraded input; target pixel grid exact'}
 (ROOT/'results/repair_training_config.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'data_generated':meta}),flush=True)
 init=EXP/'models/stage1_all_standard_best.pt'
 for name,path,weighted,seed in [('repair_stage2_single','repair_train_single.npz',False,base_seed+100),('repair_stage2_all','repair_train_all.npz',False,base_seed+100),('repair_stage2_all_weighted','repair_train_all.npz',True,base_seed+100)]:
  train_branch(name,ROOT/'data'/path,ROOT/'data/known_geometry_validation.npz',init,weighted,seed,steps=240)
if __name__=='__main__':main()
