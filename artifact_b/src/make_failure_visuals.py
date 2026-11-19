#!/usr/bin/env python3
from pathlib import Path
import csv,sys,json
import numpy as np,torch,cv2
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'source/experiment';sys.path.insert(0,str(ROOT/'source/prior_project/ArtifactB_Next_Research/src'))
from unet_model import CompactUNet
from methods import classical_orig,qr_aware

def readcsv(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def writecsv(p, rows):
 if not rows:return
 with Path(p).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def loadmodel(p):
 ck=torch.load(p,map_location='cpu',weights_only=False);m=CompactUNet();m.load_state_dict(ck['state_dict']);m.eval();return m
def netout(m,im):
 with torch.inference_mode():p=m(torch.from_numpy((im.astype(np.float32)/255.)[None,None])).numpy()[0,0]
 return np.where(p>=.5,0,255).astype(np.uint8)
def main():
 preds=readcsv(ROOT/'results/all_stage2_comparison_predictions.csv');rawmap={(r['identity_id'],r['image_number'],int(r['severity_level'])):r for r in preds if r['method']=='RAW'}
 compmap={(r['identity_id'],r['image_number'],int(r['severity_level']),r['method']):r for r in preds}
 specs=[('recovery','REPAIR_ALL',lambda r:int(r['raw_exact'])==0 and int(r['exact_decode'])==1,3),
        ('regression','UNET_ALL_STANDARD',lambda r:int(r['raw_exact'])==1 and int(r['exact_decode'])==0,3),
        ('unchanged_success','REPAIR_ALL',lambda r:int(r['raw_exact'])==1 and int(r['exact_decode'])==1,2),
        ('failure','REPAIR_ALL',lambda r:int(r['raw_exact'])==0 and int(r['exact_decode'])==0,4)]
 selected=[]
 for label,method,predicate,lev in specs:
  candidates=sorted((r for r in preds if r['method']==method and int(r['severity_level'])==lev and predicate(r)),key=lambda r:(r['condition'],r['identity_id'],int(r['image_number'])))
  if label=='failure':
   candidates=[r for r in candidates if all(int(compmap[(r['identity_id'],r['image_number'],lev,m)]['exact_decode'])==0 for m in ['CLASSICAL_ORIG','QR_AWARE','STAGE1_ALL_STANDARD','UNET_ALL_STANDARD','UNET_ALL_STRUCTURE_WEIGHTED','REPAIR_SINGLE','REPAIR_ALL_WEIGHTED'])] or candidates
  if candidates:selected.append((label,candidates[0]))
 # map rows to stored arrays, with the original 220-image order.
 manifest=[r for r in readcsv(EXP/'data/stage2_pairs/stage2_pair_manifest.csv') if r['split']=='validation']
 data=np.load(EXP/'data/stage2_pairs/validation.npz');x,y=data['x'],data['y']
 outputs=np.load(ROOT/'data/stage2_repair_outputs.npz')['outputs'];
 models={
  'STAGE1_ALL_STANDARD':loadmodel(EXP/'models/stage1_all_standard_best.pt'),
  'UNET_ALL_STANDARD':loadmodel(EXP/'models/stage2_all_standard_best.pt'),
  'UNET_ALL_STRUCTURE_WEIGHTED':loadmodel(EXP/'models/stage2_all_structure_weighted_best.pt')}
 columns=['SYNTHETIC input ROI','Paired Clean matrix*','RAW','Original Classical','QR-aware Classical','Stage 1 ALL','Stage 2 ALL','Stage 2 weighted','Repair SINGLE','Repair ALL','Repair ALL weighted']
 outcome_rows=[];panel=[]
 output_methods=['RAW','CLASSICAL_ORIG','QR_AWARE','STAGE1_ALL_STANDARD','UNET_ALL_STANDARD','UNET_ALL_STRUCTURE_WEIGHTED','REPAIR_SINGLE','REPAIR_ALL','REPAIR_ALL_WEIGHTED']
 aliases={'Stage 1 ALL':'STAGE1_ALL_STANDARD','Stage 2 ALL':'UNET_ALL_STANDARD','Stage 2 weighted':'UNET_ALL_STRUCTURE_WEIGHTED','Repair SINGLE':'REPAIR_SINGLE','Repair ALL':'REPAIR_ALL','Repair ALL weighted':'REPAIR_ALL_WEIGHTED'}
 for label,r in selected:
  key=(r['identity_id'],r['image_number'],int(r['severity_level']));idx=next(i for i,m in enumerate(manifest) if (m['identity_id'],m['image_number'],int(m['severity_level']))==key);m=manifest[idx];pi=int(m['pair_index']) if m.get('pair_index') else None
  im=x[pi] if pi is not None else np.full((256,256),255,np.uint8);ref=y[pi] if pi is not None else np.full((256,256),255,np.uint8)
  ims={'SYNTHETIC input ROI':im,'Paired Clean matrix*':ref,'RAW':im}
  ims['Original Classical']=classical_orig(im,kernel=81,sigma=9,window=51,k=.18)
  ims['QR-aware Classical']=qr_aware(im,window_factor=5,k=.18,use_anchor=True,use_guided=True)[0]
  for method,net in models.items():ims[method]=netout(net,im)
  for mi,method in enumerate(['REPAIR_SINGLE','REPAIR_ALL','REPAIR_ALL_WEIGHTED']):ims[method]=outputs[mi,idx]
  row={'display_case':label,'identity_id':key[0],'image_number':key[1],'condition':m['condition'],'severity_level':key[2],'geometry_ok':m['geometry_ok'],'paired_clean_reference':'matrix content checked only; pixel alignment not established'}
  for method in output_methods:
   v=compmap[(key[0],key[1],key[2],method)];row[method+'_exact_decode']=v['exact_decode'];row[method+'_false_decode']=v['false_decode']
  outcome_rows.append(row);panel.append((label,key,m,ims,row))
 writecsv(ROOT/'results/visual_example_outcomes.csv',outcome_rows)
 fig,axes=plt.subplots(len(panel),len(columns),figsize=(len(columns)*2.2,len(panel)*2.1))
 if len(panel)==1:axes=np.array([axes])
 for ci,c in enumerate(columns):axes[0,ci].set_title(c,fontsize=9)
 for ri,(label,key,m,ims,rec) in enumerate(panel):
  for ci,c in enumerate(columns):
   ax=axes[ri,ci];img=ims[aliases.get(c,c)];ax.imshow(img,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
   if ci>=2:
    method=output_methods[ci-2];v=compmap[(key[0],key[1],key[2],method)];status='FALSE' if int(v['false_decode']) else ('EXACT' if int(v['exact_decode']) else 'FAIL')
    ax.set_xlabel(status,fontsize=8,color=('#b42318' if status!='EXACT' else '#067647'))
  axes[ri,0].set_ylabel(f"{label}\n{key[0]} {m['condition']} L{key[2]}",fontsize=8)
 fig.suptitle('Stage 2 repair visual audit — same validation ROI; display scaling only',fontsize=14)
 fig.text(.5,.006,'* Paired Clean matrix verifies QR content, but independent pixel-level registration is unavailable; it is not a valid Stage 2 MER target. Outputs are actual model/classical results, not retouched.',ha='center',fontsize=9)
 fig.tight_layout(rect=[0,.025,1,.96]);fig.savefig(ROOT/'figures/stage2_repair_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
 # Same cases, enlarged exact pixels for the upper-left finder and timing line using nearest neighbor only.
 zcols=['Input finder','Classical finder','Stage 1 finder','Stage 2 finder','Repair ALL finder','Input timing','Classical timing','Stage 2 timing','Repair ALL timing']
 zfig,zaxes=plt.subplots(len(panel),len(zcols),figsize=(len(zcols)*2.1,len(panel)*2.0));
 if len(panel)==1:zaxes=np.array([zaxes])
 for ri,(label,key,m,ims,rec) in enumerate(panel):
  n=int(m['module_count']);pitch=256/(n+8);fy=int(round(7.5*pitch));fx=int(round(7.5*pitch));rad=max(8,int(round(5.5*pitch)))
  hcrop=(slice(max(0,fy-rad),min(256,fy+rad)),slice(max(0,fx-rad),min(256,fx+rad)))
  rowimgs=[ims['SYNTHETIC input ROI'][hcrop],ims['Original Classical'][hcrop],ims['STAGE1_ALL_STANDARD'][hcrop],ims['UNET_ALL_STANDARD'][hcrop],ims['REPAIR_ALL'][hcrop]]
  ty=int(round((4+6.5)*pitch));tx=int(round((4+6.5)*pitch));half=max(3,int(round(1.5*pitch)));tims=[ims[q][max(0,ty-half):min(256,ty+half),int(round(3*pitch)):int(round((n+5)*pitch))] for q in ['SYNTHETIC input ROI','Original Classical','UNET_ALL_STANDARD','REPAIR_ALL']]
  for ci,img in enumerate(rowimgs+tims):
   ax=zaxes[ri,ci];zoom=cv2.resize(img,(240,240),interpolation=cv2.INTER_NEAREST);ax.imshow(zoom,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.axis('off')
   if ri==0:ax.set_title(zcols[ci],fontsize=8)
  zaxes[ri,0].set_ylabel(label,fontsize=8)
 zfig.suptitle('Finder-pattern and horizontal timing-module crops; nearest-neighbor display enlargement',fontsize=13);zfig.tight_layout(rect=[0,0,1,.95]);zfig.savefig(ROOT/'figures/stage2_module_failure_zoom.png',dpi=180,bbox_inches='tight');plt.close(zfig)
 print('visual cases',[(x[0],x[1][0],x[1][1],x[1][2]) for x in panel])
if __name__=='__main__':main()
