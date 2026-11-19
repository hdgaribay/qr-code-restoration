#!/usr/bin/env python3
"""Create paper-ready Stage 2 validation plots from the saved CSV outputs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LOGS=ROOT/'logs'; FIGS=ROOT/'figures'; FIGS.mkdir(exist_ok=True)
def load(name):
    with (LOGS/name).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))

summary=load('method_comparison_validation.csv')
colors={'RAW':'#555555','CLASSICAL_ORIG':'#1f77b4','QR_AWARE':'#2ca02c','UNET_SINGLE_STANDARD':'#d62728',
        'UNET_ALL_STANDARD':'#9467bd','UNET_ALL_STRUCTURE_WEIGHTED':'#ff7f0e','UNET_PRIOR_STANDARD_PILOT':'#8c564b',
        'UNET_PRIOR_STRUCTURE_WEIGHTED_PILOT':'#e377c2'}
labels={'RAW':'RAW','CLASSICAL_ORIG':'Original classical','QR_AWARE':'QR-aware classical',
        'UNET_SINGLE_STANDARD':'U-Net SINGLE','UNET_ALL_STANDARD':'U-Net ALL',
        'UNET_ALL_STRUCTURE_WEIGHTED':'U-Net ALL + QR loss','UNET_PRIOR_STANDARD_PILOT':'Prior standard U-Net pilot',
        'UNET_PRIOR_STRUCTURE_WEIGHTED_PILOT':'Prior structure-weighted pilot'}
methods=list(colors)
fig,ax=plt.subplots(figsize=(9,5.5))
for method in methods:
    rr=sorted([r for r in summary if r['method']==method and r['condition']=='ALL'],key=lambda r:int(r['severity_level']))
    if not rr:continue
    x=np.array([int(r['severity_level']) for r in rr]); y=np.array([float(r['exact_decode_rate'])*100 for r in rr])
    lo=np.array([float(r['exact_ci_identity_low'])*100 for r in rr]);hi=np.array([float(r['exact_ci_identity_high'])*100 for r in rr])
    ax.plot(x,y,marker='o',lw=2,label=labels[method],color=colors[method])
    ax.fill_between(x,lo,hi,color=colors[method],alpha=.10)
ax.set_xticks([1,2,3,4],['L1\n0.70','L2\n0.50','L3\n0.30','L4\n0.20'])
ax.set_ylim(-2,102);ax.set_xlabel('Shadow illumination factor (within shadow region)')
ax.set_ylabel('Exact decode rate (%)');ax.set_title('Validation decoding across synthetic shadow severity')
ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=8,frameon=False);fig.tight_layout()
fig.savefig(FIGS/'stage2_decode_rate_by_severity.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,8),sharey=True)
for ax,(sev,cond) in zip(axes.flat,[(1,'S'),(2,'DS'),(3,'SG'),(4,'DSG')]):
    rr=[r for r in summary if int(r['severity_level'])==sev and r['condition']==cond]
    rr=sorted(rr,key=lambda r:methods.index(r['method']) if r['method'] in methods else 99)
    for r in rr:
        m=r['method']
        if m not in methods: continue
        xx=methods.index(m); rate=float(r['exact_decode_rate'])*100
        ax.bar(xx,rate,color=colors[m],width=.78)
        lo=float(r['exact_ci_identity_low'])*100;hi=float(r['exact_ci_identity_high'])*100
        ax.errorbar(xx,rate,yerr=[[max(0,rate-lo)],[max(0,hi-rate)]],color='black',capsize=2,lw=1)
    ax.set_xticks(range(len(methods)),[labels[m] for m in methods],rotation=35,ha='right',fontsize=7)
    ax.set_title(f'L{sev} {cond}');ax.set_ylim(0,105);ax.grid(axis='y',alpha=.2)
    ax.set_ylabel('Exact decode rate (%)')
fig.suptitle('Focused conditions across severity (identity bootstrap 95% intervals)')
fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(FIGS/'stage2_focus_conditions.png',dpi=180);plt.close(fig)

contrasts=load('hypothesis_contrasts_validation.csv')
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
spec=[('H1_SINGLE_vs_ALL','U-Net ALL minus U-Net SINGLE'),('H2_STANDARD_vs_STRUCTURE_WEIGHTED','Structure-weighted minus standard')]
focus=['S','DS','SG','DSG']; pal=dict(zip(focus,['#1f77b4','#ff7f0e','#2ca02c','#d62728']))
for ax,(h,title) in zip(axes,spec):
  for cond in focus:
    rr=sorted([r for r in contrasts if r['hypothesis']==h and r['condition']==cond],key=lambda r:int(r['severity_level']))
    x=np.array([int(r['severity_level']) for r in rr]); y=np.array([float(r['paired_rate_difference'])*100 for r in rr])
    lo=np.array([float(r['identity_bootstrap_95_low'])*100 for r in rr]);hi=np.array([float(r['identity_bootstrap_95_high'])*100 for r in rr])
    ax.plot(x,y,marker='o',label=cond,color=pal[cond]);ax.fill_between(x,lo,hi,color=pal[cond],alpha=.12)
  ax.axhline(0,color='black',lw=1);ax.set_xticks([1,2,3,4]);ax.set_xlabel('Severity level');ax.set_title(title);ax.grid(alpha=.25)
axes[0].set_ylabel('Paired exact decode rate difference (percentage points)')
axes[1].legend(title='Condition',frameon=False);fig.suptitle('Paired H1 and H2 contrasts on locked validation identities')
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(FIGS/'stage2_hypothesis_contrasts.png',dpi=180);plt.close(fig)
