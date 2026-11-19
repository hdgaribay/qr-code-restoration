"""Fresh synthetic QR shadow-pair generation (not real-photo evaluation data)."""
import cv2, numpy as np

def _alignment_centers(version):
    # ISO/IEC 18004 alignment-center table, sufficient through V10.
    table={1:[],2:[6,18],3:[6,22],4:[6,26],5:[6,30],6:[6,34],7:[6,22,38],8:[6,24,42],9:[6,26,46],10:[6,28,50]}
    return table.get(version,[])

def fixed_pattern_mask(n, quiet=4):
    """Finder, timing and alignment geometry, independent of encoded payload."""
    v=max(1,(n-21)//4+1); m=np.zeros((n,n),np.float32)
    for y,x in ((0,0),(0,n-7),(n-7,0)): m[y:y+7,x:x+7]=1
    if n>16:
        m[6,8:n-8]=1; m[8:n-8,6]=1
    centers=_alignment_centers(v)
    for cy in centers:
        for cx in centers:
            if (cy<9 and cx<9) or (cy<9 and cx>n-10) or (cy>n-10 and cx<9): continue
            m[max(0,cy-2):cy+3,max(0,cx-2):cx+3]=1
    canvas=np.zeros((n+2*quiet,n+2*quiet),np.float32); canvas[quiet:quiet+n,quiet:quiet+n]=m
    return canvas

def make_pair(payload, impairment, rng):
    """Return uint8 256x256 degraded input and clean target plus fixed QR mask."""
    enc=cv2.QRCodeEncoder_create()
    raw=enc.encode(payload)
    if raw.ndim==3: raw=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
    # OpenCV's encoder returns a two-module border by default. Add another two
    # modules per side to preserve the proposal's four-module quiet zone.
    n=raw.shape[0]-4
    core=raw[2:2+n,2:2+n]
    framed=np.full((n+8,n+8),255,np.uint8); framed[4:4+n,4:4+n]=core
    clean=cv2.resize(framed,(256,256),interpolation=cv2.INTER_NEAREST)
    x=np.linspace(-1,1,256,dtype=np.float32); xx,yy=np.meshgrid(x,x)
    if impairment in ('shadow','all'):
        softness=rng.uniform(.025,.20); edge=yy-rng.uniform(-.25,.25)+rng.uniform(-.22,.22)*xx
        trans=1-rng.uniform(.25,.78)/(1+np.exp(-edge/max(softness,.005)))
        if impairment=='all' and rng.random()<.5: trans*=1-rng.uniform(.15,.4)*np.exp(-((xx-rng.uniform(-.5,.5))**2+(yy-rng.uniform(-.5,.5))**2)/rng.uniform(.03,.18))
    else: trans=np.ones((256,256),np.float32)
    degraded=clean.astype(np.float32)*trans
    if impairment=='all' and rng.random()<.45:
        s=rng.choice([3,5,7]); degraded=cv2.GaussianBlur(degraded,(s,s),rng.uniform(.5,1.6))
    if impairment=='all' and rng.random()<.4:
        amp=rng.uniform(35,110); blob=np.exp(-((xx-rng.uniform(-.4,.4))**2+(yy-rng.uniform(-.4,.4))**2)/rng.uniform(.015,.08))
        degraded=np.clip(degraded+amp*blob,0,255)
    degraded=np.clip(degraded+rng.normal(0,rng.uniform(0,5),(256,256)),0,255).astype(np.uint8)
    # Output symbol matrix is generated fresh; target is explicitly synthetic.
    mask=fixed_pattern_mask(n)
    mask=cv2.resize(mask,(256,256),interpolation=cv2.INTER_NEAREST)
    return degraded,clean,mask

if __name__=='__main__':
    raise SystemExit('Import make_pair and provide explicit random seeds in the training runner; synthetic pairs are never real test data.')
