"""Fixed full-frame preprocessing and shadow restoration methods."""
import cv2, numpy as np, zxingcpp, time, math
from PIL import Image, ImageOps

MAX_EDGE=1400

def load_fixed_frame(path):
    im=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
    im.thumbnail((MAX_EDGE,MAX_EDGE),Image.Resampling.LANCZOS)
    return np.asarray(im)

def read_exact(image):
    # One decoder call, one format, no retries or payload-informed postprocessing.
    found=zxingcpp.read_barcodes(Image.fromarray(image),formats=zxingcpp.BarcodeFormat.QRCode)
    return found[0].text if found and found[0].text else ''

def gray8(image):
    return cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) if image.ndim==3 else image.astype(np.uint8)

def sauvola(gray, window=51, k=0.22, R=128.0):
    g=gray.astype(np.float32)
    window=max(3,int(window)|1)
    mean=cv2.boxFilter(g,-1,(window,window),normalize=True,borderType=cv2.BORDER_REFLECT)
    mean_sq=cv2.boxFilter(g*g,-1,(window,window),normalize=True,borderType=cv2.BORDER_REFLECT)
    std=np.sqrt(np.maximum(mean_sq-mean*mean,0.0))
    threshold=mean*(1.0+k*(std/R-1.0))
    # QR paper is white; preserve standard black-module polarity.
    return np.where(g>threshold,255,0).astype(np.uint8)

def basic(image,clip=2.0,block=51,C=5):
    g=gray8(image)
    clahe=cv2.createCLAHE(clipLimit=float(clip),tileGridSize=(8,8)).apply(g)
    return cv2.adaptiveThreshold(clahe,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,int(block)|1,float(C))

def close_illum(gray,kernel=61,sigma=21):
    k=max(3,int(kernel)|1)
    se=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
    L=cv2.morphologyEx(gray,cv2.MORPH_CLOSE,se)
    sig=max(0.0,float(sigma))
    if sig: L=cv2.GaussianBlur(L,(0,0),sig)
    return np.maximum(L.astype(np.float32),3.0)

def normalize_with(L,gray):
    g=gray.astype(np.float32)
    base=float(np.median(L))
    return np.clip(g/(L+1e-3)*base,0,255).astype(np.uint8)

def classical_orig(image,kernel=61,sigma=21,window=51,k=0.22):
    g=gray8(image); L=close_illum(g,kernel,sigma); N=normalize_with(L,g)
    return sauvola(N,window,k)

def guided_filter(I,p,radius=15,eps=1e-3):
    I=I.astype(np.float32)/255.0; p=p.astype(np.float32)/255.0
    r=max(1,int(radius)); k=(2*r+1,2*r+1)
    mean_I=cv2.boxFilter(I,-1,k,normalize=True,borderType=cv2.BORDER_REFLECT)
    mean_p=cv2.boxFilter(p,-1,k,normalize=True,borderType=cv2.BORDER_REFLECT)
    corr_I=cv2.boxFilter(I*I,-1,k,normalize=True,borderType=cv2.BORDER_REFLECT)
    corr_Ip=cv2.boxFilter(I*p,-1,k,normalize=True,borderType=cv2.BORDER_REFLECT)
    var_I=corr_I-mean_I*mean_I; cov_Ip=corr_Ip-mean_I*mean_p
    a=cov_Ip/(var_I+float(eps)); b=mean_p-a*mean_I
    mean_a=cv2.boxFilter(a,-1,k,normalize=True,borderType=cv2.BORDER_REFLECT)
    mean_b=cv2.boxFilter(b,-1,k,normalize=True,borderType=cv2.BORDER_REFLECT)
    return np.clip((mean_a*I+mean_b)*255.0,0,255).astype(np.float32)

def _qr_quad(gray):
    det=cv2.QRCodeDetector(); ok,pts=det.detect(gray)
    if ok and pts is not None:
        return pts.reshape(-1,2).astype(np.float32)
    return None

def _finder_pitch(gray,quad):
    # Estimate module pitch from nested finder-pattern contours in a rectified QR quad.
    side=560
    dst=np.float32([[0,0],[side-1,0],[side-1,side-1],[0,side-1]])
    M=cv2.getPerspectiveTransform(quad,dst)
    patch=cv2.warpPerspective(gray,M,(side,side),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    thr=cv2.threshold(patch,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    contours,hier=cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    sizes=[]
    if hier is not None:
        h=hier[0]
        for i,c in enumerate(contours):
            x,y,w,hgt=cv2.boundingRect(c); ratio=w/max(hgt,1); area=w*hgt
            depth=0; q=i
            while h[q][2]>=0 and depth<6: depth+=1; q=h[q][2]
            # Outer finder boxes occupy roughly 15-35% of a QR symbol.
            if 0.80<=ratio<=1.25 and 0.018*side*side<=area<=0.13*side*side and depth>=2:
                sizes.append(w/7.0)
    if sizes:
        # Convert from rectified pixels to image-space pixels.
        pts=quad; qr_side=float(np.median([np.linalg.norm(pts[1]-pts[0]),np.linalg.norm(pts[2]-pts[1]),np.linalg.norm(pts[3]-pts[2]),np.linalg.norm(pts[0]-pts[3])]))
        return float(np.median(sizes)*qr_side/side), 'finder_contours'
    side_len=float(np.median([np.linalg.norm(quad[1]-quad[0]),np.linalg.norm(quad[2]-quad[1]),np.linalg.norm(quad[3]-quad[2]),np.linalg.norm(quad[0]-quad[3])]))
    # Conservative input-only fallback: fixed default module pitch for scaled image.
    return max(3.0,side_len/49.0), 'fixed_modules49_fallback'

def _anchor_surface(gray,quad,w):
    h,ww=gray.shape; pts=quad; x1,y1=pts.min(axis=0); x2,y2=pts.max(axis=0)
    pad=max(1.0,4.0*w); ix1=max(0,int(x1-pad)); ix2=min(ww,int(x2+pad)); iy1=max(0,int(y1-pad)); iy2=min(h,int(y2+pad))
    # Sample only outside the detected QR boundary, within its four-module quiet-zone band.
    xx,yy=np.meshgrid(np.arange(ix1,ix2),np.arange(iy1,iy2)); mask=(xx<x1)&(xx>=ix1)
    mask|=(xx>=x2)&(xx<ix2); mask|=(yy<y1)&(yy>=iy1); mask|=(yy>=y2)&(yy<iy2)
    ys,xs=np.where(mask)
    if len(xs)<80: return None,0.0,0
    x=xs+ix1; y=ys+iy1; z=gray[y,x].astype(np.float64)
    # Trim likely text/edge outliers and fit a quadratic illumination surface.
    lo,hi=np.percentile(z,[10,90]); keep=(z>=lo)&(z<=hi); x=x[keep];y=y[keep];z=z[keep]
    xn=(x-ww/2)/ww; yn=(y-h/2)/h
    A=np.column_stack([np.ones_like(xn),xn,yn,xn*xn,xn*yn,yn*yn])
    coef,_,rank,_=np.linalg.lstsq(A,z,rcond=None)
    if rank<6:return None,0.0,len(z)
    gy,gx=np.mgrid[0:h,0:ww]; gx=(gx-ww/2)/ww;gy=(gy-h/2)/h
    surf=coef[0]+coef[1]*gx+coef[2]*gy+coef[3]*gx*gx+coef[4]*gx*gy+coef[5]*gy*gy
    resid=float(np.median(np.abs(A@coef-z)))
    conf=float(np.clip(len(z)/1200,0,1)*np.exp(-resid/45.0))
    return np.clip(surf,3,255).astype(np.float32),conf,len(z)

def qr_aware(image,window_factor=7,k=0.22,use_anchor=True,use_guided=True):
    g=gray8(image); quad=_qr_quad(g)
    if quad is not None:
        w,source=_finder_pitch(g,quad)
    else:
        w=6.0; source='fixed_pitch6_fallback'
    ksize=max(3,int(round(8*w))|1)
    L=cv2.morphologyEx(g,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize,ksize))).astype(np.float32)
    anchors=0.0; n_anchor=0
    if use_anchor and quad is not None:
        surf,anchors,n_anchor=_anchor_surface(g,quad,w)
        if surf is not None: L=0.5*L+0.5*surf
    if use_guided:
        L=guided_filter(g,L,radius=max(3,int(round(2*w))),eps=1e-3)
    L=np.maximum(L,3.0)
    N=normalize_with(L,g)
    win=max(15,int(round(window_factor*w))|1)
    out=sauvola(N,win,k)
    return out,{'estimated_module_pitch':float(w),'pitch_source':source,'closing_kernel':ksize,'anchor_confidence':float(anchors),'anchor_samples':int(n_anchor),'guided_filter':bool(use_guided),'sauvola_window':win}

METHODS=['RAW','BASIC','CLASSICAL_ORIG','QR_AWARE','ABL_MORPH_ONLY','ABL_NORM_ONLY','ABL_SAUVOLA_ONLY','ABL_PLAIN_MORPH_SAUVOLA','ABL_NO_ANCHOR','ABL_NO_GUIDED']
def run_method(name,image,params=None):
    params=params or {}
    if name=='RAW': return image.copy(),{}
    if name=='BASIC': return basic(image,**params),{}
    if name=='CLASSICAL_ORIG': return classical_orig(image,**params),{}
    if name=='QR_AWARE': return qr_aware(image,**params)
    if name=='ABL_MORPH_ONLY':
        g=gray8(image); L=close_illum(g,params.get('kernel',61),params.get('sigma',21)); return normalize_with(L,g),{}
    if name=='ABL_NORM_ONLY':
        g=gray8(image); p=np.percentile(g,[2,98]); out=np.clip((g-p[0])*255/max(p[1]-p[0],1),0,255).astype(np.uint8); return out,{}
    if name=='ABL_SAUVOLA_ONLY': return sauvola(gray8(image),params.get('window',51),params.get('k',0.22)),{}
    if name=='ABL_PLAIN_MORPH_SAUVOLA': return classical_orig(image,**params),{}
    if name=='ABL_NO_ANCHOR':
        out,meta=qr_aware(image,use_anchor=False,**params); return out,meta
    if name=='ABL_NO_GUIDED':
        out,meta=qr_aware(image,use_guided=False,**params); return out,meta
    raise ValueError(name)
