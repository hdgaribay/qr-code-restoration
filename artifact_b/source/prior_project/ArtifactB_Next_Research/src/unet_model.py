"""Proposal B compact U-Net and loss. Optional dependency: PyTorch.

This module is source-complete but the current run environment lacks PyTorch;
see results/unet_status.json. It deliberately has no test-time QR geometry or
payload input.
"""
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # make the blocker explicit at import/use time
    torch = None
    _TORCH_IMPORT_ERROR = exc

if torch is not None:
    class Block(nn.Module):
        def __init__(self, a, b):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))
        def forward(self, x): return self.net(x)

    class CompactUNet(nn.Module):
        """Four levels at 16/32/64/128 channels; grayscale probability output."""
        def __init__(self):
            super().__init__()
            widths=(16,32,64,128)
            self.enc=nn.ModuleList([Block(1,widths[0]),Block(widths[0],widths[1]),Block(widths[1],widths[2]),Block(widths[2],widths[3])])
            self.pool=nn.MaxPool2d(2)
            self.up=nn.ModuleList([nn.ConvTranspose2d(128,64,2,2),nn.ConvTranspose2d(64,32,2,2),nn.ConvTranspose2d(32,16,2,2)])
            self.dec=nn.ModuleList([Block(128,64),Block(64,32),Block(32,16)])
            self.head=nn.Conv2d(16,1,1)
        def forward(self,x):
            skips=[]
            for i,block in enumerate(self.enc):
                x=block(x)
                if i<3: skips.append(x); x=self.pool(x)
            for up,block,skip in zip(self.up,self.dec,reversed(skips)):
                x=up(x)
                if x.shape[-2:]!=skip.shape[-2:]: x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
                x=block(torch.cat([skip,x],dim=1))
            return torch.sigmoid(self.head(x))

    def proposal_loss(prob, target, fixed_pattern_mask=None, alpha=2.0, lam=0.2):
        """Weighted BCE + gradient L1; fixed mask is geometry-only QR structure."""
        eps=1e-6
        if fixed_pattern_mask is None: weight=torch.ones_like(target)
        else: weight=1.0+float(alpha)*fixed_pattern_mask
        bce=-(target*torch.log(prob.clamp(eps,1-eps))+(1-target)*torch.log((1-prob).clamp(eps,1-eps)))
        gx_p=prob[:,:,:,1:]-prob[:,:,:,:-1]; gy_p=prob[:,:,1:,:]-prob[:,:,:-1,:]
        gx_t=target[:,:,:,1:]-target[:,:,:,:-1]; gy_t=target[:,:,1:,:]-target[:,:,:-1,:]
        grad=(gx_p-gx_t).abs().mean()+(gy_p-gy_t).abs().mean()
        return (weight*bce).mean()+float(lam)*grad

else:
    class CompactUNet:
        def __init__(self,*args,**kwargs): raise RuntimeError('PyTorch unavailable; neural experiment not executable in this environment') from _TORCH_IMPORT_ERROR
    def proposal_loss(*args,**kwargs): raise RuntimeError('PyTorch unavailable; neural experiment not executable in this environment') from _TORCH_IMPORT_ERROR
