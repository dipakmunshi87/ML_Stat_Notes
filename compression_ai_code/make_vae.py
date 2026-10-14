"""Appendix D: a convolutional VAE on log-normal convergence maps, showing it
span RECONSTRUCTION (posterior mean -> same sky, smoothed) and GENERATION
(prior samples -> new skies), with PDF and power-spectrum diagnostics."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew
import torch, torch.nn as nn
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; PURP="#8E6FB0"; ORANGE="#E0922F"
rng=np.random.default_rng(9); torch.manual_seed(0)

# ---------- data: log-normal convergence maps ----------
N=64
kx=np.fft.fftfreq(N)[:,None]*np.ones((1,N)); ky=np.fft.fftfreq(N)[None,:]*np.ones((N,1))
k2g=kx**2+ky**2; k2g[0,0]=1e-8; kk=np.sqrt(k2g)
Pk=1.0/(1.0+(kk/0.06)**2)**1.3; Pk[0,0]=0.0; Pk/=Pk.max(); LN=0.9
def gen():
    wh=(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N)))/np.sqrt(2)
    g=np.fft.ifft2(wh*np.sqrt(Pk)*N).real; g/=g.std()+1e-9
    kap=np.exp(LN*g-0.5*LN**2)-1.0; return (kap/(kap.std()+1e-9)).astype(np.float32)
X=np.stack([gen() for _ in range(1200)])[:,None]
Xt=torch.tensor(X)

# ---------- conv VAE ----------
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.ReLU())
class VAE(nn.Module):
    def __init__(s,latent=32,c=16):
        super().__init__()
        s.enc=nn.Sequential(cbr(1,c),nn.MaxPool2d(2),cbr(c,2*c),nn.MaxPool2d(2),cbr(2*c,4*c),nn.MaxPool2d(2))
        s.c4=4*c; s.flat=4*c*8*8
        s.fmu=nn.Linear(s.flat,latent); s.flv=nn.Linear(s.flat,latent); s.fd=nn.Linear(latent,s.flat)
        s.dec=nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),cbr(4*c,2*c),
                            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),cbr(2*c,c),
                            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),cbr(c,c),
                            nn.Conv2d(c,1,1))
    def encode(s,x): h=s.enc(x).flatten(1); return s.fmu(h),s.flv(h)
    def decode(s,z): return s.dec(s.fd(z).view(-1,s.c4,8,8))
    def forward(s,x):
        mu,lv=s.encode(x); z=mu+torch.exp(0.5*lv)*torch.randn_like(mu); return s.decode(z),mu,lv

net=VAE(latent=32); opt=torch.optim.Adam(net.parameters(),1e-3); beta=1.0
nb=X.shape[0]
for ep in range(45):
    net.train(); perm=torch.randperm(nb); tot=0
    for j in range(0,nb,64):
        idx=perm[j:j+64]; xb=Xt[idx]; opt.zero_grad()
        xh,mu,lv=net(xb)
        rec=((xh-xb)**2).sum()/xb.size(0)
        kl=(-0.5*(1+lv-mu.pow(2)-lv.exp()).sum())/xb.size(0)
        loss=rec+beta*kl; loss.backward(); opt.step(); tot+=loss.item()*xb.size(0)
    if ep%10==0 or ep==44: print(f"ep {ep:2d} loss {tot/nb:.1f} rec {rec.item():.1f} kl {kl.item():.1f}")
net.eval()

# ---------- reconstruction (posterior MEAN) and prior SAMPLES ----------
target=torch.tensor(gen()[None,None])
with torch.no_grad():
    mu,lv=net.encode(target); recon=net.decode(mu).numpy()[0,0]          # posterior mean -> same sky
    zs=torch.randn(200,32); samp=net.decode(zs).numpy()[:,0]             # prior -> new skies
target=target.numpy()[0,0]

def radial_ps(field):
    kint=np.round(kk*N).astype(int); P=np.abs(np.fft.fft2(field))**2
    nb=kint.max()+1
    out=np.bincount(kint.ravel(),P.ravel(),nb)/np.maximum(np.bincount(kint.ravel(),None,nb),1)
    return out[1:N//2]
kaxis=np.arange(1,N//2)
truereals=np.stack([gen() for _ in range(80)])

# ---------- figure ----------
fig=plt.figure(figsize=(11.2,6.8)); fig.patch.set_facecolor("white")
gs=fig.add_gridspec(2,4,height_ratios=[1,0.95],hspace=0.42,wspace=0.28)
vmin,vmax=np.percentile(target,3),np.percentile(target,94)
def shw(ax,m,t,c):
    ax.imshow(m,vmin=vmin,vmax=vmax,cmap="viridis"); ax.set_title(t,fontsize=9.2,color=c); ax.axis("off")
shw(fig.add_subplot(gs[0,0]),target,"a target sky",INK)
shw(fig.add_subplot(gs[0,1]),recon,"VAE reconstruction\n(posterior mean μ)",ACC)
shw(fig.add_subplot(gs[0,2]),samp[0],"VAE sample\n(prior z ~ N(0,I))",GREEN)
shw(fig.add_subplot(gs[0,3]),samp[1],"another VAE sample",GREEN)

bins=np.linspace(-1.5,5,70); ctr=0.5*(bins[1:]+bins[:-1])
axp=fig.add_subplot(gs[1,0:2])
for r in truereals[:30]:
    axp.plot(ctr,np.histogram(r,bins,density=True)[0],color=MUT,lw=0.5,alpha=0.25)
axp.plot([],[],color=MUT,lw=1.3,label="true log-normal")
for sarr in samp[:30]:
    axp.plot(ctr,np.histogram(sarr,bins,density=True)[0],color=GREEN,lw=0.5,alpha=0.25)
axp.plot([],[],color=GREEN,lw=1.3,label="VAE samples")
axp.plot(ctr,np.histogram(recon,bins,density=True)[0],color=ACC,lw=2.2,ls="--",label="VAE reconstruction")
axp.set_yscale("log"); axp.set_ylim(1e-3,3); axp.set_xlabel("$\\kappa$"); axp.set_ylabel("PDF")
axp.set_title("One-point PDF",fontsize=10); axp.legend(fontsize=7.8,frameon=False)
for sp in axp.spines.values(): sp.set_color("#e3e8ee")

axk=fig.add_subplot(gs[1,2:4])
for r in truereals[:40]: axk.plot(kaxis,radial_ps(r),color=MUT,lw=0.5,alpha=0.22)
axk.plot([],[],color=MUT,lw=1.3,label="true log-normal")
for sarr in samp[:40]: axk.plot(kaxis,radial_ps(sarr),color=GREEN,lw=0.5,alpha=0.22)
axk.plot([],[],color=GREEN,lw=1.3,label="VAE samples")
axk.plot(kaxis,radial_ps(recon),color=ACC,lw=2.2,ls="--",label="VAE reconstruction")
axk.set_xscale("log"); axk.set_yscale("log"); axk.set_xlabel("wavenumber $k$"); axk.set_ylabel("$P(k)$")
axk.set_title("Power spectrum",fontsize=10); axk.legend(fontsize=7.8,frameon=False)
for sp in axk.spines.values(): sp.set_color("#e3e8ee")

fig.suptitle("D — a VAE spans reconstruction (same sky, smoothed) and generation (new skies, learned statistics)",fontsize=11.4,y=1.0)
fig.savefig(os.path.join(OUT,"fig_vae.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_vae.png")
np.savez(os.path.join(OUT,"vae_arrays.npz"),target=target,recon=recon,samp=samp[:40],
         truereals=truereals[:40])

sk_true=np.mean([skew(r.ravel()) for r in truereals]); sk_samp=np.mean([skew(s.ravel()) for s in samp]); sk_rec=skew(recon.ravel())
vt=np.mean([r.var() for r in truereals]); vs=np.mean([s.var() for s in samp]); vr=recon.var()
with open(os.path.join(OUT,"numbers_vae.txt"),"w") as f:
    f.write(f"skew_true={sk_true:.2f}\nskew_samp={sk_samp:.2f}\nskew_recon={sk_rec:.2f}\n")
    f.write(f"var_true={vt:.2f}\nvar_samp={vs:.2f}\nvar_recon={vr:.2f}\n")
print(f"skew true {sk_true:.2f} samp {sk_samp:.2f} recon {sk_rec:.2f}")
print(f"var  true {vt:.2f} samp {vs:.2f} recon {vr:.2f}")
