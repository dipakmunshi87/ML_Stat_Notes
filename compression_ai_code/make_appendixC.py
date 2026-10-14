"""Appendix C figures: reconstruction (same field, degraded) vs realization
(new field, same statistics).
 C1 - a single 1D Fourier mode: new phase = a shift, power spectrum unchanged.
 C2 - two 1D modes: new phases = a different-looking field, SAME power spectrum.
 C3 - a 2D log-normal field: many generated realizations keep the log-normal PDF
      and the power spectrum, while a deterministic lossy reconstruction narrows
      the PDF and suppresses small-scale power."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; PURP="#8E6FB0"; ORANGE="#E0922F"
rng=np.random.default_rng(5)

# ======================================================= C1: single 1D mode
x=np.linspace(0,2*np.pi,400,endpoint=False)
k1=1
f0=np.cos(k1*x)
realizations=[np.cos(k1*x+ph) for ph in [2.1,4.0]]
def ps1d(f):
    F=np.fft.rfft(f)/len(f); return np.abs(F)**2
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,3.3)); fig.patch.set_facecolor("white")
a1.plot(x,f0,color=INK,lw=2.4,label="original  $\\cos(k_1x)$")
a1.plot(x,realizations[0],color=GREEN,lw=1.8,ls="--",label="realization (new phase)")
a1.plot(x,realizations[1],color=ORANGE,lw=1.8,ls=":")
a1.set_title("One mode: a new phase is just a shift",fontsize=10.5); a1.set_xlabel("$x$"); a1.set_yticks([-1,0,1])
a1.legend(fontsize=8.3,frameon=False,loc="upper right")
for sp in a1.spines.values(): sp.set_color("#e3e8ee")
a2.plot(ps1d(f0),"o-",color=INK,lw=2,ms=5,label="original")
a2.plot(ps1d(realizations[0]),"s--",color=GREEN,lw=1.6,ms=4,label="realizations")
a2.set_xlim(-0.5,6); a2.set_title("Power spectrum: identical",fontsize=10.5)
a2.set_xlabel("mode $k$"); a2.set_ylabel("$|F_k|^2$"); a2.legend(fontsize=8.3,frameon=False)
for sp in a2.spines.values(): sp.set_color("#e3e8ee")
fig.suptitle("C1 — a single Fourier mode (one wavelength)",fontsize=12,y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_c1.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_c1.png")

# ======================================================= C2: two 1D modes
A1,A2=1.0,0.8; k2=2
def twomode(p1,p2): return A1*np.cos(1*x+p1)+A2*np.cos(k2*x+p2)
orig=twomode(0.0,0.0)
r1=twomode(rng.uniform(0,2*np.pi),rng.uniform(0,2*np.pi))
r2=twomode(rng.uniform(0,2*np.pi),rng.uniform(0,2*np.pi))
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.4,3.3)); fig.patch.set_facecolor("white")
a1.plot(x,orig,color=INK,lw=2.4,label="original")
a1.plot(x,r1,color=GREEN,lw=1.8,ls="--",label="realization 1")
a1.plot(x,r2,color=ORANGE,lw=1.8,ls="--",label="realization 2")
a1.set_title("Two modes: new phases → a different shape",fontsize=10.5); a1.set_xlabel("$x$")
a1.legend(fontsize=8.3,frameon=False,loc="upper right")
for sp in a1.spines.values(): sp.set_color("#e3e8ee")
a2.plot(ps1d(orig),"o-",color=INK,lw=2,ms=5,label="original")
a2.plot(ps1d(r1),"s--",color=GREEN,lw=1.6,ms=4,label="realization 1")
a2.plot(ps1d(r2),"^:",color=ORANGE,lw=1.6,ms=4,label="realization 2")
a2.set_xlim(-0.5,6); a2.set_title("Power spectrum: still identical",fontsize=10.5)
a2.set_xlabel("mode $k$"); a2.set_ylabel("$|F_k|^2$"); a2.legend(fontsize=8.3,frameon=False)
for sp in a2.spines.values(): sp.set_color("#e3e8ee")
fig.suptitle("C2 — two Fourier modes (one + two wavelengths): the relative phase sets the shape",fontsize=11.3,y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_c2.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_c2.png")

# ======================================================= C3: 2D log-normal
N=64
kx=np.fft.fftfreq(N)[:,None]*np.ones((1,N)); ky=np.fft.fftfreq(N)[None,:]*np.ones((N,1))
k2g=kx**2+ky**2; k2g[0,0]=1e-8; kk=np.sqrt(k2g)
Pk=1.0/(1.0+(kk/0.06)**2)**1.3; Pk[0,0]=0.0; Pk/=Pk.max()
LN=0.9
def gen_lognormal():
    wh=(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N)))/np.sqrt(2)
    g=np.fft.ifft2(wh*np.sqrt(Pk)*N).real; g/=g.std()+1e-9
    kap=np.exp(LN*g-0.5*LN**2)-1.0
    return (kap/(kap.std()+1e-9))
# target + generated realizations
target=gen_lognormal()
reals=[gen_lognormal() for _ in range(300)]
# deterministic lossy reconstruction: keep the 2% largest Fourier coefficients
def fourier_compress(field,frac=0.02):
    F=np.fft.fft2(field); flat=F.ravel()
    k=int(frac*flat.size); idx=np.argsort(np.abs(flat))[::-1]
    mask=np.zeros(flat.size,bool); mask[idx[:k]]=True
    F2=(flat*mask).reshape(F.shape)
    return np.fft.ifft2(F2).real
recon=fourier_compress(target)

# radial power spectrum
kint=np.round(kk*N).astype(int)
def radial_ps(field):
    P=np.abs(np.fft.fft2(field))**2
    nb=kint.max()+1
    out=np.bincount(kint.ravel(),P.ravel(),nb)/np.maximum(np.bincount(kint.ravel(),None,nb),1)
    return out[1:N//2]
kaxis=np.arange(1,N//2)
ps_target=radial_ps(target); ps_recon=radial_ps(recon)
ps_reals=np.array([radial_ps(r) for r in reals[:60]])

# ---- assemble a 2-row figure --------------------------------------------------
fig=plt.figure(figsize=(11.2,6.6)); fig.patch.set_facecolor("white")
gs=fig.add_gridspec(2,4,height_ratios=[1,0.95],hspace=0.42,wspace=0.28)
vmin,vmax=np.percentile(target,3),np.percentile(target,94)
def shw(ax,m,t,c):
    ax.imshow(m,vmin=vmin,vmax=vmax,cmap="viridis"); ax.set_title(t,fontsize=9.3,color=c); ax.axis("off")
shw(fig.add_subplot(gs[0,0]),target,"target field",INK)
shw(fig.add_subplot(gs[0,1]),reals[0],"realization 1",GREEN)
shw(fig.add_subplot(gs[0,2]),reals[1],"realization 2",GREEN)
shw(fig.add_subplot(gs[0,3]),recon,"lossy reconstruction (2%)",RED)

# PDF
axp=fig.add_subplot(gs[1,0:2])
bins=np.linspace(-1.5,5,70); ctr=0.5*(bins[1:]+bins[:-1])
for r in reals[:40]:
    axp.plot(ctr,np.histogram(r,bins,density=True)[0],color=GREEN,lw=0.6,alpha=0.18)
axp.plot(ctr,np.histogram(target,bins,density=True)[0],color=INK,lw=2.4,label="target")
axp.plot([],[],color=GREEN,lw=1.4,label="40 realizations")
axp.plot(ctr,np.histogram(recon,bins,density=True)[0],color=RED,lw=2.2,ls="--",label="reconstruction")
axp.set_yscale("log"); axp.set_ylim(1e-3,3)
axp.set_title("One-point PDF: realizations stay log-normal",fontsize=10)
axp.set_xlabel("$\\kappa$"); axp.set_ylabel("PDF"); axp.legend(fontsize=8,frameon=False)
for sp in axp.spines.values(): sp.set_color("#e3e8ee")

# power spectrum
axk=fig.add_subplot(gs[1,2:4])
for p in ps_reals: axk.plot(kaxis,p,color=GREEN,lw=0.6,alpha=0.18)
axk.plot(kaxis,ps_target,color=INK,lw=2.4,label="target")
axk.plot([],[],color=GREEN,lw=1.4,label="realizations")
axk.plot(kaxis,ps_recon,color=RED,lw=2.2,ls="--",label="reconstruction")
axk.set_xscale("log"); axk.set_yscale("log")
axk.set_title("Power spectrum: realizations match; reconstruction drops",fontsize=10)
axk.set_xlabel("wavenumber $k$"); axk.set_ylabel("$P(k)$"); axk.legend(fontsize=8,frameon=False)
for sp in axk.spines.values(): sp.set_color("#e3e8ee")

fig.suptitle("C3 — a 2D log-normal sky: generated realizations preserve the statistics; a reconstruction degrades them",fontsize=11.6,y=1.0)
fig.savefig(os.path.join(OUT,"fig_c3.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_c3.png")

# ---- numbers for the text -----------------------------------------------------
sk_t=skew(target.ravel()); sk_r=np.mean([skew(r.ravel()) for r in reals]); sk_c=skew(recon.ravel())
var_t=target.var(); var_r=np.mean([r.var() for r in reals]); var_c=recon.var()
with open(os.path.join(OUT,"numbers_c.txt"),"w") as f:
    f.write(f"skew_target={sk_t:.2f}\nskew_realiz={sk_r:.2f}\nskew_recon={sk_c:.2f}\n")
    f.write(f"var_target={var_t:.2f}\nvar_realiz={var_r:.2f}\nvar_recon={var_c:.2f}\n")
print(f"skew target {sk_t:.2f}  realiz {sk_r:.2f}  recon {sk_c:.2f}")
print(f"var  target {var_t:.2f}  realiz {var_r:.2f}  recon {var_c:.2f}")
