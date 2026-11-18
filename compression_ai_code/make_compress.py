"""Figures for the 'Data compression & AI' note.
Reproducible demo: compress simulated weak-lensing convergence maps three ways
 - PCA / Karhunen-Loeve  (optimal LINEAR compression)
 - wavelet top-k         (sparse / nonlinear classical, JPEG2000-style)
 - a dense autoencoder   (learned NONLINEAR compression = 'nonlinear PCA')
and measure, vs the compression ratio, BOTH
 - reconstruction RMSE, and
 - how much of the NON-GAUSSIAN information (field skewness) survives.
Plus two conceptual schematics."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pywt
from scipy.stats import skew
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"
ORANGE="#E0922F"; PURP="#8E6FB0"
rng=np.random.default_rng(11)

# ---------- simulated weak-lensing convergence fields (log-normal) -------------
N=64
kx=np.fft.fftfreq(N)[:,None]*np.ones((1,N)); ky=np.fft.fftfreq(N)[None,:]*np.ones((N,1))
k2=kx**2+ky**2; k2[0,0]=1e-8; kk=np.sqrt(k2)
Pk=1.0/(1.0+(kk/0.06)**2)**1.3; Pk[0,0]=0.0; Pk/=Pk.max()
LN_SIG=0.9
def sample_kappa():
    wh=(rng.standard_normal((N,N))+1j*rng.standard_normal((N,N)))/np.sqrt(2)
    g=np.fft.ifft2(wh*np.sqrt(Pk)*N).real; g/=g.std()+1e-9
    kap=np.exp(LN_SIG*g-0.5*LN_SIG**2)-1.0
    return (kap/(kap.std()+1e-9)).astype(np.float32)

NTR, NTE = 1000, 300
Xtr=np.stack([sample_kappa().ravel() for _ in range(NTR)])
Xte=np.stack([sample_kappa().ravel() for _ in range(NTE)])
true_skew=float(skew(Xte.ravel()))
print(f"data ready. true ensemble skewness = {true_skew:.2f}")

KS=[8,16,32,64,128,256]             # code sizes for the linear/sparse curves
KS_AE=[16,64,256]                   # code sizes for the (costlier) autoencoder
ratio=[ (N*N)/k for k in KS ]       # compression ratio = 4096 / k
ratio_ae=[ (N*N)/k for k in KS_AE ]

# ---------- 1. PCA / Karhunen-Loeve (optimal linear) --------------------------
mean=Xtr.mean(0, keepdims=True)
# economy SVD of centred data: rows = samples
U,S,Vt=np.linalg.svd(Xtr-mean, full_matrices=False)   # Vt: components x 4096
def pca_recon(X,k):
    proj=(X-mean)@Vt[:k].T
    return proj@Vt[:k]+mean
pca_rmse=[]; pca_sk=[]
for k in KS:
    R=pca_recon(Xte,k)
    pca_rmse.append(float(np.sqrt(np.mean((R-Xte)**2))))
    pca_sk.append(float(skew(R.ravel())))

# ---------- 2. wavelet top-k (sparse) -----------------------------------------
def wave_keepk(img2d,k):
    coeffs=pywt.wavedec2(img2d,"sym4",level=3)
    arr,sl=pywt.coeffs_to_array(coeffs)
    flat=arr.ravel(); idx=np.argsort(np.abs(flat))[::-1]
    mask=np.zeros_like(flat); mask[idx[:k]]=1.0
    arr2=(flat*mask).reshape(arr.shape)
    return pywt.waverec2(pywt.array_to_coeffs(arr2,sl,output_format="wavedec2"),"sym4")[:N,:N]
wav_rmse=[]; wav_sk=[]
for k in KS:
    R=np.stack([wave_keepk(Xte[i].reshape(N,N),k).ravel() for i in range(NTE)])
    wav_rmse.append(float(np.sqrt(np.mean((R-Xte)**2))))
    wav_sk.append(float(skew(R.ravel())))

# ---------- 3. convolutional autoencoder (learned nonlinear) ------------------
import torch, torch.nn as nn
torch.manual_seed(0); dev="cpu"
Xtr_t=torch.tensor(Xtr).reshape(-1,1,N,N); Xte_t=torch.tensor(Xte).reshape(-1,1,N,N)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.ReLU())
class ConvAE(nn.Module):
    def __init__(s,k,c=16):
        super().__init__()
        s.enc=nn.Sequential(cbr(1,c),nn.MaxPool2d(2),cbr(c,2*c),nn.MaxPool2d(2),
                            cbr(2*c,4*c),nn.MaxPool2d(2))         # -> 4c x 8 x 8
        s.to_code=nn.Linear(4*c*8*8,k)
        s.from_code=nn.Linear(k,4*c*8*8)
        s.c4=4*c
        s.dec=nn.Sequential(nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),cbr(4*c,2*c),
                            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),cbr(2*c,c),
                            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),cbr(c,c),
                            nn.Conv2d(c,1,1))
    def forward(s,x):
        z=s.enc(x).flatten(1); z=s.to_code(z)
        h=s.from_code(z).view(-1,s.c4,8,8)
        return s.dec(h)
ae_rmse=[]; ae_sk=[]; ae_recon={}
for k in KS_AE:
    net=ConvAE(k).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3); lf=nn.MSELoss()
    nb=NTR
    for ep in range(30):
        perm=torch.randperm(nb)
        for j in range(0,nb,64):
            idx=perm[j:j+64]; opt.zero_grad()
            l=lf(net(Xtr_t[idx]),Xtr_t[idx]); l.backward(); opt.step()
    net.eval()
    with torch.no_grad(): R=net(Xte_t).numpy().reshape(NTE,-1)
    ae_rmse.append(float(np.sqrt(np.mean((R-Xte)**2))))
    ae_sk.append(float(skew(R.ravel())))
    ae_recon[k]=R
    print(f"convAE k={k:3d}  rmse {ae_rmse[-1]:.3f}  skew {ae_sk[-1]:.2f}")

print("PCA rmse", [round(x,3) for x in pca_rmse])
print("WAV rmse", [round(x,3) for x in wav_rmse])
print("AE  rmse", [round(x,3) for x in ae_rmse])

# ---------- figure: results (RMSE + skewness vs compression ratio) ------------
fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,3.9)); fig.patch.set_facecolor("white")
a1.plot(ratio,pca_rmse,"o-",color=ACC,lw=2,label="PCA (linear)")
a1.plot(ratio,wav_rmse,"s-",color=PURP,lw=2,label="wavelet (sparse)")
a1.plot(ratio_ae,ae_rmse,"^-",color=GREEN,lw=2,label="autoencoder (learned)")
a1.set_xscale("log"); a1.set_xlabel("compression ratio  (4096 / code size)")
a1.set_ylabel("reconstruction RMSE"); a1.set_title("Fidelity vs compression",fontsize=10.5)
a1.legend(fontsize=8.5,frameon=False); a1.invert_xaxis()
for sp in a1.spines.values(): sp.set_color("#e3e8ee")
a2.axhline(true_skew,color=MUT,lw=1.4,ls="--",label="true (uncompressed)")
a2.plot(ratio,pca_sk,"o-",color=ACC,lw=2,label="PCA (linear)")
a2.plot(ratio,wav_sk,"s-",color=PURP,lw=2,label="wavelet")
a2.plot(ratio_ae,ae_sk,"^-",color=GREEN,lw=2,label="autoencoder")
a2.set_xscale("log"); a2.set_xlabel("compression ratio  (4096 / code size)")
a2.set_ylabel("recovered skewness"); a2.set_title("Non-Gaussian information retained",fontsize=10.5)
a2.legend(fontsize=8,frameon=False); a2.invert_xaxis()
for sp in a2.spines.values(): sp.set_color("#e3e8ee")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_compress_results.png"),dpi=150); plt.close(fig)
print("wrote fig_compress_results.png")

# ---------- figure: one map compressed 64x three ways (k=64) ------------------
KF=64
ex=0
vmin,vmax=np.percentile(Xte[ex],3),np.percentile(Xte[ex],92)   # brighten: clip the tails
pcaF=pca_recon(Xte[ex:ex+1],KF)[0]
wavF=wave_keepk(Xte[ex].reshape(N,N),KF).ravel()
aeF =ae_recon[KF][ex]
np.savez(os.path.join(OUT,"maps.npz"),orig=Xte[ex],pca=pcaF,wav=wavF,ae=aeF)  # for cheap re-plots
panels=[("Original  κ",Xte[ex],INK),
        (f"PCA  (64×)\nRMSE {np.sqrt(np.mean((pcaF-Xte[ex])**2)):.2f}",pcaF,ACC),
        (f"Wavelet  (64×)\nRMSE {np.sqrt(np.mean((wavF-Xte[ex])**2)):.2f}",wavF,PURP),
        (f"Autoencoder  (64×)\nRMSE {np.sqrt(np.mean((aeF-Xte[ex])**2)):.2f}",aeF,GREEN)]
fig,axes=plt.subplots(1,4,figsize=(10.6,3.0)); fig.patch.set_facecolor("white")
for ax,(t,m,c) in zip(axes,panels):
    ax.imshow(m.reshape(N,N),vmin=vmin,vmax=vmax,cmap="viridis"); ax.set_title(t,fontsize=9.4,color=c); ax.axis("off")
fig.suptitle("The same convergence map compressed 64× (4096 → 64 numbers)",fontsize=11.5,y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_compress_maps.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_compress_maps.png")

# ---------- conceptual schematic: compression = model + code ------------------
def concept():
    fig,ax=plt.subplots(figsize=(9.6,3.4)); fig.patch.set_facecolor("white")
    ax.set_xlim(0,12); ax.set_ylim(0,5); ax.axis("off")
    def box(x,w,color,ec,lab,sub=""):
        ax.add_patch(FancyBboxPatch((x,1.7),w,1.6,boxstyle="round,pad=0.03,rounding_size=0.12",
            lw=1.5,edgecolor=ec,facecolor=color))
        ax.text(x+w/2,2.75,lab,ha="center",va="center",fontsize=10.5,fontweight="bold",color=INK)
        if sub: ax.text(x+w/2,2.25,sub,ha="center",va="center",fontsize=8,color=MUT,style="italic")
    box(0.2,2.3,"#dbe8f6",ACC2,"data","x  (4096 pixels)")
    box(3.2,2.3,"#eafaea",GREEN,"encoder","learned model")
    box(6.2,1.7,"#fff3d9","#c88a2a","code","k numbers (bits)")
    box(8.5,2.3,"#f6ded9",RED,"decoder","reconstruct")
    box(11.0,0.9,"#f0f3f7",MUT,"x̂","")
    for x0,x1 in [(2.5,3.2),(5.5,6.2),(7.9,8.5),(10.8,11.0)]:
        ax.add_patch(FancyArrowPatch((x0,2.5),(x1,2.5),arrowstyle="-|>",mutation_scale=13,lw=1.5,color=MUT))
    ax.text(6.0,4.3,"a good model = a short code  ⟺  learning is compression",
            ha="center",fontsize=11,color=ACC2,fontweight="bold")
    ax.text(6.0,0.7,"The smaller the code that still rebuilds the data, the more structure the model has captured (MDL / Occam).",
            ha="center",fontsize=8.6,color=MUT,style="italic")
    fig.savefig(os.path.join(OUT,"fig_compress_concept.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
    print("wrote fig_compress_concept.png")
concept()

with open(os.path.join(OUT,"numbers.txt"),"w") as f:
    f.write(f"true_skew={true_skew:.2f}\n")
    for nm,arr in [("pca_rmse",pca_rmse),("wav_rmse",wav_rmse),("ae_rmse",ae_rmse),
                   ("pca_sk",pca_sk),("wav_sk",wav_sk),("ae_sk",ae_sk)]:
        f.write(nm+"="+",".join(f"{x:.3f}" for x in arr)+"\n")
    f.write("ks="+",".join(str(k) for k in KS)+"\n")
    f.write("ks_ae="+",".join(str(k) for k in KS_AE)+"\n")
print("done")
