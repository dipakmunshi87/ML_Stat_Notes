"""Two worked 1D examples for the compression note:
 (A) PCA / Karhunen-Loeve compression of a family of 1D signals.
 (B) linear PCA vs a nonlinear autoencoder compressing a CURVED dataset to 1 number."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; PURP="#8E6FB0"
rng=np.random.default_rng(1)

# ============================================================ A. 1D PCA / KL
M=256; x=np.linspace(0,1,M)
# family of smooth signals: a few random Gaussian bumps each
def sig():
    y=np.zeros(M)
    for _ in range(3):
        c=rng.uniform(0.1,0.9); w=rng.uniform(0.03,0.12); a=rng.uniform(-1,1)
        y+=a*np.exp(-0.5*((x-c)/w)**2)
    return y
X=np.stack([sig() for _ in range(400)])
mean=X.mean(0); U,S,Vt=np.linalg.svd(X-mean,full_matrices=False)
var=S**2/np.sum(S**2)
def recon(v,k):
    z=(v-mean)@Vt[:k].T; return z@Vt[:k]+mean

fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(11.0,3.2)); fig.patch.set_facecolor("white")
for i in range(5): a1.plot(x,X[i],lw=1.2,alpha=.8)
a1.set_title("A family of 1D signals  $x$",fontsize=10.3); a1.set_xlabel("position")
a1.set_yticks([])
for sp in a1.spines.values(): sp.set_color("#e3e8ee")

a2.bar(range(1,11),var[:10],color=ACC,alpha=.85)
a2.set_title("Variance per KL mode  $\\lambda_i$",fontsize=10.3); a2.set_xlabel("component $i$")
a2.set_ylabel("fraction of variance")
for sp in a2.spines.values(): sp.set_color("#e3e8ee")

t=X[0]
a3.plot(x,t,color=INK,lw=2.2,label="original")
for k,c in [(1,RED),(3,ACC),(8,GREEN)]:
    r=recon(t,k); a3.plot(x,r,lw=1.6,color=c,ls="--",
        label=f"$k={k}$  (RMSE {np.sqrt(np.mean((r-t)**2)):.2f})")
a3.set_title("Reconstruction from $k$ numbers",fontsize=10.3); a3.set_xlabel("position")
a3.set_yticks([]); a3.legend(fontsize=7.6,frameon=False)
for sp in a3.spines.values(): sp.set_color("#e3e8ee")
fig.suptitle("Worked 1D example: PCA / Karhunen–Loève compression of signals",fontsize=11.5,y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_1d_pca.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_1d_pca.png  (var in first 3 modes = %.2f)"%var[:3].sum())

# ============================================================ B. PCA vs AE on a curve
# data on a semicircle (intrinsic dim 1) in 2D, + small noise
th=rng.uniform(0.15,np.pi-0.15,600)
P=np.stack([np.cos(th),np.sin(th)],1)+0.04*rng.standard_normal((600,2))
Pm=P.mean(0); Uc,Sc,Vtc=np.linalg.svd(P-Pm,full_matrices=False)
# PCA to 1 component
z1=(P-Pm)@Vtc[:1].T; pca1=z1@Vtc[:1]+Pm
# tiny autoencoder 2 -> ... -> 1 -> ... -> 2
import torch, torch.nn as nn
torch.manual_seed(0)
Pt=torch.tensor(P,dtype=torch.float32)
ae=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1),
                 nn.Linear(1,16),nn.Tanh(),nn.Linear(16,32),nn.Tanh(),nn.Linear(32,2))
opt=torch.optim.Adam(ae.parameters(),5e-3); lf=nn.MSELoss()
for ep in range(1500):
    opt.zero_grad(); l=lf(ae(Pt),Pt); l.backward(); opt.step()
with torch.no_grad(): aer=ae(Pt).numpy()
# order AE recon by latent for a clean curve
with torch.no_grad():
    enc=nn.Sequential(*list(ae.children())[:5]); lat=enc(Pt).numpy().ravel()
oi=np.argsort(lat)

rp=np.sqrt(np.mean((pca1-P)**2)); ra=np.sqrt(np.mean((aer-P)**2))
fig,ax=plt.subplots(figsize=(6.2,4.6)); fig.patch.set_facecolor("white")
ax.scatter(P[:,0],P[:,1],s=10,color=MUT,alpha=.45,label="data (1-D curve in 2-D)")
oi2=np.argsort(z1.ravel())
ax.plot(pca1[oi2,0],pca1[oi2,1],color=ACC,lw=2.5,label=f"PCA → 1 number  (RMSE {rp:.3f})")
ax.plot(aer[oi,0],aer[oi,1],color=GREEN,lw=2.5,label=f"autoencoder → 1 number  (RMSE {ra:.3f})")
ax.set_title("Compressing a curved dataset to ONE number",fontsize=11.5)
ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$"); ax.set_aspect("equal")
ax.legend(fontsize=8.5,frameon=False,loc="lower center")
for sp in ax.spines.values(): sp.set_color("#e3e8ee")
ax.text(0.0,0.35,"linear PCA can only\nfit a straight line;\nthe autoencoder follows\nthe curve",
        fontsize=8.2,color=INK,ha="center")
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_1d_ae.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print(f"wrote fig_1d_ae.png  PCA rmse {rp:.3f}  AE rmse {ra:.3f}")
