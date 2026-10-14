"""Appendix J figure: ICA = whitening (PCA) + a rotation, on two sources.
Uniform (non-Gaussian) sources -> a square. Mixing shears it. Whitening makes
it a square again but ROTATED (PCA cannot finish). The ICA rotation (max
non-Gaussianity) un-rotates it, recovering the sources."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; ORANGE="#E0922F"; PURP="#8E6FB0"
rng=np.random.default_rng(3)

n=1400
S=rng.uniform(-1,1,size=(2,n))            # two independent, non-Gaussian (uniform) sources
A=np.array([[1.0,1.0],[0.0,2.0]])          # mixing matrix
X=A@S                                       # observed mixture
# --- whiten (this is PCA) ---
Xc=X-X.mean(1,keepdims=True)
C=np.cov(Xc)
w,E=np.linalg.eigh(C)                       # eigen-decomposition of 2x2 covariance
Wh=E@np.diag(w**-0.5)@E.T                   # C^{-1/2}
Z=Wh@Xc                                     # whitened: Cov(Z)=I  (a rotated square)
# --- find the un-rotation: angle maximizing non-Gaussianity (|excess kurtosis|) ---
def exkurt(y): return np.mean(y**4)/np.mean(y**2)**2-3
best=(-1,0.0)
for th in np.linspace(0,np.pi/2,400):
    R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    Y=R@Z; sc=abs(exkurt(Y[0]))+abs(exkurt(Y[1]))
    if sc>best[0]: best=(sc,th)
th=best[1]; R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); Y=R@Z
print(f"recovered rotation angle = {np.degrees(th):.1f} deg; source kurtosis ~ {exkurt(S[0]):.2f}")

fig,ax=plt.subplots(1,4,figsize=(12.0,3.3)); fig.patch.set_facecolor("white")
def sc(a,D,t,c):
    a.scatter(D[0],D[1],s=3,color=c,alpha=0.35,edgecolors="none")
    a.set_title(t,fontsize=9.6,color=INK); a.set_aspect("equal")
    a.set_xticks([]); a.set_yticks([])
    for sp in a.spines.values(): sp.set_color("#e3e8ee")
sc(ax[0],S,"1. sources  s\n(independent, square)",GREEN)
sc(ax[1],Xc,"2. mixed  x = A s\n(sheared → correlated)",ORANGE)
sc(ax[2],Z,"3. whitened  z = C$^{-1/2}$x\n(square again, but ROTATED)",ACC)
sc(ax[3],Y,"4. ICA rotation  y = R z\n(un-rotated → recovered)",PURP)
fig.suptitle("J — ICA = whitening (PCA) + the rotation that maximizes non-Gaussianity",fontsize=11.6,y=1.03)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_ica.png"),dpi=118,bbox_inches="tight"); plt.close(fig)
print("wrote fig_ica.png")
