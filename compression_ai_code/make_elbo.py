"""Appendix G figure: the ELBO of a 1D linear-Gaussian VAE, worked by hand.
Model z~N(0,1), x=z+noise (sigma_n^2=1); observe x=2.
ELBO(mu,s2) = reconstruction - KL, maximised at mu*=x/2=1, s2*=1/2.
Left : the three terms vs mu (reconstruction pulls to x, KL/rate pulls to 0, ELBO balances at 1).
Right: ELBO contour over (mu, s2) with the maximum marked."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; ORANGE="#E0922F"; PURP="#8E6FB0"
x=2.0; sn2=1.0
def recon(mu,s2): return -0.5*np.log(2*np.pi*sn2)-0.5*((x-mu)**2+s2)/sn2
def kl(mu,s2):    return 0.5*(mu**2+s2-np.log(s2)-1.0)
def elbo(mu,s2):  return recon(mu,s2)-kl(mu,s2)

fig,(a1,a2)=plt.subplots(1,2,figsize=(10.8,4.1)); fig.patch.set_facecolor("white")

# left: terms vs mu, at optimal s2=0.5
mu=np.linspace(-1.5,3.5,200); s2o=0.5
a1.plot(mu,recon(mu,s2o),color=ORANGE,lw=2,label="reconstruction  (−distortion)")
a1.plot(mu,-kl(mu,s2o),color=ACC,lw=2,label="− KL  (−rate)")
a1.plot(mu,elbo(mu,s2o),color=GREEN,lw=2.6,label="ELBO = recon − KL")
a1.axvline(2,color=ORANGE,lw=0.9,ls=":"); a1.axvline(0,color=ACC,lw=0.9,ls=":"); a1.axvline(1,color=GREEN,lw=1.1,ls="--")
a1.annotate("recon wants μ=x=2",(2,recon(2,s2o)),textcoords="offset points",xytext=(-4,8),fontsize=7.6,color="#b5791f",ha="right")
a1.annotate("rate wants μ=0",(0,-kl(0,s2o)),textcoords="offset points",xytext=(6,-2),fontsize=7.6,color=ACC2)
a1.annotate("ELBO peaks at μ=1",(1,elbo(1,s2o)),textcoords="offset points",xytext=(6,-22),fontsize=8,color="#3f7a38",fontweight="bold")
a1.plot(1,elbo(1,s2o),"o",color=GREEN,ms=7)
a1.set_title("The rate–distortion balance  (at $s^2=\\frac{1}{2}$)",fontsize=10.3)
a1.set_xlabel("posterior mean  μ"); a1.set_ylabel("value (nats)"); a1.legend(fontsize=7.8,frameon=False,loc="lower center")
for sp in a1.spines.values(): sp.set_color("#e3e8ee")

# right: ELBO contour over (mu, s2)
M,S=np.meshgrid(np.linspace(-0.5,2.5,160),np.linspace(0.08,1.6,160))
Z=elbo(M,S)
cf=a2.contourf(M,S,Z,levels=18,cmap="viridis")
a2.plot(1.0,0.5,"*",color="white",ms=18,markeredgecolor="k",markeredgewidth=0.6)
a2.annotate("optimum\nμ*=1,  $s^{2}$*=½",(1.0,0.5),textcoords="offset points",xytext=(10,8),fontsize=8.5,color="white",fontweight="bold")
a2.set_title("ELBO surface  (maximum = true posterior)",fontsize=10.3)
a2.set_xlabel("posterior mean  μ"); a2.set_ylabel("posterior variance  $s^2$")
cb=fig.colorbar(cf,ax=a2,fraction=0.046,pad=0.03); cb.set_label("ELBO (nats)",fontsize=8)
for sp in a2.spines.values(): sp.set_color("#e3e8ee")

fig.suptitle("G — a 1D VAE's ELBO, worked by hand (x = 2):  maximised exactly at the true posterior",fontsize=11.4,y=1.01)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_elbo.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_elbo.png")
print(f"check: ELBO(1,0.5)={elbo(1,0.5):.3f}  recon={recon(1,0.5):.3f}  KL={kl(1,0.5):.3f}")
