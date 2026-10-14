"""Appendix F figure: 'amortized' made simple, in 1D.
Linear-Gaussian model  z~N(0,1), x=z+noise (var sigma_n^2).
Posterior mean is the straight line  mu(x) = x/(1+sigma_n^2).
Left : per-datapoint inference -- each x needs its own optimization.
Right: amortized -- the encoder learns that one line; any x is a single step."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; ORANGE="#E0922F"; PURP="#8E6FB0"

sig_n2=1.0
w=1.0/(1.0+sig_n2)          # = 0.5 : the slope of the posterior-mean line
xs=[-2.0,1.0,3.0]; cols=[PURP,ORANGE,GREEN]

fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.0)); fig.patch.set_facecolor("white")

# --- left: per-datapoint optimization trajectories ---
it=np.arange(0,16)
for x,c in zip(xs,cols):
    mpost=w*x; traj=[0.0]
    for _ in it[1:]: traj.append(traj[-1]+0.35*(mpost-traj[-1]))
    a1.plot(it,traj,"-o",color=c,ms=3.5,lw=1.8,label=f"x = {x:g}  →  {mpost:g}")
    a1.axhline(mpost,color=c,lw=0.8,ls=":")
a1.set_title("Per-datapoint inference\n(each x: its own optimization)",fontsize=10.3)
a1.set_xlabel("optimization step"); a1.set_ylabel("inferred latent mean  μ")
a1.legend(fontsize=8,frameon=False,title="data → answer",title_fontsize=8)
for sp in a1.spines.values(): sp.set_color("#e3e8ee")

# --- right: the amortized rule (one line) ---
xg=np.linspace(-3.5,3.5,100)
a2.plot(xg,w*xg,color=ACC,lw=2.6,label=f"encoder rule  μ(x) = {w:g}·x")
for x,c in zip(xs,cols):
    a2.plot(x,w*x,"o",color=c,ms=9)
    a2.annotate(f"x={x:g}",(x,w*x),textcoords="offset points",xytext=(6,-10),fontsize=8,color=c)
a2.axhline(0,color="#e3e8ee",lw=0.8); a2.axvline(0,color="#e3e8ee",lw=0.8)
a2.set_title("Amortized inference\n(learn the rule once; any x is one step)",fontsize=10.3)
a2.set_xlabel("data  x"); a2.set_ylabel("inferred latent mean  μ")
a2.legend(fontsize=8.3,frameon=False,loc="upper left")
for sp in a2.spines.values(): sp.set_color("#e3e8ee")

fig.suptitle("F — 'amortized' in 1D: trade a per-point optimization for a single learned rule",fontsize=11.6,y=1.02)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_amortized.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_amortized.png   w =",w)
