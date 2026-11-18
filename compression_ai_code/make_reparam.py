"""Appendix E figure: the reparameterization trick / amortized stochastic
variational inference, showing randomness entering only through epsilon so
gradients flow through mu and sigma."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; ORANGE="#E0922F"

fig,ax=plt.subplots(figsize=(10.4,4.4)); fig.patch.set_facecolor("white")
ax.set_xlim(0,13); ax.set_ylim(0,6); ax.axis("off")
def box(x,y,w,h,fc,ec,lab,sub="",fs=10.5):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.03,rounding_size=0.1",
        lw=1.6,edgecolor=ec,facecolor=fc,zorder=3))
    ax.text(x,y+(0.12 if sub else 0),lab,ha="center",va="center",fontsize=fs,fontweight="bold",color=INK,zorder=4)
    if sub: ax.text(x,y-0.34,sub,ha="center",va="center",fontsize=7.6,color=MUT,style="italic",zorder=4)
def arrow(x0,y0,x1,y1,col=ACC2,ls="-",lw=1.6):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=12,lw=lw,color=col,linestyle=ls,zorder=2))

box(0.9,3,1.3,1.2,"#dbe8f6",ACC2,"x","input map")
box(3.1,3,1.5,1.4,"#eafaea",GREEN,"encoder","q(z|x) net")
box(5.2,3.9,1.1,0.9,"#fff0cf","#c88a2a","μ")
box(5.2,2.1,1.1,0.9,"#fff0cf","#c88a2a","σ")
box(7.1,3.0,1.35,1.0,"#fdeeec",RED,"z = μ + σ⊙ε","sample")
box(9.5,3,1.5,1.4,"#f6ded9","#9a4a40","decoder","p(x|z) net")
box(11.8,3,1.3,1.2,"#f0f3f7",MUT,"x̂","output")
# epsilon noise node (outside the net)
ax.add_patch(Circle((7.1,5.2),0.42,facecolor="#eceff3",edgecolor=MUT,lw=1.4,zorder=3))
ax.text(7.1,5.2,"ε",ha="center",va="center",fontsize=12,fontweight="bold",color=INK,zorder=4)
ax.text(7.1,5.95,"noise  ε ~ N(0, I)",ha="center",fontsize=8.4,color=MUT,style="italic")

arrow(1.55,3,2.35,3); arrow(3.85,3.3,4.65,3.75); arrow(3.85,2.7,4.65,2.25)
arrow(5.75,3.7,6.45,3.2); arrow(5.75,2.3,6.45,2.8)
arrow(7.1,4.78,7.1,3.55,col=RED,ls=(0,(4,2)))   # epsilon injected
arrow(7.78,3,8.75,3); arrow(10.25,3,11.15,3)

ax.annotate("gradients back-propagate through μ and σ  (the deterministic path)",
            (6.4,0.9),ha="center",fontsize=8.8,color=ACC2)
ax.annotate("randomness enters ONLY through ε, sampled outside the network",
            (7.1,4.3),ha="center",fontsize=8.0,color="#9a4a40",style="italic")
ax.text(6.5,5.65,"the reparameterization trick",ha="center",fontsize=12.5,fontweight="bold",color=INK)
ax.annotate("", (3.1,1.5),(9.5,1.5),arrowprops=dict(arrowstyle="<->",color="#c9d2db",lw=1.2))
ax.text(6.3,1.25,"trained end-to-end by stochastic-gradient ascent on the ELBO  (= amortized SVI)",
        ha="center",fontsize=8.2,color=MUT)
fig.savefig(os.path.join(OUT,"fig_reparam.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_reparam.png")
