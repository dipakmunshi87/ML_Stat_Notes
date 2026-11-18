"""Appendix H figure: a normalizing flow in 1D = change of variables.
The map x = e^z pushes a standard Gaussian p(z) into a log-normal p(x).
Where the map is steep, probability is squeezed (density divided by the slope
= the Jacobian). This IS the generator of the log-normal field used all through
the note."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; ORANGE="#E0922F"; PURP="#8E6FB0"

z=np.linspace(-2.4,2.0,400); x=np.exp(z)
phi=lambda t: np.exp(-t**2/2)/np.sqrt(2*np.pi)
lognorm=lambda xx: phi(np.log(xx))/xx

fig=plt.figure(figsize=(7.8,6.2)); fig.patch.set_facecolor("white")
gs=fig.add_gridspec(2,2,width_ratios=[1,3.1],height_ratios=[3.1,1],wspace=0.05,hspace=0.05)
axm=fig.add_subplot(gs[0,1]); axl=fig.add_subplot(gs[0,0],sharey=axm); axb=fig.add_subplot(gs[1,1],sharex=axm)
fig.add_subplot(gs[1,0]).axis("off")

axm.plot(z,x,color=INK,lw=2.6)
axm.set_ylim(0,7.5); axm.set_xlim(-2.4,2.0)
axm.tick_params(labelbottom=False,labelleft=False)
axm.text(-2.2,6.6,"the map  x = e$^{z}$",fontsize=12,color=INK,fontweight="bold")
# sample points mapped z -> x
for zz,c in [(-1.0,PURP),(0.5,ORANGE),(1.5,GREEN)]:
    xx=np.exp(zz)
    axm.plot([zz,zz],[0,xx],color=c,lw=1,ls=":"); axm.plot([-2.4,zz],[xx,xx],color=c,lw=1,ls=":")
    axm.plot(zz,xx,"o",color=c,ms=6)
axm.annotate("steep here →\ndensity squeezed",(1.5,np.exp(1.5)),textcoords="offset points",xytext=(-78,6),fontsize=8,color="#3f7a38")
axm.annotate("flat here →\ndensity piles up",(-1.0,np.exp(-1.0)),textcoords="offset points",xytext=(14,6),fontsize=8,color="#6a4f91")
for sp in axm.spines.values(): sp.set_color("#e3e8ee")

# bottom: Gaussian p(z)
axb.fill_between(z,phi(z),color=ACC,alpha=0.35); axb.plot(z,phi(z),color=ACC,lw=1.8)
axb.set_ylim(0,0.5); axb.invert_yaxis()
axb.set_xlabel("latent  z   (Gaussian  base)"); axb.set_yticks([])
axb.text(-2.3,0.12,"p(z) = Gaussian",fontsize=9,color=ACC2)
for sp in axb.spines.values(): sp.set_color("#e3e8ee")

# left: lognormal p(x)
xx=np.linspace(0.02,7.5,400)
axl.fill_betweenx(xx,lognorm(xx),color=GREEN,alpha=0.30); axl.plot(lognorm(xx),xx,color=GREEN,lw=1.8)
axl.set_xlim(0,0.75); axl.invert_xaxis()
axl.set_ylabel("data  x   (log-normal)"); axl.set_xticks([])
axl.text(0.7,5.6,"p(x) =\nlog-normal",fontsize=9,color="#3f7a38")
for sp in axl.spines.values(): sp.set_color("#e3e8ee")

fig.suptitle("H — a normalizing flow in 1D:  push a Gaussian through x = e$^{z}$,  get a log-normal",fontsize=11.6,y=0.98)
fig.savefig(os.path.join(OUT,"fig_flow.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_flow.png")
# a couple of pen-and-paper density checks
for xv in [1.0,np.e]:
    print(f"p_x({xv:.3f}) = phi(ln x)/x = {phi(np.log(xv))/xv:.3f}")
