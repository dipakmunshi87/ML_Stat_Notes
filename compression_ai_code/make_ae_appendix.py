"""Appendix B figures: how an autoencoder works, pen-and-paper.
 (1) a 1D linear autoencoder (4 -> 1 -> 4) drawn as a network with the actual
     numbers flowing through encode -> bottleneck -> decode -> error;
 (2) a tiny 2D image (4x4) compressed to 2 numbers via two feature 'stencils',
     shown as heatmaps with every value visible."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; PURP="#8E6FB0"

# ================================================================= 1D diagram
x=np.array([2.,2.,-1.,-1.]); w=np.array([.5,.5,-.5,-.5])
z=float(w@x); xh=z*w; err=x-xh; rmse=float(np.sqrt(np.mean(err**2)))
print("1D:", "z=",z,"xhat=",xh,"err=",err,"rmse=",rmse)

fig,ax=plt.subplots(figsize=(9.6,5.0)); fig.patch.set_facecolor("white")
ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
xin=1.4; xmid=5.0; xout=8.6
ys=[4.6,3.6,2.4,1.4]
def node(xx,yy,val,fc,ec):
    ax.add_patch(Circle((xx,yy),0.33,facecolor=fc,edgecolor=ec,lw=1.6,zorder=4))
    ax.text(xx,yy,f"{val:g}",ha="center",va="center",fontsize=10,fontweight="bold",zorder=5,color=INK)
# edges in (with weights)
for yy,wi in zip(ys,w):
    ax.add_patch(FancyArrowPatch((xin+0.33,yy),(xmid-0.37,3.0),arrowstyle="-",lw=1.2,color="#b9c4d0",zorder=1))
    mx,my=(xin+xmid)/2,(yy+3.0)/2
    ax.text(mx-0.1,my+0.12,f"{wi:+g}",fontsize=7.6,color=ACC2,style="italic")
# edges out
for yy,wi in zip(ys,w):
    ax.add_patch(FancyArrowPatch((xmid+0.37,3.0),(xout-0.33,yy),arrowstyle="-|>",mutation_scale=10,lw=1.2,color="#b9c4d0",zorder=1))
    mx,my=(xmid+xout)/2,(3.0+yy)/2
    ax.text(mx+0.05,my+0.12,f"{wi:+g}",fontsize=7.6,color="#9a4a40",style="italic")
for yy,v in zip(ys,x): node(xin,yy,v,"#dbe8f6",ACC2)
node(xmid,3.0,z,"#fff0cf","#c88a2a")
for yy,v in zip(ys,xh): node(xout,yy,v,"#f6ded9",RED)
ax.text(xin,5.4,"input  x",ha="center",fontsize=10.5,fontweight="bold",color=ACC2)
ax.text(xmid,4.3,"code  z",ha="center",fontsize=10.5,fontweight="bold",color="#b8860b")
ax.text(xout,5.4,"output  x̂",ha="center",fontsize=10.5,fontweight="bold",color="#9a4a40")
ax.text(xmid,1.75,"bottleneck\n(1 number)",ha="center",fontsize=8,color=MUT,style="italic")
ax.text(xin-0.05,0.55,"encode:  z = w·x\n = .5(2)+.5(2)−.5(−1)−.5(−1) = 3",
        ha="center",fontsize=8.4,color=ACC2)
ax.text(xout+0.0,0.55,"decode:  x̂ = z·w\n = 3·[.5,.5,−.5,−.5]",
        ha="center",fontsize=8.4,color="#9a4a40")
ax.text(xmid,0.35,f"error x−x̂ = [{', '.join(f'{e:g}' for e in err)}]   (RMSE {rmse:g})\nlost: the constant offset, which is ⊥ to w",
        ha="center",fontsize=8.3,color=RED)
ax.text(5.0,5.75,"A linear autoencoder by hand:  4 → 1 → 4",ha="center",fontsize=12.5,fontweight="bold",color=INK)
fig.savefig(os.path.join(OUT,"fig_ae_1d.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_ae_1d.png")

# ================================================================= 2D image
X=np.array([[3,3,-1,-1],
            [3,3,-1,-1],
            [1,1,-3,-3],
            [1,1,-3,-3]],float)
A=np.tile(np.array([1,1,-1,-1.]),(4,1))/4.0      # left vs right  (unit norm)
B=np.tile(np.array([[1],[1],[-1],[-1.]]),(1,4))/4.0  # top vs bottom
zA=float(np.sum(A*X)); zB=float(np.sum(B*X))
Xh=zA*A+zB*B; res=X-Xh
print("2D: zA=",zA,"zB=",zB,"residual max",np.abs(res).max())

def grid(ax,M,title,cmap="coolwarm",vlim=3.2,tc=INK):
    im=ax.imshow(M,cmap=cmap,vmin=-vlim,vmax=vlim)
    for (i,j),v in np.ndenumerate(M):
        ax.text(j,i,f"{v:g}",ha="center",va="center",fontsize=9,
                color="white" if abs(v)>vlim*0.55 else "black")
    ax.set_title(title,fontsize=9.6,color=tc); ax.set_xticks([]); ax.set_yticks([])

fig,ax=plt.subplots(2,3,figsize=(9.8,6.6)); fig.patch.set_facecolor("white")
grid(ax[0,0],X,"input image  (4×4 = 16 numbers)",tc=ACC2)
grid(ax[0,1],A,"stencil A  (unit 1: left − right)",tc=ACC)
grid(ax[0,2],B,"stencil B  (unit 2: top − bottom)",tc=ACC)
ax[1,0].axis("off")
ax[1,0].text(0.5,0.62,f"code:\n\n$z_A = \\sum A\\odot X = {zA:g}$\n\n$z_B = \\sum B\\odot X = {zB:g}$",
             ha="center",va="center",fontsize=12,color="#b8860b",transform=ax[1,0].transAxes)
ax[1,0].text(0.5,0.12,"16 numbers → 2 numbers",ha="center",fontsize=9,color=MUT,style="italic",transform=ax[1,0].transAxes)
grid(ax[1,1],Xh,f"reconstruction  $z_A A + z_B B$",tc=GREEN)
grid(ax[1,2],res,"residual (lost) = 0",tc=MUT)
fig.suptitle("Compressing a 2D image with an autoencoder: 16 → 2 → 16",fontsize=12.5,y=0.99)
fig.tight_layout(); fig.savefig(os.path.join(OUT,"fig_ae_2d.png"),dpi=150,bbox_inches="tight"); plt.close(fig)
print("wrote fig_ae_2d.png")
