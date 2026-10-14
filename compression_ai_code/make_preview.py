"""1200x630 OpenGraph preview card for the Data Compression & AI note."""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT=os.path.dirname(os.path.abspath(__file__))
ACC="#3D7EBF"; ACC2="#245a95"; RED="#D9675C"; GREEN="#5BA552"; INK="#1c2530"; MUT="#5b6b7a"; ORANGE="#E0922F"
fig=plt.figure(figsize=(12,6.3),dpi=100); fig.patch.set_facecolor("#ffffff")
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,12); ax.set_ylim(0,6.3); ax.axis("off")
# left accent bar
ax.add_patch(plt.Rectangle((0,0),0.22,6.3,color=ACC))
ax.text(0.7,5.35,"A reproducible technical note",fontsize=15,color=MUT)
ax.text(0.68,4.35,"Data Compression",fontsize=52,color=ACC2,fontweight="bold")
ax.text(0.68,3.35,"and AI",fontsize=52,color=ACC2,fontweight="bold")
ax.text(0.72,2.55,"Why learning is compression — from Shannon and MDL through PCA,",fontsize=16.5,color=INK)
ax.text(0.72,2.12,"MOPED, autoencoders, VAEs and normalizing flows,",fontsize=16.5,color=INK)
ax.text(0.72,1.69,"with weak-lensing examples and pen-and-paper demos.",fontsize=16.5,color=INK)
ax.text(0.72,0.75,"Dr. Dipak Munshi  ·  ML & Statistics Notes",fontsize=15,color=ACC2,fontweight="bold")
# small compression motif on the right: data -> code -> data
def box(x,w,c,lab):
    ax.add_patch(FancyBboxPatch((x,4.0),w,1.0,boxstyle="round,pad=0.02,rounding_size=0.08",
        lw=1.6,edgecolor=c,facecolor="#eef4fb"))
    ax.text(x+w/2,4.5,lab,ha="center",va="center",fontsize=12,color=INK,fontweight="bold")
box(8.7,1.5,ACC2,"data")
ax.add_patch(FancyBboxPatch((10.55,4.25),0.7,0.5,boxstyle="round,pad=0.02,rounding_size=0.06",lw=1.6,edgecolor=ORANGE,facecolor="#fff3d9"))
ax.text(10.9,4.5,"code",ha="center",va="center",fontsize=10.5,color=INK,fontweight="bold")
ax.add_patch(FancyArrowPatch((10.25,4.5),(10.5,4.5),arrowstyle="-|>",mutation_scale=13,lw=1.6,color=MUT))
ax.text(10.0,3.4,"a good model = a short code",ha="center",fontsize=11.5,color=ACC2,style="italic")
fig.savefig(os.path.join(OUT,"compression_ai_preview.png"),dpi=100); plt.close(fig)
print("wrote compression_ai_preview.png")
