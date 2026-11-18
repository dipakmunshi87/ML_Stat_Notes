import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

import os
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quant_toolkit_preview.png")
W, H = 12.0, 6.3
BG = "#0b1730"
fig = plt.figure(figsize=(W, H), dpi=100); fig.patch.set_facecolor(BG)

# radial glow
axbg = fig.add_axes([0, 0, 1, 1]); axbg.set_axis_off()
yy, xx = np.mgrid[0:H*100, 0:W*100]
r = np.sqrt((xx-0.72*W*100)**2 + (yy-0.5*H*100)**2)
glow = np.exp(-(r/520)**2)
axbg.imshow(glow, cmap=LinearSegmentedColormap.from_list("g", [BG, "#13305c"]), extent=[0, W*100, 0, H*100], origin="lower", aspect="auto", zorder=0)
axbg.set_xlim(0, W*100); axbg.set_ylim(0, H*100)

# implied volatility surface
ax = fig.add_axes([0.43, -0.04, 0.62, 1.02], projection="3d"); ax.set_facecolor("none")
k = np.linspace(-0.45, 0.45, 80); T = np.linspace(0.05, 2.0, 80); K, TT = np.meshgrid(k, T)
IV = 0.16 + 0.42*K**2/(1 + 2.0*TT) - 0.14*K/(1 + 0.9*TT) + 0.03*np.exp(-TT/0.3)
cmap = LinearSegmentedColormap.from_list("iv", ["#2a6fcf", "#35a9d8", "#5ee6c8", "#b7e88a", "#ffd166"])
norm = matplotlib.colors.PowerNorm(0.45, IV.min(), IV.max())
ax.plot_surface(K, TT, IV*100, facecolors=cmap(norm(IV)), rstride=2, cstride=2, linewidth=0.3, edgecolor=(1, 1, 1, 0.14), antialiased=True, shade=False, alpha=0.97, zorder=3)
ax.contour(K, TT, IV*100, zdir="z", offset=11, levels=14, cmap=cmap, linewidths=0.9, alpha=0.85)
ax.view_init(elev=27, azim=-48)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane): pane.fill = False; pane.set_edgecolor((1, 1, 1, 0.08))
ax.grid(False)
ax.set_xlabel("log-moneyness  ln(K/S)", color="#8fa6c8", fontsize=9, labelpad=4)
ax.set_ylabel("maturity T (years)", color="#8fa6c8", fontsize=9, labelpad=4)
ax.set_zlabel("implied vol (%)", color="#8fa6c8", fontsize=9, labelpad=2)
ax.tick_params(colors="#8fa6c8", labelsize=7.5, pad=1)
ax.set_xticks([-0.4, 0, 0.4]); ax.set_yticks([0.5, 1.0, 1.5, 2.0]); ax.set_zticks([15, 20, 25, 30])
ax.set_zlim(11, 36)
ax.xaxis.line.set_color((1, 1, 1, 0.15)); ax.yaxis.line.set_color((1, 1, 1, 0.15)); ax.zaxis.line.set_color((1, 1, 1, 0.15))
ax.text2D(0.70, 0.93, "the implied-volatility surface", transform=ax.transAxes, color="#5ee6ff", fontsize=10.5, fontweight="bold", ha="center")
ax.text2D(0.70, 0.895, "what the market says Black–Scholes gets wrong", transform=ax.transAxes, color="#8fa6c8", fontsize=8.5, ha="center")

# typography
axt = fig.add_axes([0, 0, 1, 1]); axt.set_axis_off(); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
axt.add_patch(FancyBboxPatch((0.0, 0.0), 0.012, 1.0, boxstyle="square,pad=0", fc="#5ee6ff", ec="none", zorder=3))
axt.text(0.055, 0.87, "PHYSICS · MATHEMATICS · STATISTICS", color="#8fa6c8", fontsize=11, fontweight="bold", zorder=4)
axt.text(0.055, 0.72, "The Quant", color="#ffffff", fontsize=50, fontweight="bold", zorder=4)
axt.text(0.055, 0.585, "Toolkit", color="#ffffff", fontsize=50, fontweight="bold", zorder=4)
axt.text(0.055, 0.49, "for the finance industry, from first principles", color="#dbe6f5", fontsize=17, zorder=4)
lines = ["Discounting & bonds · fat tails · estimation · PCA",
         "Brownian motion · Itô · Black–Scholes · Greeks · smiles",
         "Rates · Monte Carlo · GARCH · random matrices · Kelly",
         "VaR / EVT · ML signals · backtests · Langevin & Fokker–Planck"]
for i, s in enumerate(lines):
    axt.text(0.055, 0.395 - i*0.058, s, color="#a9b8d0", fontsize=11.3, zorder=4)
axt.text(0.055, 0.105, "22 chapters  ·  25 computed figures  ·  worked examples throughout", color="#5ee6ff", fontsize=11, fontweight="bold", zorder=4)
axt.text(0.055, 0.05, "Dr Dipak Munshi  ·  ML & Statistics Notes", color="#ffffff", fontsize=11.5, fontweight="bold", zorder=4)
fig.savefig(OUT, dpi=100, facecolor=BG)
print("saved", OUT)
