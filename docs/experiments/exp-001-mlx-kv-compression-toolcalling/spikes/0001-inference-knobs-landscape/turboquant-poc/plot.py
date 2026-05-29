import numpy as np
import matplotlib.pyplot as plt
from benchmark import run

plt.rcParams.update({"figure.dpi": 130, "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

for ax, d in zip(axes, (128, 256)):
    rows = run(d=d, n_ctx=2048)
    for name, bpc, comp, kcos, kld, needle, ocos in rows:
        asym = name.startswith("asym")
        ax.scatter(comp, needle, s=90,
                   c="#d9480f" if asym else "#1c7ed6",
                   marker="D" if asym else "o", zorder=3,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(name, (comp, needle), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5)
    ax.axvline(5.0, ls="--", c="gray", lw=1, alpha=.7)
    ax.text(5.05, ax.get_ylim()[0] + 3, "5x target", color="gray", fontsize=8)
    ax.set_title(f"head_dim = {d}  "
                 f"({'Gemma-style' if d == 256 else 'Llama-style'})")
    ax.set_xlabel("KV-cache compression vs fp16  (higher = less memory)")
    ax.set_ylabel("needle retrieval %  (higher = better quality)")
    ax.grid(alpha=.25)

from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#1c7ed6",
                  markersize=9, label="symmetric K=V"),
           Line2D([0], [0], marker="D", color="w", markerfacecolor="#d9480f",
                  markersize=9, label="asymmetric K/V")]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.02))
fig.suptitle("TurboQuant POC — quality vs memory (synthetic KV, outlier "
             "channels)", y=1.08, fontsize=12, weight="bold")
fig.tight_layout()
fig.savefig("/home/claude/turboquant/tradeoff.png", bbox_inches="tight")
print("saved tradeoff.png")
