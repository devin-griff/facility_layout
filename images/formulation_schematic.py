"""Regenerates the Formulation-tab schematic (formulation.png).

Illustrates disjunct Y^1 ("i is left of j") of the decoupled model: two
blocks with the rectilinear edge gaps labeled — dx_{i,j} horizontal,
dy_{i,j} vertical. Keep this in sync with the model in app.py; re-run after
any notation change so the image never goes stale again.

    python formulation_schematic.py    # writes formulation.png next to this file

The app loads this PNG directly. The companion notebook ("facility layout.ipynb")
embeds a base64 *copy* as a cell attachment so it renders when downloaded
standalone; this script re-embeds it automatically (needs nbformat installed).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(10, 5.2))

# Block i upper-left, block j lower-right (i is left of AND above j).
i_x, i_y, i_w, i_h = 1.2, 3.4, 2.4, 1.4   # x[1.2, 3.6], y[3.4, 4.8]
j_x, j_y, j_w, j_h = 4.6, 1.2, 2.6, 1.4   # x[4.6, 7.2], y[1.2, 2.6]

for bx, by, bw, bh in [(i_x, i_y, i_w, i_h), (j_x, j_y, j_w, j_h)]:
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, lw=1.8,
                           edgecolor="black"))

ax.text(i_x + i_w / 2, i_y + i_h / 2, r"$i$", ha="center", va="center",
        fontsize=22, style="italic")
ax.text(j_x + j_w / 2, j_y + j_h / 2, r"$j$", ha="center", va="center",
        fontsize=22, style="italic")

# Dotted alignment guides at the inner edges that define the two gaps.
gx1, gx2 = i_x + i_w, j_x          # i's right edge, j's left edge
gy_lo, gy_hi = j_y + j_h, i_y      # j's top edge, i's bottom edge
for gx in (gx1, gx2):
    ax.plot([gx, gx], [1.9, 5.0], ls=":", color="black", lw=1.9)
for gy in (gy_lo, gy_hi):
    ax.plot([0.7, 8.0], [gy, gy], ls=":", color="black", lw=1.9)

# dx: horizontal edge gap, drawn in the empty column between the blocks.
ay = 4.45
ax.annotate("", xy=(gx2, ay), xytext=(gx1, ay),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
ax.text((gx1 + gx2) / 2, ay + 0.12, r"$dx_{i,j}$", ha="center", va="bottom",
        fontsize=18)

# dy: vertical edge gap, drawn to the right of block j.
axx = 7.6
ax.annotate("", xy=(axx, gy_hi), xytext=(axx, gy_lo),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.3))
ax.text(axx + 0.12, (gy_lo + gy_hi) / 2, r"$dy_{i,j}$", ha="left",
        va="center", fontsize=18)

# Active-disjunct indicator.
ax.text(5.2, 4.45, r"$Y^{1}_{i,j} = 1$", ha="center", va="center", fontsize=20)

ax.set_xlim(0.5, 8.4)
ax.set_ylim(0.8, 5.2)
ax.set_aspect("equal")
ax.axis("off")

out = Path(__file__).parent / "formulation.png"
fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
print("wrote", out)

# Keep the companion notebook's embedded copy in sync: it stores this PNG as
# a base64 cell attachment so it renders when the .ipynb is downloaded alone.
try:
    import base64
    import nbformat

    nb_path = Path(__file__).parent.parent / "facility layout.ipynb"
    if nb_path.exists():
        b64 = base64.b64encode(out.read_bytes()).decode("ascii")
        nb = nbformat.read(str(nb_path), as_version=4)
        hits = 0
        for cell in nb.cells:
            src = ("".join(cell.source) if isinstance(cell.source, list)
                   else cell.source)
            if (cell.cell_type == "markdown"
                    and "attachment:formulation.png" in src):
                cell.attachments = {"formulation.png": {"image/png": b64}}
                hits += 1
        if hits:
            nbformat.write(nb, str(nb_path))
            print(f"re-embedded into {nb_path.name} ({hits} cell)")
except Exception as e:  # nbformat missing or notebook unreadable — non-fatal
    print("notebook embed skipped:", e)
