import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
from pyomo.environ import value

def plot_rect_layout(m, title="Optimal Layout", annotate=True, margin=0.05, figsize=(8, 8)):
    idxs = list(m.x.index_set())
    rects = []
    for i in idxs:
        xi = float(value(m.x[i])); yi = float(value(m.y[i]))
        li = float(value(m.l[i])); wi = float(value(m.w[i]))
        rects.append((i, xi, yi, li, wi))

    xmin = min(x for _, x, _, _, _ in rects)
    ymin = min(y for _, _, y, _, _ in rects)
    xmax = max(x + l for _, x, _, l, _ in rects)
    ymax = max(y + w for _, _, y, _, w in rects)
    wspan = xmax - xmin; hspan = ymax - ymin
    pad_x = margin * wspan if wspan > 0 else 1.0
    pad_y = margin * hspan if hspan > 0 else 1.0

    fig, ax = plt.subplots(figsize=figsize)
    for i, xi, yi, li, wi in rects:
        ax.add_patch(Rectangle((xi, yi), li, wi, fill=False, linewidth=1.5, zorder=1))
        if annotate:
            ax.text(
                xi + li/2.0, yi + wi/2.0, f"{i}",
                ha="center", va="center",
                color="blue", fontsize=10,
                zorder=3, clip_on=False,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8),
                path_effects=[pe.withStroke(linewidth=2, foreground="white")]
            )

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.show()

def rect_table(m, decimals=2, styled=True, zero_tol=1e-12):

    rows = [{
        "Rectangle": i,
        "X (Lower Left)": float(value(m.x[i])),
        "Y (Lower Left)": float(value(m.y[i])),
        "Length (X)":     float(value(m.l[i])),
        "Width (Y)":      float(value(m.w[i])),
    } for i in m.x.index_set()]

    df = pd.DataFrame(rows).sort_values("Rectangle").reset_index(drop=True)

    # Coerce near-zeros to exact 0.0 so they render as 0.00 (or 0.000, etc.)
    numeric_cols = ["X (Lower Left)", "Y (Lower Left)", "Length (X)", "Width (Y)"]
    for c in numeric_cols:
        s = df[c].astype(float)
        s = s.mask(s.abs() < zero_tol, 0.0)
        df[c] = s

    if not styled:
        # Fixed decimal string with zeros shown
        fmt = f"{{:.{decimals}f}}"
        for c in numeric_cols:
            df[c] = df[c].map(lambda x: fmt.format(x))
        return df

    fmt = f"{{:.{decimals}f}}"
    st = (
        df.style
        # Always show decimals, including zeros:
        .format({c: fmt.format for c in numeric_cols}, na_rep="—")
        .set_caption("Optimal Rectangle Layout")
        .set_table_styles([
            {"selector": "caption","props":[("text-align","left"),("font-size","16px"),
                                            ("font-weight","bold"),("color","#2C3E50"),
                                            ("padding","10px 0 5px 0")]},
            {"selector": "th","props":[("background-color","#2980B9"),("color","white"),
                                       ("font-weight","bold"),("text-align","center"),
                                       ("padding","6px")]},
            {"selector": "td","props":[("text-align","center"),("padding","6px")]}
        ])
        # Optional visuals (safe; won’t hide zeros):
        .highlight_min(color="#E6F7FF", axis=0)
        .highlight_max(color="#F9EBEA", axis=0)
        .background_gradient(subset=["Length (X)", "Width (Y)"], cmap="Blues", low=0.2, high=0.8)
    )
    return st