# =============================================================================
# Facility Layout Optimizer — a Streamlit tutorial app.
#
# Plant facility layout problem solved via Pyomo GDP. Place rectangular
# blocks in 2D space to minimize:
#   - facility bounding-box dimensions  (l_f + w_f), plus
#   - cost-weighted Manhattan pipe distances between blocks
#                                       (Σ c_ij · (t_ij + s_ij))
#
# Library roadmap:
#   - streamlit  — UI framework. Each interaction reruns this script
#                  top-to-bottom; persistent values live in `st.session_state`.
#   - pyomo      — algebraic modeling, including the `pyomo.gdp` submodule
#                  for native Disjunction blocks.
#   - HiGHS      — MIP solver, called via Pyomo's `appsi_highs` interface.
#                  Ships as a pip wheel (`highspy`) — no system install.
#   - pandas     — DataFrames for the editable block-dimensions and
#                  cost-matrix tables.
#   - altair     — interactive layout figure (rectangles + pipe lines +
#                  hover tooltips).
#
# Model structure:
#   The non-overlap structure is naturally a 4-way disjunction per block
#   pair (i is left / right / above / below j); rotation (when enabled) is
#   a 2-way disjunction per block (default vs. 90° rotated). Both are
#   written as `pyomo.gdp.Disjunction` blocks and reformulated to a MILP
#   via the multi-Big-M transformation (`gdp.mbigm`), then solved with HiGHS.
#
# Symmetry breaking:
#   `sym=1` is hardcoded. The trivial mirror symmetries make the LP
#   relaxation eight-fold degenerate; pinning block 1 to be "left of and
#   below" block 2 kills four of the eight equivalences and dramatically
#   speeds up the MIP. See `sym_1` and `sym_2` in `build_model`.
#
# Time limit + incumbent handling:
#   At n=10 the solve can blow past 10 s. We set a wall-clock time limit
#   and use Pyomo's `load_solutions=False` path to optionally load the best
#   feasible solution found before the cutoff. The Layout tab annotates
#   "Optimal" vs "Incumbent (suboptimal)" accordingly.
#
# File roadmap (matching section banners below):
#   1. Page config + CSS + home-logo.
#   2. Constants and defaults.
#   3. State helpers — session_state init, scenario generators, resets.
#   4. Solver — build_model + log-capturing solve + incumbent loader.
#   5. Visualization — Altair layout figure with rectangles + pipe overlay.
#   6. Tab renderers — Layout, Data, Formulation, Logs.
#   7. Main — sidebar widgets + tab assembly.
# =============================================================================

import base64
import io
import random
from pathlib import Path

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition


# ── 1. Page config + CSS + home-logo ──────────────────────────────────────────

st.set_page_config(
    page_title="Facility Layout",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar pattern (quad-tank style): home-logo lives at the top of the sidebar
# in normal flow, and Streamlit's sticky `stSidebarHeader` is hidden so the
# logo sits flush at the top. See griffith-pse-app-template SETUP.md "Sidebar
# vs. no sidebar" for the rationale.
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    user-select: none;
    -webkit-user-select: none;
}
.home-logo-corner {
    display: block;
    margin: 0 0 0.75rem;
}
.home-logo-corner img {
    width: 32px;
    height: 32px;
    border-radius: 4px;
    display: block;
}
[data-testid="stSidebarHeader"] {
    display: none !important;
}
[data-testid="stSidebarUserContent"] {
    padding-top: 0.5rem !important;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
    padding-bottom: 0rem !important;
}
</style>
""", unsafe_allow_html=True)

_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
_HOME_LOGO_HTML = (
    '<a class="home-logo-corner" href="https://griffith-pse.com" target="_self">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE — home" />'
    '</a>'
)
st.sidebar.markdown(_HOME_LOGO_HTML, unsafe_allow_html=True)


# ── 2. Constants and defaults ─────────────────────────────────────────────────

N_MIN = 2
N_MAX = 10
N_DEFAULT = 6

# Block-dimension cap — random scenarios draw from {1, 2, 3}. Editable in the
# Data tab without an upper bound enforced (the bigger you go, the harder the
# solve gets).
DIM_MAX_RAND = 3

# Cost-matrix range for the "Random pipes" scenario.
COST_MAX_RAND = 10

# Solve time limit slider range (seconds).
TIMELIMIT_MIN = 5
TIMELIMIT_MAX = 30
TIMELIMIT_DEFAULT = 10

# RNG seed used by the scenario generators. Fixing it keeps the same instance
# across reruns until the user clicks "Initialize at defaults" (which bumps
# the seed).
DEFAULT_SEED = 1


# ── 3. State helpers ──────────────────────────────────────────────────────────

def _generate_scenario(n, scenario, seed):
    """Build (l0_dict, w0_dict, c_matrix, d_default) for the chosen scenario.

    Returns:
        l0:  {i: length}     for i in 1..n
        w0:  {i: width}      for i in 1..n
        cmat: list of lists, n×n, lower-triangular (cmat[i-1][j-1] for i > j),
              upper-triangular and diagonal are 0. Returned as a pandas-friendly
              shape so the data editor can show it directly.
    """
    rng = random.Random(seed)
    l0 = {i: rng.randint(1, DIM_MAX_RAND) for i in range(1, n + 1)}
    w0 = {i: rng.randint(1, DIM_MAX_RAND) for i in range(1, n + 1)}
    cmat = [[0.0] * n for _ in range(n)]
    if scenario == "Central rack":
        # Block 1 is the rack; every other block has unit cost to it, zero
        # cost between non-rack blocks.
        for i in range(2, n + 1):
            cmat[i - 1][0] = 1.0
    elif scenario == "Random pipes":
        for i in range(2, n + 1):
            for j in range(1, i):
                cmat[i - 1][j - 1] = float(rng.randint(1, COST_MAX_RAND))
    # "Custom" — leave at zeros; user fills in.
    return l0, w0, cmat


def _init_state():
    ss = st.session_state
    ss.setdefault("n", N_DEFAULT)
    ss.setdefault("rotate", False)
    ss.setdefault("scenario", "Central rack")
    ss.setdefault("d_min", 1.0)
    ss.setdefault("seed", DEFAULT_SEED)
    if "l0" not in ss or "w0" not in ss or "cmat" not in ss:
        l0, w0, cmat = _generate_scenario(ss["n"], ss["scenario"], ss["seed"])
        ss["l0"] = l0
        ss["w0"] = w0
        ss["cmat"] = cmat


def _resync_for_n_or_scenario():
    """Called when the user changes n or the scenario radio. Regenerates the
    block dims and cost matrix to a fresh instance for the new shape."""
    ss = st.session_state
    l0, w0, cmat = _generate_scenario(ss["n"], ss["scenario"], ss["seed"])
    ss["l0"] = l0
    ss["w0"] = w0
    ss["cmat"] = cmat


def _initialize_at_defaults():
    """`on_click` for the sidebar reset button. Bumps the seed so the user
    gets a *different* random instance, then regenerates the scenario."""
    st.session_state["seed"] += 1
    _resync_for_n_or_scenario()
    st.session_state.pop("res", None)


# ── 4. Solver ─────────────────────────────────────────────────────────────────

def build_model(n, l0, w0, cmat, d_uniform, rotate, sym):
    """Construct the GDP facility-layout model.

    Args:
        n         : int, number of blocks
        l0, w0    : {1..n: float} block default length/width
        cmat      : n×n list-of-lists, lower-triangular pipe costs (cmat[i-1][j-1] for i>j)
        d_uniform : float, min separation distance applied to every pair
        rotate    : bool, allow 90° rotation per block
        sym       : 0 or 1, enable symmetry-breaking on blocks 1 and 2

    Returns the unsolved Pyomo `ConcreteModel`. The caller applies the GDP
    transformation and runs the solver.
    """
    m = pyo.ConcreteModel()

    # Blocks indexed 1..n; pair set is the strict lower triangle (i > j).
    m.n = pyo.Set(ordered=True, initialize=pyo.RangeSet(1, n))
    m.p = pyo.Set(initialize=m.n * m.n, dimen=2,
                  filter=lambda m, i, j: i > j)

    # Default-orientation dimensions.
    m.w0 = pyo.Param(m.n, initialize=w0)
    m.l0 = pyo.Param(m.n, initialize=l0)

    # Pair parameters: pipe cost and minimum required separation.
    c_dict = {(i, j): float(cmat[i - 1][j - 1]) for i, j in m.p}
    d_dict = {(i, j): float(d_uniform) for i, j in m.p}
    m.c = pyo.Param(m.p, initialize=c_dict)
    m.d = pyo.Param(m.p, initialize=d_dict)

    # Conservative upper bound on placement coordinates: stack all blocks
    # along one axis at their longest dimension. Keeps the LP relaxation
    # bounded without being too loose to be useful.
    m.UB = pyo.Param(initialize=sum(max(m.l0[i], m.w0[i]) for i in m.n))

    # Decision variables.
    m.x = pyo.Var(m.n, bounds=(0, m.UB))      # lower-left x
    m.y = pyo.Var(m.n, bounds=(0, m.UB))      # lower-left y
    m.l = pyo.Var(m.n, bounds=(0, m.UB))      # block length (= l0 unless rotated)
    m.w = pyo.Var(m.n, bounds=(0, m.UB))      # block width  (= w0 unless rotated)
    m.t = pyo.Var(m.p, bounds=(0, m.UB))      # x-axis Manhattan separation
    m.s = pyo.Var(m.p, bounds=(0, m.UB))      # y-axis Manhattan separation
    m.l_f = pyo.Var(within=pyo.NonNegativeReals)  # facility length
    m.w_f = pyo.Var(within=pyo.NonNegativeReals)  # facility width

    # Facility bounds: every block lies inside the facility's bounding box.
    @m.Constraint(m.n)
    def facility_length(m, i):
        return m.l_f >= m.x[i] + m.l[i]

    @m.Constraint(m.n)
    def facility_width(m, i):
        return m.w_f >= m.y[i] + m.w[i]

    # Minimum separation: t_ij must be at least the user-set min distance.
    # Note: this only enforces the ACTIVE-disjunct axis (whichever of the
    # four left/right/above/below is selected), so the bound is conservative.
    @m.Constraint(m.p)
    def min_dist(m, i, j):
        return m.t[i, j] >= m.d[i, j]

    # Symmetry breaking: anchor block 1 left-of-and-below block 2's center.
    # Kills 4 of 8 trivial reflective symmetries; halves the search space.
    if sym == 1:
        @m.Constraint()
        def sym_1(m):
            return m.x[1] + m.l[1] / 2 <= m.x[2] + m.l[2] / 2

        @m.Constraint()
        def sym_2(m):
            return m.y[1] + m.w[1] / 2 <= m.y[2] + m.w[2] / 2

    # Objective: minimize facility size + Σ pipe-weighted Manhattan distances.
    m.obj = pyo.Objective(
        expr=m.l_f + m.w_f
             + sum(m.c[i, j] * (m.t[i, j] + m.s[i, j]) for i, j in m.p),
        sense=pyo.minimize,
    )

    # Non-overlap GDP: 4-way disjunction per pair. Each disjunct fixes the
    # spatial relationship and connects (t, s) to the active separation.
    @m.Disjunction(m.p)
    def no_overlap(m, i, j):
        return [
            # i is left of j
            [m.x[j] - (m.x[i] + m.l[i]) == m.t[i, j],
             m.y[i] - (m.y[j] + m.w[j]) <= m.s[i, j],
             m.y[j] - (m.y[i] + m.w[i]) <= m.s[i, j]],
            # i is right of j
            [m.x[i] - (m.x[j] + m.l[j]) == m.t[i, j],
             m.y[i] - (m.y[j] + m.w[j]) <= m.s[i, j],
             m.y[j] - (m.y[i] + m.w[i]) <= m.s[i, j]],
            # i is above j
            [m.y[i] - (m.y[j] + m.w[j]) == m.t[i, j],
             m.x[i] - (m.x[j] + m.l[j]) <= m.s[i, j],
             m.x[j] - (m.x[i] + m.l[i]) <= m.s[i, j]],
            # i is below j
            [m.y[j] - (m.y[i] + m.w[i]) == m.t[i, j],
             m.x[i] - (m.x[j] + m.l[j]) <= m.s[i, j],
             m.x[j] - (m.x[i] + m.l[i]) <= m.s[i, j]],
        ]

    # Rotation GDP (optional): 2-way disjunction per block.
    if rotate:
        @m.Disjunction(m.n)
        def rotation(m, i):
            return [
                [m.l[i] == m.l0[i], m.w[i] == m.w0[i]],   # default
                [m.l[i] == m.w0[i], m.w[i] == m.l0[i]],   # 90° rotated
            ]
    else:
        @m.Constraint(m.n)
        def fix_l(m, i):
            return m.l[i] == m.l0[i]

        @m.Constraint(m.n)
        def fix_w(m, i):
            return m.w[i] == m.w0[i]

    return m


def _solve_capturing(m, time_limit):
    """Run HiGHS with FD-level log capture. Returns (results, log_text).

    Uses `load_solutions=False` so the caller can decide whether the result
    is loadable based on `termination_condition` + `found_feasible_solution()`.
    See app docstring for why.
    """
    log_text = ""
    try:
        with capture_output(capture_fd=True) as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True,
                                   timelimit=int(time_limit),
                                   load_solutions=False)
        log_text = buf.getvalue()
    except TypeError:
        # Older Pyomo without capture_fd.
        with capture_output() as buf:
            solver = pyo.SolverFactory("appsi_highs")
            results = solver.solve(m, tee=True,
                                   timelimit=int(time_limit),
                                   load_solutions=False)
        log_text = buf.getvalue()
    return results, log_text


def _has_feasible(results):
    """Cross-version probe: did the solver find ANY feasible solution?
    `appsi_highs`'s LegacySolverInterface exposes this slightly differently
    across Pyomo releases. Try the documented hook first, fall back to
    bound-finiteness check."""
    fn = getattr(results, "found_feasible_solution", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    # Fallback: if the upper bound is finite, the solver has *some* feasible
    # solution — incumbent loading should work.
    try:
        ub = results.problem[0].upper_bound
        return ub is not None and ub != float("inf")
    except Exception:
        return False


def solve(n, l0, w0, cmat, d_uniform, rotate, sym, time_limit):
    """Top-level entrypoint. Returns a plain dict the UI can stash in
    session_state without holding a live Pyomo model."""

    m = build_model(n, l0, w0, cmat, d_uniform, rotate, sym)
    pyo.TransformationFactory("gdp.mbigm").apply_to(m)

    try:
        results, log = _solve_capturing(m, time_limit)
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                f"HiGHS solver not available. Run `pip install highspy`. ({e})"
            ),
            "log": "",
        }

    tc = results.solver.termination_condition

    # Status branch — load solutions only when there's something to load.
    feasible = _has_feasible(results)
    status = None
    if tc == TerminationCondition.optimal:
        m.solutions.load_from(results)
        status = "optimal"
    elif tc in (TerminationCondition.maxTimeLimit, TerminationCondition.userInterrupt):
        if feasible:
            m.solutions.load_from(results)
            status = "incumbent"
        else:
            status = "no_feasible"
    elif tc in (TerminationCondition.infeasible,
                TerminationCondition.infeasibleOrUnbounded):
        status = "infeasible"
    elif tc == TerminationCondition.unbounded:
        status = "unbounded"
    else:
        status = str(tc)

    if status not in ("optimal", "incumbent"):
        return {"status": status, "log": log}

    # Pull values out for the UI.
    blocks = []
    for i in m.n:
        blocks.append({
            "i": i,
            "x": float(pyo.value(m.x[i])),
            "y": float(pyo.value(m.y[i])),
            "l": float(pyo.value(m.l[i])),
            "w": float(pyo.value(m.w[i])),
            "rotated": bool(rotate and abs(float(pyo.value(m.l[i])) - l0[i]) > 1e-6),
        })

    pairs = []
    for (i, j) in m.p:
        pairs.append({
            "i": i, "j": j,
            "c": float(pyo.value(m.c[i, j])),
            "t": float(pyo.value(m.t[i, j])),
            "s": float(pyo.value(m.s[i, j])),
        })

    # Result-level summary numbers for the status banner.
    obj = float(pyo.value(m.obj))
    facility = (float(pyo.value(m.l_f)), float(pyo.value(m.w_f)))
    pipe_cost = sum(p["c"] * (p["t"] + p["s"]) for p in pairs)

    # Best-known bound (for gap reporting on incumbent path).
    try:
        lower_bound = results.problem[0].lower_bound
    except Exception:
        lower_bound = None

    gap = None
    if lower_bound is not None and lower_bound > 0 and obj > 0:
        gap = max(0.0, (obj - lower_bound) / max(abs(obj), 1e-12))

    return {
        "status": status,
        "blocks": blocks,
        "pairs": pairs,
        "obj": obj,
        "facility": facility,
        "pipe_cost": pipe_cost,
        "lower_bound": lower_bound,
        "gap": gap,
        "log": log,
    }


# ── 5. Visualization ─────────────────────────────────────────────────────────

# Block fill colors are driven by "connectivity": for each block i, the sum
# of its pipe costs to all other blocks. The central rack scenario produces
# one strongly-connected block (block 1) plus n-1 weakly-connected blocks,
# which the gradient surfaces clearly.
def _connectivity(blocks, pairs):
    conn = {b["i"]: 0.0 for b in blocks}
    for p in pairs:
        conn[p["i"]] = conn.get(p["i"], 0.0) + p["c"]
        conn[p["j"]] = conn.get(p["j"], 0.0) + p["c"]
    return conn


def build_layout_chart(res):
    """Multi-layered Altair chart for the optimal layout.

    Layers:
      1. Pipe overlay (lines between block centers, opacity ∝ c_ij)
      2. Block rectangles (color = connectivity)
      3. Block-id labels at centers
      4. Outer facility bounding box (dashed)
    """
    blocks = res["blocks"]
    pairs = res["pairs"]
    l_f, w_f = res["facility"]

    conn = _connectivity(blocks, pairs)
    # Block dataframe with center coordinates for the label and pipe layers.
    df_blocks = pd.DataFrame([{
        "i":   b["i"],
        "x":   b["x"],
        "y":   b["y"],
        "x2":  b["x"] + b["l"],
        "y2":  b["y"] + b["w"],
        "cx":  b["x"] + b["l"] / 2,
        "cy":  b["y"] + b["w"] / 2,
        "l":   b["l"],
        "w":   b["w"],
        "rotated": "yes" if b["rotated"] else "no",
        "connectivity": conn[b["i"]],
    } for b in blocks])

    # Layer 1 — pipe overlay. Build a row per pair with the centers of i and j
    # and the cost. Opacity scales 0..1 with c / max_c.
    max_c = max((p["c"] for p in pairs), default=0.0)
    if max_c > 0:
        pipe_rows = []
        center = {b["i"]: (b["x"] + b["l"] / 2, b["y"] + b["w"] / 2) for b in blocks}
        for p in pairs:
            if p["c"] <= 0:
                continue
            xi, yi = center[p["i"]]
            xj, yj = center[p["j"]]
            pipe_rows.append({"x": xi, "y": yi, "x2": xj, "y2": yj,
                              "c": p["c"], "pair": f"{p['i']}—{p['j']}"})
        df_pipes = pd.DataFrame(pipe_rows)
    else:
        df_pipes = pd.DataFrame(columns=["x", "y", "x2", "y2", "c", "pair"])

    # Domain spans (with a small padding) so the facility bounding box and
    # all blocks fit comfortably without clipping.
    pad = 0.05 * max(l_f, w_f, 1.0)
    x_dom = [-pad, l_f + pad]
    y_dom = [-pad, w_f + pad]

    # Outer facility bounding box (single-row dataframe for the rect layer).
    df_facility = pd.DataFrame([{"x": 0, "y": 0, "x2": l_f, "y2": w_f}])

    # ── Build the chart layers ────────────────────────────────────────────
    base = alt.Chart(df_blocks).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), title="x"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), title="y"),
    )

    # Facility bounding box — dashed outline, no fill.
    facility_box = alt.Chart(df_facility).mark_rect(
        fill=None, stroke="#374151", strokeWidth=1.5, strokeDash=[6, 4],
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_dom)),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom)),
        x2="x2:Q",
        y2="y2:Q",
    )

    # Block rectangles — fill colored by connectivity.
    block_rects = base.mark_rect(
        stroke="#1f2937", strokeWidth=1.5,
    ).encode(
        x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
        color=alt.Color("connectivity:Q",
                        scale=alt.Scale(scheme="blues"),
                        legend=alt.Legend(title="Pipe connectivity")),
        tooltip=[
            alt.Tooltip("i:O", title="Block"),
            alt.Tooltip("x:Q", format=".2f", title="x (lower-left)"),
            alt.Tooltip("y:Q", format=".2f", title="y (lower-left)"),
            alt.Tooltip("l:Q", format=".2f", title="length"),
            alt.Tooltip("w:Q", format=".2f", title="width"),
            alt.Tooltip("rotated:N", title="Rotated"),
            alt.Tooltip("connectivity:Q", format=".2f", title="Σ pipe cost"),
        ],
    )

    # Block labels at centers.
    block_labels = alt.Chart(df_blocks).mark_text(
        fontSize=14, fontWeight="bold", color="#0a0a4e",
    ).encode(
        x="cx:Q", y="cy:Q", text="i:O",
    )

    # Pipe overlay — drawn first (below blocks) so block fills sit on top.
    pipe_lines = alt.Chart(df_pipes).mark_rule(
        stroke="#dc2626",
    ).encode(
        x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q",
        size=alt.Size("c:Q", scale=alt.Scale(range=[0.5, 4]),
                      legend=alt.Legend(title="Pipe cost")),
        opacity=alt.Opacity("c:Q", scale=alt.Scale(range=[0.25, 0.85])),
        tooltip=[
            alt.Tooltip("pair:N", title="Pair"),
            alt.Tooltip("c:Q", format=".2f", title="Pipe cost"),
        ],
    ) if len(df_pipes) else alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_rule()

    chart = (
        alt.layer(facility_box, pipe_lines, block_rects, block_labels)
        .properties(height=560)
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridColor="#e5e7eb")
    )
    return chart


# ── 6. Tab renderers ─────────────────────────────────────────────────────────

def render_layout(res):
    if res is None:
        st.info("Click **Solve Optimization** in the sidebar to compute a layout.")
        return

    status = res["status"]
    if status == "no_feasible":
        st.error(f"Solver hit the time limit before finding any feasible solution. "
                 "Try a smaller `n` or a smaller min-separation distance.")
        return
    if status == "infeasible":
        st.error("The instance is infeasible. The min-separation distance is "
                 "likely too large for the chosen blocks. Try smaller values.")
        return
    if status == "unbounded":
        st.error("Unbounded — model definition is broken. This shouldn't happen.")
        return
    if status not in ("optimal", "incumbent"):
        st.warning(f"Solver finished with status: {status}")
        return

    # Status banner.
    obj = res["obj"]
    l_f, w_f = res["facility"]
    pipe_cost = res["pipe_cost"]
    facility_size = l_f + w_f

    if status == "optimal":
        st.success(
            f"**Optimal layout found.** "
            f"Objective = {obj:.2f}  ·  "
            f"facility {l_f:.1f} × {w_f:.1f} = {l_f * w_f:.1f} sq.units  ·  "
            f"pipe cost = {pipe_cost:.2f}"
        )
    else:
        gap_str = f", gap ≈ {res['gap'] * 100:.1f}%" if res["gap"] is not None else ""
        st.warning(
            f"**Incumbent (suboptimal).** Solver hit the time limit. "
            f"Best feasible objective = {obj:.2f}{gap_str}  ·  "
            f"facility {l_f:.1f} × {w_f:.1f}  ·  "
            f"pipe cost = {pipe_cost:.2f}"
        )

    # Layout chart.
    chart = build_layout_chart(res)
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "Block fill = total pipe cost incident on that block. "
        "Red lines = pipes between block centers (thicker = higher cost). "
        "Dashed rectangle = facility bounding box."
    )


def render_data(ss):
    """Editable block-dimensions table + cost matrix."""
    st.markdown("### Block dimensions")
    st.caption("Default length and width per block. Rotation (toggleable in "
               "the sidebar) lets the optimizer swap these to align block "
               "orientation with the layout.")

    n = ss["n"]
    df_dims = pd.DataFrame({
        "Block":  list(range(1, n + 1)),
        "l₀":     [ss["l0"][i] for i in range(1, n + 1)],
        "w₀":     [ss["w0"][i] for i in range(1, n + 1)],
    })
    edited_dims = st.data_editor(
        df_dims,
        hide_index=True,
        disabled=["Block"],
        column_config={
            "Block": st.column_config.NumberColumn(width="small"),
            "l₀":    st.column_config.NumberColumn(min_value=0.1, format="%.2f"),
            "w₀":    st.column_config.NumberColumn(min_value=0.1, format="%.2f"),
        },
        key="dims_editor",
    )
    # Sync edits back into session_state.
    ss["l0"] = {int(row["Block"]): float(row["l₀"]) for _, row in edited_dims.iterrows()}
    ss["w0"] = {int(row["Block"]): float(row["w₀"]) for _, row in edited_dims.iterrows()}

    st.markdown("### Pipe cost matrix")
    st.caption("Lower-triangular n×n matrix. `c[i, j]` is the per-unit-length "
               "pipe cost between blocks i and j (only the strict lower "
               "triangle, j < i, is read by the solver — upper-triangular "
               "and diagonal cells are ignored).")

    cmat = ss["cmat"]
    df_cost = pd.DataFrame(cmat, columns=[f"j={j}" for j in range(1, n + 1)])
    df_cost.index = [f"i={i}" for i in range(1, n + 1)]
    edited_cost = st.data_editor(
        df_cost,
        column_config={
            f"j={j}": st.column_config.NumberColumn(min_value=0.0, format="%.2f")
            for j in range(1, n + 1)
        },
        key="cost_editor",
    )
    # Mirror back into the lower triangle (zero everything else).
    new_cmat = [[0.0] * n for _ in range(n)]
    for ii in range(n):
        for jj in range(n):
            if ii > jj:
                v = edited_cost.iat[ii, jj]
                new_cmat[ii][jj] = float(v) if pd.notna(v) else 0.0
    ss["cmat"] = new_cmat


def render_formulation():
    img_path = Path(__file__).parent / "images" / "formulation.png"
    if img_path.exists():
        st.image(str(img_path),
                 caption="Plant facility layout — block placement schematic.",
                 use_container_width=False)

    st.markdown(r"""
### Optimal control problem

Place $n$ rectangular blocks in the 2-D plane so that the facility's
bounding-box dimensions plus the cost-weighted Manhattan pipe distances
between block centers are minimized:

$$\min \; l_f + w_f + \sum_{i,j \in N,\; j<i} c_{ij} \big( t_{ij} + s_{ij} \big)$$

subject to the facility containing every block:

$$l_f \ge x_i + l_i, \quad w_f \ge y_i + w_i \quad \forall \, i \in N$$

worst-case position bounds $x_i, y_i \le \mathrm{UB}$ where
$\mathrm{UB} = \sum_i \max(l_i, w_i)$, a minimum separation
$t_{ij} \ge d_{ij}$, and the **non-overlap disjunction** (one of four
geometric arrangements per pair) plus the **rotation disjunction**
(default vs. 90° rotated, when rotation is enabled).

### Disjunctions

For every pair $(i, j)$ with $j < i$, one of the four arrangements must hold:

$$
\bigvee_{k=1}^{4}
\begin{bmatrix}
Y_{ij}^k \\
\text{spatial relation } k \\
\text{Manhattan separations}
\end{bmatrix}
\quad \text{($k=1$ left, $2$ right, $3$ above, $4$ below)}
$$

When rotation is enabled, each block additionally chooses orientation:

$$
\begin{bmatrix} Y_i^5 \\ l_i = l_i^0 \\ w_i = w_i^0 \end{bmatrix}
\;\lor\;
\begin{bmatrix} Y_i^6 \\ l_i = w_i^0 \\ w_i = l_i^0 \end{bmatrix}
$$

### Symmetry breaking

The trivial mirror symmetries make the LP relaxation eight-fold
degenerate. We anchor block 1 to be left-of-and-below block 2's center:

$$x_1 + l_1/2 \le x_2 + l_2/2 \qquad y_1 + w_1/2 \le y_2 + w_2/2$$

This kills four of eight reflective equivalences and noticeably tightens
the LP relaxation.

### Solution method

We discretize the GDP via the **multi-Big-M transformation**
(`gdp.mbigm`), which derives a tight Big-M coefficient per constraint
from variable bounds — much tighter than uniform Big-M. The resulting
MILP is solved with **HiGHS**.

For larger instances (n > 8) the MIP can exceed the wall-clock time
limit. The app then loads the **best feasible incumbent** found before
the cutoff and reports the optimality gap. Try smaller min-separation
distances if the solver returns infeasible.

### References

[1] L. G. Papageorgiou and G. E. Rotstein, "Continuous-Domain
Mathematical Models for Optimal Process Plant Layout," *Industrial &
Engineering Chemistry Research*, vol. 37, no. 9, pp. 3631–3639, 1998.
[doi:10.1021/ie980146v](https://doi.org/10.1021/ie980146v)

[2] P. M. Castro, I. E. Grossmann, and A. Q. Novais, "Two New Continuous-Time
Models for the Scheduling of Multistage Batch Plants with Sequence-Dependent
Changeovers," *Computers & Chemical Engineering*, 2005.
[doi:10.1016/j.compchemeng.2005.06.005](https://doi.org/10.1016/j.compchemeng.2005.06.005)

[3] H. D. Sherali and J. C. Smith, "Improving discrete model
representations via symmetry considerations," *Management Science*,
vol. 47, no. 10, pp. 1396–1407, 2001.

[4] L. T. Biegler, *Nonlinear Programming: Concepts, Algorithms, and
Applications to Chemical Processes*, SIAM, 2010 (background on the
disjunctive-programming approach).
""")


def render_logs(res):
    if res is None:
        st.info("Run a solve to see HiGHS's output.")
        return
    log = res.get("log", "")
    if log.strip():
        st.code(log, language=None)
    else:
        st.info("No solver log captured. The solver may have returned before "
                "writing to stdout.")


# ── 7. Main layout ────────────────────────────────────────────────────────────

_init_state()
ss = st.session_state

# ---- Sidebar controls ----
st.sidebar.markdown(
    "## Initial Conditions &nbsp; "
    "<span style='color: rgba(49, 51, 63, 0.6); font-size: 0.875rem; "
    "font-weight: 400;'>plant layout setup</span>",
    unsafe_allow_html=True,
)

prev_n = ss["n"]
ss["n"] = st.sidebar.slider("Number of blocks, n", N_MIN, N_MAX, ss["n"], 1, key="n_slider")

prev_scenario = ss["scenario"]
ss["scenario"] = st.sidebar.radio(
    "Pipe network",
    options=["Central rack", "Random pipes", "Custom"],
    horizontal=False,
    index=["Central rack", "Random pipes", "Custom"].index(ss["scenario"]),
    key="scenario_radio",
)

# If n or scenario just changed, regenerate the data and clear stale results.
if ss["n"] != prev_n or ss["scenario"] != prev_scenario:
    if ss["scenario"] != "Custom":
        _resync_for_n_or_scenario()
    else:
        # Going custom — keep the existing data, but resize if n changed.
        _resync_for_n_or_scenario()
    ss.pop("res", None)

ss["rotate"]  = st.sidebar.checkbox("Allow rotation", value=ss["rotate"], key="rotate_box")
ss["d_min"]   = st.sidebar.slider("Min separation distance, d", 0.0, 3.0, ss["d_min"], 0.1, key="dmin_slider")

st.sidebar.button("Initialize at defaults", on_click=_initialize_at_defaults,
                  use_container_width=True,
                  help="Re-randomize block dimensions and pipe network (with "
                       "a fresh seed) and clear any prior solve.")

st.sidebar.header("Solver")
time_limit = st.sidebar.slider(
    "Solve time limit (s)",
    TIMELIMIT_MIN, TIMELIMIT_MAX, TIMELIMIT_DEFAULT, 1, key="timelimit_slider",
)

solve_btn = st.sidebar.button("Solve Optimization", type="primary",
                              use_container_width=True)

# ---- Title ----
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Facility Layout "
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/ERGO-Code/HiGHS' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>HiGHS</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([6, 3])
with _caption_col:
    st.markdown(
        "Place rectangular blocks in 2D space to minimize the facility's "
        "bounding-box dimensions plus cost-weighted Manhattan pipe distances "
        "between blocks. Edit dimensions and pipe costs in the **Data** tab, "
        "then click **Solve Optimization** in the sidebar."
    )

# ---- Solve ----
if solve_btn:
    with st.spinner(f"Running HiGHS (time limit {time_limit}s)..."):
        try:
            res = solve(
                n=ss["n"],
                l0=ss["l0"], w0=ss["w0"],
                cmat=ss["cmat"],
                d_uniform=ss["d_min"],
                rotate=ss["rotate"], sym=1,
                time_limit=time_limit,
            )
        except Exception as e:
            st.error(f"Solver error: {e}")
            st.stop()
    ss["res"] = res

# ---- Tabs ----
tab_layout, tab_data, tab_form, tab_logs = st.tabs(
    ["▶  Layout", "📊  Data", "📐  Formulation", "📋  Logs"]
)

with tab_layout:
    render_layout(ss.get("res"))

with tab_data:
    render_data(ss)

with tab_form:
    render_formulation()

with tab_logs:
    render_logs(ss.get("res"))
