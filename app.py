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
#   - Gurobi     — MIP solver, called via Pyomo's native appsi Gurobi
#                  interface. Ships as a pip wheel (`gurobipy`); needs a
#                  Gurobi license (WLS in production via Fly secrets, or a
#                  local license file pointed to by GRB_LICENSE_FILE).
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
#   via the Big-M GDP transformation, then solved with Gurobi. Pipe
#   distances are computed by always-on dx/dy constraints, kept OUT of the
#   disjunction so the objective never depends on which spatial relation is
#   chosen — this avoids the costly continuous degeneracy that coupling
#   distance into the disjuncts would create.
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
#   3. State helpers — object-list model, add/delete, reset, randomize.
#   4. Solver — build_model + log-capturing solve + incumbent loader.
#   5. Visualization — Altair layout figure with rectangles + pipe overlay.
#   6. Tab renderers — Layout, Data, Formulation, Logs.
#   7. Main — sidebar widgets + tab assembly.
# =============================================================================

import base64
import contextlib
import io
import os
import random
import time
from pathlib import Path

import altair as alt
import pandas as pd
import pyomo.environ as pyo
import streamlit as st
from pyomo.common.errors import ApplicationError
from pyomo.gdp import Disjunction
from pyomo.opt import TerminationCondition


def _materialize_gurobi_license():
    """Production license shim. Fly secrets surface as environment
    variables, but gurobipy wants a license FILE — so if the three WLS
    values are present and no license file is configured, write one to
    the home directory and point GRB_LICENSE_FILE at it. Local dev is
    untouched: there GRB_LICENSE_FILE already points at a file on disk.
    The values never enter the repo or image — only Fly's secret store
    and the container's private filesystem."""
    if os.environ.get("GRB_LICENSE_FILE"):
        return
    access = os.environ.get("GRB_WLSACCESSID")
    secret = os.environ.get("GRB_WLSSECRET")
    license_id = os.environ.get("GRB_LICENSEID")
    if not (access and secret and license_id):
        return
    lic_path = Path.home() / "gurobi.lic"
    if not lic_path.exists():
        lic_path.write_text(
            f"WLSACCESSID={access}\n"
            f"WLSSECRET={secret}\n"
            f"LICENSEID={license_id}\n",
            encoding="utf-8",
        )
    os.environ["GRB_LICENSE_FILE"] = str(lic_path)


_materialize_gurobi_license()


# ── 1. Page config + CSS + home-logo ──────────────────────────────────────────

st.set_page_config(
    page_title="Facility Layout",
    page_icon="favicon.png",
    layout="wide",
)

# Fixed-corner home logo (no sidebar — all controls are inline on the
# Optimizer tab). Same pattern as strip-packing / diet / knapsack.
_FAVICON_DATA_URL = "data:image/png;base64," + base64.b64encode(
    (Path(__file__).parent / "favicon.png").read_bytes()
).decode()
st.markdown(
    """
    <style>
    .home-logo-corner {
        position: fixed;
        top: 0.5rem;
        left: 0.75rem;
        z-index: 999999;
    }
    .home-logo-corner img {
        width: 32px;
        height: 32px;
        border-radius: 4px;
        display: block;
    }
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.5rem !important;
    }
    </style>
    """
    f'<a href="https://griffith-pse.com" target="_self" '
    f'class="home-logo-corner">'
    f'<img src="{_FAVICON_DATA_URL}" alt="Griffith PSE — home" />'
    f"</a>",
    unsafe_allow_html=True,
)


# ── 2. Constants and defaults ─────────────────────────────────────────────────

MAX_OBJECTS = 25           # rack + up to 24 others (default instance stays 15)
MIN_OBJECTS = 2            # rack + ≥1 object (sym_1/sym_2 reference block 2)

DIM_MIN, DIM_MAX = 1, 9    # editable length / width range
DIM_RAND_MAX = 3           # Randomize draws dimensions from [1, 3]
COST_MIN, COST_MAX = 0, 9  # editable pipe-cost-to-rack range
COST_RAND_MAX = 3          # Randomize draws costs from [1, 3]

# The rack (object 1) spans the facility length: fixed long-and-thin dims, and
# always the longest object so every instance stays feasible. Reset and
# Randomize both keep these — only the other objects' dims/costs change.
RACK_LEN, RACK_WID = 9, 1
DEFAULT_N = 15             # objects present on first load / after Reset

# Minimum separation distance (integer stepper, like strip-packing's width).
D_MIN, D_MAX, D_DEFAULT = 0, 3, 1

# Time-limit presets for the inline radio (label → seconds).
_TIME_LIMITS = {"10 s": 10, "30 s": 30, "60 s": 60}

# RNG seed for Randomize; bumped each click for a fresh instance.
DEFAULT_SEED = 1

# Categorical palette — each object's index drives BOTH its editor badge color
# and its block fill in the layout, so the two views stay visually linked
# (object 1, the rack, gets the first color). Same palette as strip-packing.
_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#EECA3B",
    "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#1F77B4", "#9467BD",
]


# ── 3. State helpers ──────────────────────────────────────────────────────────
#
# The instance is an ordered list of objects. objs[0] is the rack (object 1):
# every other object has a single pipe cost to the rack, and objects don't
# connect to each other. Objects carry stable integer ids so per-row editor
# widgets keep their state across add/delete; length/width/cost map id → value
# (the rack's cost entry is unused).

def _gen_objects(seed, objs):
    """Roll dimensions and pipe costs for the NON-rack objects, leaving the
    rack (first object) pinned at its fixed RACK_LEN×RACK_WID. Used by both the
    default instance and Randomize, so neither ever resizes the rack."""
    rng = random.Random(seed)
    objs = list(objs)
    rack_id = objs[0]
    length = {rack_id: RACK_LEN}
    width = {rack_id: RACK_WID}
    cost = {rack_id: 0}
    for oid in objs[1:]:
        length[oid] = rng.randint(DIM_MIN, DIM_RAND_MAX)
        width[oid] = rng.randint(DIM_MIN, DIM_RAND_MAX)
        cost[oid] = rng.randint(1, COST_RAND_MAX)
    return objs, length, width, cost


def _default_data():
    """Initial / Reset instance: the rack plus DEFAULT_N-1 small objects."""
    return _gen_objects(DEFAULT_SEED, list(range(1, DEFAULT_N + 1)))


def _randomize_data(seed, objs):
    """Re-roll only the non-rack objects, preserving the current object count
    and row ids; the rack keeps its fixed dimensions."""
    return _gen_objects(seed, objs)


def _block_label(i):
    """Display name for block index i: the rack (object 1) reads 'rack', and
    every other object is renumbered from 1 (so block 2 → '1', block 3 → '2')."""
    return "rack" if int(i) == 1 else str(int(i) - 1)


def _set_data(objs, length, width, cost):
    ss = st.session_state
    ss["objs"], ss["length"], ss["width"], ss["cost"] = (
        list(objs), dict(length), dict(width), dict(cost)
    )


def _init_state():
    ss = st.session_state
    ss.setdefault("rotate", False)
    ss.setdefault("d_min", D_DEFAULT)
    ss.setdefault("seed", DEFAULT_SEED)
    ss.setdefault("_obj_ver", 0)
    if "objs" not in ss:
        _set_data(*_default_data())
    # Reset / Randomize set a one-shot flag and rerun; we apply it here, before
    # any editor widget is instantiated, so widget-backed keys don't clash.
    if ss.pop("_pending_reset", False):
        _set_data(*_default_data())
        ss["_obj_ver"] += 1
        ss.pop("res", None)
    if ss.pop("_pending_random", False):
        ss["seed"] += 1
        _set_data(*_randomize_data(ss["seed"], ss["objs"]))
        ss["_obj_ver"] += 1
        ss.pop("res", None)


def add_object():
    ss = st.session_state
    if len(ss["objs"]) >= MAX_OBJECTS:
        return
    new_id = (max(ss["objs"]) + 1) if ss["objs"] else 1
    ss["objs"] = ss["objs"] + [new_id]
    ss["length"] = {**ss["length"], new_id: 2}
    ss["width"] = {**ss["width"], new_id: 2}
    ss["cost"] = {**ss["cost"], new_id: 1}
    ss.pop("res", None)


def _delete_object(oid):
    """on_click for a per-row delete button. The rack (first object) can't be
    deleted, nor can the list drop below MIN_OBJECTS."""
    ss = st.session_state
    if oid == ss["objs"][0] or len(ss["objs"]) <= MIN_OBJECTS:
        return
    ss["objs"] = [i for i in ss["objs"] if i != oid]
    for key in ("length", "width", "cost"):
        ss[key] = {i: v for i, v in ss[key].items() if i != oid}
    ss.pop("res", None)


def _objs_to_inputs(ss):
    """Map the object list onto build_model's (n, l0, w0, cmat). The cost
    matrix is a star: object at display position p (≥2) gets its cost-to-rack
    in cmat[p-1][0]; everything else is zero."""
    objs = ss["objs"]
    n = len(objs)
    l0 = {p: int(ss["length"][objs[p - 1]]) for p in range(1, n + 1)}
    w0 = {p: int(ss["width"][objs[p - 1]]) for p in range(1, n + 1)}
    cmat = [[0.0] * n for _ in range(n)]
    for p in range(2, n + 1):
        cmat[p - 1][0] = float(ss["cost"][objs[p - 1]])
    return n, l0, w0, cmat


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
    m.dx = pyo.Var(m.p, bounds=(0, m.UB))     # x-axis (horizontal) edge gap
    m.dy = pyo.Var(m.p, bounds=(0, m.UB))     # y-axis (vertical) edge gap
    m.l_f = pyo.Var(within=pyo.NonNegativeReals)  # facility length
    m.w_f = pyo.Var(within=pyo.NonNegativeReals)  # facility width

    # Facility bounds: every block lies inside the facility's bounding box.
    # Length is the vertical (y) axis; width the horizontal (x) axis.
    @m.Constraint(m.n)
    def facility_length(m, i):
        return m.l_f >= m.y[i] + m.l[i]

    @m.Constraint(m.n)
    def facility_width(m, i):
        return m.w_f >= m.x[i] + m.w[i]

    # Pipe rack (block 1) spans the facility length (the vertical y-axis):
    # pinned at y=0 with the facility length fixed to the rack's length. Every
    # other object then fits within [0, l_1] in y and sits to the LEFT or RIGHT
    # of the rack (in x). The rack's x is free; only the WIDTH (horizontal x)
    # is minimized.
    m.rack_at_origin = pyo.Constraint(expr=m.y[1] == 0)
    m.facility_len_eq_rack = pyo.Constraint(expr=m.l_f == m.l[1])

    # Rectilinear edge gaps, defined GLOBALLY (not inside the disjunction):
    # dx_ij is the horizontal (x/width-axis) clearance between blocks i and j
    # (0 when they overlap in x), dy_ij the vertical (y/length-axis). They're
    # minimized in the objective, so each settles to the true gap. Keeping them
    # out of the disjuncts makes the objective independent of which spatial
    # relation is chosen — the disjunction below decides only non-overlap.
    @m.Constraint(m.p)
    def dx_lb_a(m, i, j):
        return m.dx[i, j] >= m.x[i] - (m.x[j] + m.w[j])

    @m.Constraint(m.p)
    def dx_lb_b(m, i, j):
        return m.dx[i, j] >= m.x[j] - (m.x[i] + m.w[i])

    @m.Constraint(m.p)
    def dy_lb_a(m, i, j):
        return m.dy[i, j] >= m.y[i] - (m.y[j] + m.l[j])

    @m.Constraint(m.p)
    def dy_lb_b(m, i, j):
        return m.dy[i, j] >= m.y[j] - (m.y[i] + m.l[i])

    # Symmetry breaking: anchor block 1 left-of-and-below block 2's center.
    # Kills 4 of 8 trivial reflective symmetries; halves the search space.
    if sym == 1:
        @m.Constraint()
        def sym_1(m):
            return m.x[1] + m.w[1] / 2 <= m.x[2] + m.w[2] / 2

        @m.Constraint()
        def sym_2(m):
            return m.y[1] + m.l[1] / 2 <= m.y[2] + m.l[2] / 2

    # Objective: minimize facility size + Σ pipe-weighted Manhattan distances.
    m.obj = pyo.Objective(
        expr=m.l_f + m.w_f
             + sum(m.c[i, j] * (m.dx[i, j] + m.dy[i, j]) for i, j in m.p),
        sense=pyo.minimize,
    )

    # Non-overlap GDP: 4-way disjunction per pair. Each disjunct is a single
    # inequality forcing one spatial relation with the minimum separation d
    # baked in. Distance is handled by the global dx/dy constraints above, so
    # these decide only feasibility (which pairs are separated, on which
    # axis) — never the objective.
    @m.Disjunction(m.p)
    def no_overlap(m, i, j):
        return [
            [m.x[i] + m.w[i] + m.d[i, j] <= m.x[j]],   # i left of j
            [m.x[j] + m.w[j] + m.d[i, j] <= m.x[i]],   # i right of j
            [m.y[i] + m.l[i] + m.d[i, j] <= m.y[j]],   # i below j
            [m.y[j] + m.l[j] + m.d[i, j] <= m.y[i]],   # i above j
        ]

    # Rotation GDP (optional): 2-way disjunction per block — EXCEPT block 1
    # (the rack), which keeps a fixed orientation even when rotation is on.
    # Fixing the rack is optimum-preserving: it only canonicalizes the
    # layout's overall orientation (the transpose symmetry), and every other
    # block can still rotate to recover the transposed layout at the same
    # objective. It also keeps the rack's footprint stable for the viewer.
    if rotate:
        # Rotation disjunction over the non-rack blocks (2..n) only.
        _rot_blocks = [i for i in m.n if i != 1]

        @m.Disjunction(_rot_blocks)
        def rotation(m, i):
            return [
                [m.l[i] == m.l0[i], m.w[i] == m.w0[i]],   # default
                [m.l[i] == m.w0[i], m.w[i] == m.l0[i]],   # 90° rotated
            ]

        # Block 1 (rack) is fixed in its default orientation regardless.
        m.fix_rack_l = pyo.Constraint(expr=m.l[1] == m.l0[1])
        m.fix_rack_w = pyo.Constraint(expr=m.w[1] == m.w0[1])
    else:
        @m.Constraint(m.n)
        def fix_l(m, i):
            return m.l[i] == m.l0[i]

        @m.Constraint(m.n)
        def fix_w(m, i):
            return m.w[i] == m.w0[i]

    return m


class _LicenseBusyError(RuntimeError):
    """Raised when Gurobi's WLS checkout fails even after a retry —
    typically the license's concurrent-session seats are all taken.
    solve() maps this onto the `license_busy` status."""


def _run_gurobi(m, time_limit):
    """Solve the (already GDP-transformed) MILP via the NATIVE appsi
    Gurobi interface, loading the solution onto `m` when one exists.
    Returns (termination_condition, primal, dual, log).

    The native interface (not the legacy SolverFactory("appsi_gurobi"))
    is required: the legacy wrapper's symbol-map bookkeeping crashes on
    GDP-transformed models ('DisjunctData' has no attribute 'solutions').
    Gurobi checks out a WLS seat when its environment starts; a checkout
    collision gets one quiet retry, then surfaces as license_busy, and
    the seat is always released afterward."""
    from pyomo.contrib.appsi.solvers import Gurobi as AppsiGurobi

    opt = AppsiGurobi()
    opt.config.time_limit = float(time_limit)
    opt.config.load_solution = False
    opt.config.stream_solver = True  # log into the redirected stdout
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            for attempt in (1, 2):
                try:
                    res = opt.solve(m)
                    break
                except Exception as e:
                    lowered = str(e).lower()
                    if "license" in lowered or "wls" in lowered:
                        if attempt == 1:
                            time.sleep(2.0)
                            continue
                        raise _LicenseBusyError(str(e)) from e
                    raise
            if res.best_feasible_objective is not None:
                res.solution_loader.load_vars()
    finally:
        try:
            opt.release_license()
        except Exception:
            pass

    # Scrub license-identifying lines before the log reaches the Logs tab
    # (Gurobi's WLS banner prints the license ID and registrant).
    log = "\n".join(
        ln for ln in buf.getvalue().splitlines()
        if not any(k in ln.lower()
                   for k in ("wls", "registered to", "academic license"))
    )
    # Map the appsi TerminationCondition onto the legacy enum this module
    # branches on; names match for the cases we handle, else `unknown`.
    tc = getattr(
        TerminationCondition, res.termination_condition.name,
        TerminationCondition.unknown,
    )
    return tc, res.best_feasible_objective, res.best_objective_bound, log


def solve(n, l0, w0, cmat, d_uniform, rotate, sym, time_limit):
    """Top-level entrypoint. Returns a plain dict the UI can stash in
    session_state without holding a live Pyomo model."""

    t0 = time.time()
    m = build_model(n, l0, w0, cmat, d_uniform, rotate, sym)
    pyo.TransformationFactory("gdp.bigm").apply_to(m)

    try:
        tc, primal, dual, log = _run_gurobi(m, time_limit)
    except _LicenseBusyError:
        return {
            "status": "license_busy",
            "message": (
                "The Gurobi license is busy (it allows a limited number of "
                "concurrent solves). Wait a few seconds and click Solve again."
            ),
            "log": "",
        }
    except ApplicationError as e:
        return {
            "status": "solver_missing",
            "message": (
                f"Gurobi solver not available. Run `pip install gurobipy` and "
                f"provide a license. ({e})"
            ),
            "log": "",
        }

    # Status branch. A feasible incumbent exists iff Gurobi returned a
    # primal objective.
    feasible = primal is not None
    if tc == TerminationCondition.optimal:
        status = "optimal"
    elif tc in (TerminationCondition.maxTimeLimit,
                TerminationCondition.userInterrupt):
        status = "incumbent" if feasible else "no_feasible"
    elif tc in (TerminationCondition.infeasible,
                TerminationCondition.infeasibleOrUnbounded):
        status = "infeasible"
    elif tc == TerminationCondition.unbounded:
        status = "unbounded"
    else:
        # Anything else (e.g. interrupted/unknown) still counts as an
        # incumbent if a feasible solution was loaded.
        status = "incumbent" if feasible else str(tc)

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
            "dx": float(pyo.value(m.dx[i, j])),
            "dy": float(pyo.value(m.dy[i, j])),
        })

    # Result-level summary numbers for the status banner.
    obj = float(pyo.value(m.obj))
    facility = (float(pyo.value(m.l_f)), float(pyo.value(m.w_f)))
    pipe_cost = sum(p["c"] * (p["dx"] + p["dy"]) for p in pairs)

    # Best-known bound (Gurobi's dual bound) for gap reporting on the
    # incumbent path.
    lower_bound = dual if (dual is not None and dual != float("-inf")
                           and dual == dual) else None

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
        "elapsed": time.time() - t0,
        "log": log,
    }


# ── 5. Visualization ─────────────────────────────────────────────────────────

# "Connectivity" = for each block, the sum of its incident pipe costs. No
# longer drives the fill (blocks use their palette/badge color now), but still
# feeds the block tooltip and the hover-to-highlight adjacency.
def _connectivity(blocks, pairs):
    """Per-block realized piping cost: each pipe touching a block contributes
    its cost-weighted rectilinear distance c·(dx+dy) — the same quantity summed
    into 'Total piping cost'. So an object shows its own pipe's cost and the
    rack (touched by every pipe) shows the grand total. The coefficient c alone
    isn't a cost; it ignores distance. (Preview pairs carry no dx/dy and have
    c=0, so .get keeps them at zero.)"""
    conn = {b["i"]: 0.0 for b in blocks}
    for p in pairs:
        cost = p["c"] * (p.get("dx", 0.0) + p.get("dy", 0.0))
        conn[p["i"]] = conn.get(p["i"], 0.0) + cost
        conn[p["j"]] = conn.get(p["j"], 0.0) + cost
    return conn


def _pipe_segments(block_i, block_j):
    """Two-segment L whose total drawn length equals the modeled rectilinear
    gap dx + dy: it connects the nearest edges, or runs along the shared
    overlap mid-line on an axis where the blocks overlap (gap 0). So the pipe
    drawn on screen is exactly as long as the pipe the objective costs. Width
    is along x, length along y.

    Returns a list of two segments, each {"x", "y", "x2", "y2"}.
    """
    xi, yi, wi, li = block_i["x"], block_i["y"], block_i["w"], block_i["l"]
    xj, yj, wj, lj = block_j["x"], block_j["y"], block_j["w"], block_j["l"]

    # Horizontal: connect the nearest x-edges, or the overlap mid-line (gap 0).
    if xi + wi <= xj:                      # i left of j
        src_x, dst_x = xi + wi, xj
    elif xj + wj <= xi:                    # i right of j
        src_x, dst_x = xi, xj + wj
    else:                                  # x-overlap → no horizontal run
        src_x = dst_x = (max(xi, xj) + min(xi + wi, xj + wj)) / 2

    # Vertical: connect the nearest y-edges, or the overlap mid-line.
    if yi + li <= yj:                      # i below j
        src_y, dst_y = yi + li, yj
    elif yj + lj <= yi:                    # i above j
        src_y, dst_y = yi, yj + lj
    else:                                  # y-overlap → no vertical run
        src_y = dst_y = (max(yi, yj) + min(yi + li, yj + lj)) / 2

    # L-shape: horizontal leg then vertical leg. Total length = dx + dy.
    return [
        {"x": src_x, "y": src_y, "x2": dst_x, "y2": src_y},
        {"x": dst_x, "y": src_y, "x2": dst_x, "y2": dst_y},
    ]


def build_layout_chart(res):
    """Multi-layered Altair chart for the optimal layout.

    Layers (back-to-front):
      1. Outer facility bounding box (dashed)
      2. Pipe overlay — L-shaped paths via edge-port routing, opacity ∝ c_ij,
         linked-hover dims non-hovered pipes
      3. Block rectangles (fill = connectivity), border highlights orange
         when a pipe connecting this block is hovered
      4. Block-id labels at centers

    Width is the horizontal (x) axis and length the vertical (y) axis, matching
    the formulation — so the rack, spanning the fixed length, renders
    vertically while the variable width grows horizontally (a wide layout).
    """
    blocks = res["blocks"]
    pairs = res["pairs"]
    l_f, w_f = res["facility"]

    conn = _connectivity(blocks, pairs)
    blocks_by_id = {b["i"]: b for b in blocks}

    # Adjacency map — for each block, the set of other blocks it's connected
    # to via a non-zero-cost pipe. Embedded in df_blocks as a comma-delimited
    # string (",2,3,5,") so the linked block-hover expression can test
    # membership via `indexof(...)` in Vega.
    adj = {b["i"]: set() for b in blocks}
    for p in pairs:
        if p["c"] > 0:
            adj[p["i"]].add(p["j"])
            adj[p["j"]].add(p["i"])

    df_blocks = pd.DataFrame([{
        "i":   b["i"],
        "label": _block_label(b["i"]),
        # Rack label reads top-to-bottom along its tall, thin bar; the other
        # (roughly square) blocks keep horizontal labels.
        "angle": 90 if int(b["i"]) == 1 else 0,
        "x":   b["x"],
        "y":   b["y"],
        "x2":  b["x"] + b["w"],
        "y2":  b["y"] + b["l"],
        "cx":  b["x"] + b["w"] / 2,
        "cy":  b["y"] + b["l"] / 2,
        "l":   b["l"],
        "w":   b["w"],
        "rotated": "yes" if b["rotated"] else "no",
        "connectivity": conn[b["i"]],
        "connected_str": "," + ",".join(str(x) for x in sorted(adj[b["i"]])) + ",",
        "color": _PALETTE[(int(b["i"]) - 1) % len(_PALETTE)],
    } for b in blocks])

    # Pipe dataframe — edge-port-routed L-shapes, two segments per pair.
    # Includes integer i_id/j_id columns so the linked-hover selection can
    # match against block IDs (the `pair` string is for the tooltip only).
    max_c = max((p["c"] for p in pairs), default=0.0)
    if max_c > 0:
        pipe_rows = []
        for p in pairs:
            if p["c"] <= 0:
                continue
            seg_a, seg_b = _pipe_segments(blocks_by_id[p["i"]], blocks_by_id[p["j"]])
            pair_label = f"{_block_label(p['i'])}—{_block_label(p['j'])}"
            for seg in (seg_a, seg_b):
                pipe_rows.append({
                    **seg,
                    "c": p["c"],
                    "pair": pair_label,
                    "i_id": int(p["i"]),
                    "j_id": int(p["j"]),
                })
        df_pipes = pd.DataFrame(pipe_rows)
    else:
        df_pipes = pd.DataFrame(
            columns=["x", "y", "x2", "y2", "c", "pair", "i_id", "j_id"]
        )

    # Domain spans with padding so nothing clips at the edges. Width is the
    # horizontal (x) axis, length the vertical (y) axis.
    pad = 0.05 * max(l_f, w_f, 1.0)
    x_dom = [-pad, w_f + pad]
    y_dom = [-pad, l_f + pad]

    # Equal-aspect sizing: identical pixels-per-unit on both axes, so the
    # layout is geometrically faithful and the integer ticks read as an even
    # square grid. Scale to the largest size fitting within BOTH a width and a
    # height cap — width fills the wide viz column for the usual wide-and-short
    # layout, while the height cap stops a tall, narrow instance from blowing
    # up vertically. A single scale factor keeps squares square.
    _x_span = (x_dom[1] - x_dom[0]) or 1.0
    _y_span = (y_dom[1] - y_dom[0]) or 1.0
    _max_w_px, _max_h_px = 1050, 520
    _scale = min(_max_w_px / _x_span, _max_h_px / _y_span)
    _w_px = max(160, round(_x_span * _scale))
    _h_px = max(160, round(_y_span * _scale))

    df_facility = pd.DataFrame([{"x": 0, "y": 0, "x2": w_f, "y2": l_f}])

    # ── Linked-hover selections ───────────────────────────────────────────
    # Two parallel selections: one bound to the pipe layer (`hover`), one to
    # the block layer (`block_hover`). They use different fields and feed
    # the same set of conditional expressions on each layer, so hovering
    # either a pipe OR a block produces the matched highlight pattern.
    #
    # `empty=True` is harmless here — we drive everything from explicit
    # `length(... || []) > 0` checks in the expressions below, so the
    # `empty` interpretation never affects the visible result.
    #
    # `nearest=False` (omitted) means the cursor must be directly on the
    # mark to trigger selection. No snap-from-afar.
    # The block-hover selection is always present. The pipe-hover selection
    # and its expressions exist only when there ARE pipes — otherwise the
    # stroke expression would reference a selection that was never added,
    # which breaks Vega rendering (the unsolved preview, or any all-zero-cost
    # instance, hits this).
    has_pipes = len(df_pipes) > 0
    block_hover = alt.selection_point(
        name="block_hover", on="mouseover", fields=["i"], empty=True,
    )
    if has_pipes:
        hover = alt.selection_point(
            name="hover", on="mouseover", fields=["i_id", "j_id"], empty=True,
        )
        # Pipe is bright if both selections empty, OR it's the hovered pipe,
        # OR it touches the hovered block.
        pipe_opacity_expr = (
            "(length(hover.i_id || []) === 0 && length(block_hover.i || []) === 0)"
            " || (length(hover.i_id || []) > 0 && datum.i_id === hover.i_id[0]"
            "     && datum.j_id === hover.j_id[0])"
            " || (length(block_hover.i || []) > 0 && (datum.i_id === block_hover.i[0]"
            "     || datum.j_id === block_hover.i[0]))"
        )
        # Block highlighted if it's an endpoint of the hovered pipe, IS the
        # hovered block, or is connected to it.
        block_stroke_expr = (
            "(length(hover.i_id || []) > 0 && (datum.i === hover.i_id[0]"
            "     || datum.i === hover.j_id[0]))"
            " || (length(block_hover.i || []) > 0 && (datum.i === block_hover.i[0]"
            "     || indexof(datum.connected_str, ',' + toString(block_hover.i[0]) + ',') >= 0))"
        )
    else:
        # No pipes: highlight only on direct block-hover (no pipe selection).
        block_stroke_expr = (
            "length(block_hover.i || []) > 0 && (datum.i === block_hover.i[0]"
            " || indexof(datum.connected_str, ',' + toString(block_hover.i[0]) + ',') >= 0)"
        )

    # ── Build the chart layers ────────────────────────────────────────────
    base = alt.Chart(df_blocks).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), title="x"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), title="y"),
    )

    facility_box = alt.Chart(df_facility).mark_rect(
        fill=None, stroke="#374151", strokeWidth=1.5, strokeDash=[6, 4],
    ).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=x_dom), title="x"),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_dom), title="y"),
        x2="x2:Q",
        y2="y2:Q",
    )

    # Block rectangles — stroke color/width are conditional on either the
    # pipe-hover or the block-hover selection. The block layer hosts the
    # block_hover param; the expression also references the pipe `hover`
    # selection (which lives on the pipe layer) so a single pipe-hover
    # also lights up its endpoint blocks.
    block_rects = base.mark_rect().encode(
        x2="x2:Q", y2="y2:Q",
        color=alt.Color("color:N", scale=None, legend=None),
        stroke=alt.condition(
            block_stroke_expr,
            alt.value("#f59e0b"),       # orange highlight
            alt.value("#1f2937"),        # default dark border
        ),
        strokeWidth=alt.condition(
            block_stroke_expr,
            alt.value(3.5),
            alt.value(1.5),
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Block"),
            alt.Tooltip("x:Q", format=".0f", title="x (lower-left)"),
            alt.Tooltip("y:Q", format=".0f", title="y (lower-left)"),
            alt.Tooltip("l:Q", format=".0f", title="length"),
            alt.Tooltip("w:Q", format=".0f", title="width"),
            alt.Tooltip("rotated:N", title="Rotated"),
            alt.Tooltip("connectivity:Q", format=".0f", title="Piping cost"),
        ],
    ).add_params(block_hover)

    block_labels = alt.Chart(df_blocks).mark_text(
        fontSize=14, fontWeight="bold", color="#0a0a4e",
    ).encode(
        x=alt.X("cx:Q", title="x"), y=alt.Y("cy:Q", title="y"), text="label:N",
        angle=alt.Angle("angle:Q", scale=None),   # raw degrees, no scaling
    )

    # Pipe overlay — two co-located layers. The visible layer is a thin
    # color rule with size proportional to pipe cost; the hit-target is a
    # transparent wider rule that captures the hover event and shows the
    # tooltip. Decoupling them lets us keep pipes visually thin while
    # giving the cursor a more forgiving hit zone.
    layers = [facility_box]
    if has_pipes:
        visible_pipes = alt.Chart(df_pipes).mark_rule(
            stroke="#dc2626",
        ).encode(
            x=alt.X("x:Q", title="x"), y=alt.Y("y:Q", title="y"),
            x2="x2:Q", y2="y2:Q",
            size=alt.Size("c:Q",
                          scale=alt.Scale(range=[0.5, 4]),
                          legend=None),
            opacity=alt.condition(pipe_opacity_expr, alt.value(0.9), alt.value(0.25)),
        )
        pipe_hit_targets = alt.Chart(df_pipes).mark_rule(
            stroke="transparent",
            strokeWidth=8,
        ).encode(
            x=alt.X("x:Q", title="x"), y=alt.Y("y:Q", title="y"),
            x2="x2:Q", y2="y2:Q",
            tooltip=[
                alt.Tooltip("pair:N", title="Pair (i—j)"),
                alt.Tooltip("c:Q", format=".0f", title="Pipe cost"),
            ],
        ).add_params(hover)
        layers.append(alt.layer(visible_pipes, pipe_hit_targets))
    layers.extend([block_rects, block_labels])

    # "Not Solved" badge on the initialization preview so the unsolved state
    # reads at a glance. Fixed-pixel placement (top-centre) is independent of
    # the data scale; the light plate keeps the label legible over the blocks.
    if res.get("status") == "preview":
        _one = pd.DataFrame([{"_": 0}])
        _bx, _by = _w_px / 2.0, 24.0
        plate = alt.Chart(_one).mark_rect(
            fill="white", fillOpacity=0.85, stroke="#9ca3af",
            strokeWidth=1, cornerRadius=6,
        ).encode(
            x=alt.value(_bx - 54), x2=alt.value(_bx + 54),
            y=alt.value(_by - 14), y2=alt.value(_by + 14),
        )
        badge = alt.Chart(_one).mark_text(
            text="Not Solved", fontSize=15, fontWeight="bold", color="#b91c1c",
        ).encode(x=alt.value(_bx), y=alt.value(_by))
        layers.extend([plate, badge])

    chart = (
        alt.layer(*layers)
        .properties(
            width=_w_px, height=_h_px,
            # Disable Vega-Embed's "⋮" actions menu (Save / View Source /
            # Open in Vega Editor) — vega-embed reads embed options from the
            # spec's usermeta. The Streamlit element toolbar (fullscreen /
            # show-data) is hidden via CSS in render_optimizer.
            usermeta={"embedOptions": {"actions": False}},
        )
        .configure_view(strokeOpacity=0)
        .configure_axis(grid=True, gridColor="#e5e7eb", tickMinStep=1)
    )
    return chart


# ── 6. Tab renderers ─────────────────────────────────────────────────────────

def _render_metric(slot, label, value):
    """Metric-shaped block via raw HTML — small gray label, large value.
    Matches strip-packing's top-row metrics so the five read consistently."""
    slot.markdown(
        "<div style='margin:0.25rem 0 1.3rem 0; line-height:1.2;'>"
        "<div style='font-size:0.875rem; margin-bottom:0.6rem; "
        f"white-space:nowrap;'>{label}</div>"
        "<div style='font-size:1.8rem; font-weight:400; line-height:1.1; "
        f"white-space:nowrap;'>{value}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_object_editor(ss):
    """Inline object editor (left column of the Optimizer tab): one row per
    object with Length / Width / pipe-cost-to-rack steppers. Object 1 is the
    rack — fixed in the list, no pipe-cost cell, not deletable. Add / Reset /
    Randomize below. Same fixed-slot pattern as strip-packing."""
    st.markdown(f"#### Objects (max {MAX_OBJECTS})")

    ver = ss["_obj_ver"]
    cols_spec = [0.5, 1.2, 1.2, 1.2, 0.6]

    header = st.columns(cols_spec)
    header[1].markdown("**Length**")
    header[2].markdown("**Width**")
    header[3].markdown("**Pipe cost**")

    objs = ss["objs"]
    n = len(objs)
    changed = False
    # Fixed slot count (constant element count avoids the delete ghost-row
    # flash — same reasoning as strip-packing / circle-packing).
    for slot in range(MAX_OBJECTS):
        if slot >= n:
            st.empty()
            continue
        oid = objs[slot]
        idx = slot + 1
        is_rack = (slot == 0)
        color = _PALETTE[(idx - 1) % len(_PALETTE)]
        c = st.columns(cols_spec, vertical_alignment="center")
        # Colored badge: the rack reads "rack" (a wider pill); the others are
        # numbered from 1. Color keys off the block index so the badge always
        # matches that object's fill in the layout.
        _badge_w = "padding:0 0.45rem;" if is_rack else "width:1.6rem;"
        c[0].markdown(
            f'<div style="display:inline-flex;align-items:center;'
            f'justify-content:center;{_badge_w}height:1.6rem;'
            f'border-radius:0.3rem;background:{color};color:#fff;'
            f'font-weight:700;font-size:0.85rem;white-space:nowrap;">'
            f'{_block_label(idx)}</div>',
            unsafe_allow_html=True,
        )
        new_l = c[1].number_input(
            "Length", min_value=DIM_MIN, max_value=DIM_MAX, step=1,
            value=int(ss["length"][oid]), key=f"len_{oid}_{ver}",
            label_visibility="collapsed",
        )
        new_w = c[2].number_input(
            "Width", min_value=DIM_MIN, max_value=DIM_MAX, step=1,
            value=int(ss["width"][oid]), key=f"wid_{oid}_{ver}",
            label_visibility="collapsed",
        )
        new_c = ss["cost"].get(oid, 0)
        if not is_rack:               # rack has no pipe-cost cell (left blank)
            new_c = c[3].number_input(
                "Pipe cost", min_value=COST_MIN, max_value=COST_MAX, step=1,
                value=int(ss["cost"][oid]), key=f"cost_{oid}_{ver}",
                label_visibility="collapsed",
            )
        if not is_rack and n > MIN_OBJECTS:
            c[4].button("🗑", key=f"del_{oid}_{ver}",
                        on_click=_delete_object, args=(oid,))
        if (new_l != ss["length"][oid] or new_w != ss["width"][oid]
                or (not is_rack and new_c != ss["cost"][oid])):
            ss["length"] = {**ss["length"], oid: new_l}
            ss["width"] = {**ss["width"], oid: new_w}
            if not is_rack:
                ss["cost"] = {**ss["cost"], oid: new_c}
            changed = True

    if changed:
        ss.pop("res", None)
        st.rerun()

    bcols = st.columns(3)
    if bcols[0].button("➕ Add", key="add_obj",
                       disabled=n >= MAX_OBJECTS, use_container_width=True):
        add_object()
        st.rerun()
    if bcols[1].button("↺ Reset", key="reset_obj", use_container_width=True):
        ss["_pending_reset"] = True
        st.rerun()
    if bcols[2].button("🎲 Randomize", key="rand_obj",
                       use_container_width=True):
        ss["_pending_random"] = True
        st.rerun()


def _preview_res(ss):
    """Naive 'initialized' layout for the unsolved view, consistent with the
    rack-spans-the-facility constraint: the rack sits at the origin spanning
    the facility length (vertical y), with the other objects column-packed to
    its right within that length. Costs are zeroed so no pipes draw. Shaped
    like a solve result so build_layout_chart renders it directly."""
    objs = ss["objs"]
    n = len(objs)
    gap = 1.0
    rl = float(ss["length"][objs[0]])              # rack length (along y) = facility length
    rw = float(ss["width"][objs[0]])               # rack width (along x), thin
    # Rack on the left, spanning the length (y); objects column-packed to its
    # right, stacking in y within the rack length and wrapping to a new column.
    blocks = [{"i": 1, "x": 0.0, "y": 0.0, "l": rl, "w": rw, "rotated": False}]
    x = rw + gap
    y = 0.0
    col_w = 0.0
    for p in range(2, n + 1):
        oid = objs[p - 1]
        lp, wp = float(ss["length"][oid]), float(ss["width"][oid])
        if y > 0.0 and y + lp > rl + 1e-9:          # column full → next column
            y = 0.0
            x += col_w + gap
            col_w = 0.0
        blocks.append({"i": p, "x": x, "y": y, "l": lp, "w": wp,
                       "rotated": False})
        y += lp + gap
        col_w = max(col_w, wp)
    pairs = [{"i": i, "j": j, "c": 0.0}
             for i in range(1, n + 1) for j in range(1, i)]
    return {"status": "preview", "blocks": blocks, "pairs": pairs,
            "facility": (rl, x + col_w)}


def _clear_solution():
    """Drop the stored solve so the view falls back to the initialization
    preview. Wired as the on_change for the top controls (min distance /
    rotation / time limit): changing an option invalidates the displayed
    layout, which was solved under the previous settings."""
    st.session_state.pop("res", None)


def render_optimizer(ss):
    """Main tab: object editor on the left, layout + inline controls and
    metrics on the right."""
    # Editor styling matched to strip-packing / circle-packing: tighter row
    # spacing + compact, right-aligned number fields, and click-only steppers
    # (pointer-events:none blocks click-to-type on the <input> while the +/-
    # buttons, separate elements, stay live).
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"]
            [data-testid="stHorizontalBlock"] {
            margin-bottom: -0.75rem;
            gap: 0.5rem !important;
        }
        [data-testid="stMainBlockContainer"] [data-testid="stWidgetLabel"] {
            margin-bottom: 0.25rem !important;
        }
        div[role="radiogroup"] {
            gap: 0.4rem !important;
        }
        div[role="radiogroup"] label {
            margin-right: 0 !important;
        }
        [data-testid="stNumberInputContainer"] input {
            padding-top: 0.25rem; padding-bottom: 0.25rem;
            text-align: right; padding-right: 0.4rem;
            pointer-events: none;
            user-select: none;
            caret-color: transparent;
        }
        /* Cap the field width so the value sits next to the +/- buttons
           instead of stretching across the column — compact rows, while
           still wide enough that the steppers stay visible. */
        [data-testid="stNumberInputContainer"] {
            max-width: 6.5rem;
        }
        /* Drop the on-hover chart chrome for a clean presentation view:
           Streamlit's element toolbar (fullscreen / show-data) and any
           remnant of Vega-Embed's "⋮" actions menu (also disabled via the
           chart's usermeta embedOptions in build_layout_chart). */
        [data-testid="stElementToolbar"] {
            display: none !important;
        }
        .vega-embed details,
        .vega-embed .vega-actions {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    editor_col, viz_col = st.columns([4, 8])

    with editor_col:
        _render_object_editor(ss)

    with viz_col:
        # Everything in one row (strip-packing layout): Solve / Min distance /
        # Rotation / Time limit, then the five metric slots. The layout paints
        # below through a placeholder so the controls commit before we draw.
        top = st.columns(
            [0.7, 1.5, 1.2, 1.9, 1.0, 1.0, 1.1, 0.9, 1.0],
            vertical_alignment="bottom",
        )
        with top[0]:
            solve_clicked = st.button("Solve", type="primary",
                                      use_container_width=True)
        with top[1]:
            d_min = st.number_input(
                "Min distance", min_value=D_MIN, max_value=D_MAX, step=1,
                value=int(ss["d_min"]), key="dmin_input",
                on_change=_clear_solution,
            )
        with top[2]:
            rotate = st.checkbox("Rotation", value=ss["rotate"],
                                 key="rotate_box", on_change=_clear_solution)
        with top[3]:
            time_label = st.radio(
                "Time limit", options=list(_TIME_LIMITS.keys()), index=0,
                horizontal=True, key="time_radio",
                on_change=_clear_solution,
            )
        facL_slot = top[4].empty()
        facW_slot = top[5].empty()
        pipe_slot = top[6].empty()
        gap_slot = top[7].empty()
        time_slot = top[8].empty()
        # Spacer so the plot sits a little below the controls/metrics row.
        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
        viz_slot = st.empty()
        spinner_slot = st.empty()

    ss["d_min"] = int(d_min)
    ss["rotate"] = bool(rotate)
    time_limit = _TIME_LIMITS[time_label]

    if solve_clicked:
        n, l0, w0, cmat = _objs_to_inputs(ss)
        with spinner_slot.container():
            with st.spinner(
                f"Running Gurobi (Big-M, time limit {time_limit}s)..."
            ):
                ss["res"] = solve(n, l0, w0, cmat, float(ss["d_min"]),
                                  ss["rotate"], 1, time_limit)
        spinner_slot.empty()

    res = ss.get("res")

    with viz_slot.container():
        if res is None:
            st.altair_chart(build_layout_chart(_preview_res(ss)),
                            use_container_width=False)
        elif res["status"] in ("optimal", "incumbent"):
            st.altair_chart(build_layout_chart(res), use_container_width=False)
        elif res["status"] == "no_feasible":
            st.error("Hit the time limit before finding any feasible layout. "
                     "Try fewer objects, a smaller minimum distance, or a "
                     "longer time limit.")
        elif res["status"] == "infeasible":
            st.error("Infeasible — an object may be longer than the rack "
                     "(every object must fit within the rack's length), or the "
                     "minimum separation is too large. Try shortening objects, "
                     "lengthening the rack, or reducing the distance.")
        elif res["status"] == "license_busy":
            st.error(res.get("message", "Gurobi license busy — try again."))
        elif res["status"] == "solver_missing":
            st.error(res.get("message", "Solver not available."))
        else:
            st.warning(f"Solver returned: {res['status']}")

    has = res is not None and res["status"] in ("optimal", "incumbent")
    if has:
        l_f, w_f = res["facility"]
        facL, facW = f"{l_f:.1f}", f"{w_f:.1f}"
        pipe = f"{res['pipe_cost']:.2f}"
        if res["status"] == "optimal":
            gap = "0%"
        elif res.get("gap") is not None:
            gap = f"{res['gap'] * 100:.1f}%"
        else:
            gap = "—"
        elapsed = res.get("elapsed")
        tstr = f"{elapsed:.1f}s" if elapsed is not None else "—"
    else:
        facL = facW = pipe = gap = tstr = "—"

    _render_metric(facL_slot, "Facility length", facL)
    _render_metric(facW_slot, "Facility width", facW)
    _render_metric(pipe_slot, "Total piping cost", pipe)
    _render_metric(gap_slot, "Gap", gap)
    _render_metric(time_slot, "Total time", tstr)


def render_formulation():
    img_path = Path(__file__).parent / "images" / "formulation.png"
    if img_path.exists():
        # Render at ~half width via a half-width column; the PNG is high-res,
        # so it just downscales and stays crisp.
        _img_col, _ = st.columns(2)
        with _img_col:
            st.image(str(img_path),
                     caption="Plant facility layout — block placement schematic.",
                     use_container_width=True)

    st.markdown(r"""
### Optimal control problem

Place $n$ rectangular objects so that the facility's bounding-box
dimensions plus the cost-weighted Manhattan pipe distances to the rack are
minimized. Width is the horizontal ($x$) axis, length the vertical ($y$):

$$\min \; l_f + w_f + \sum_{i,j \in N,\; j<i} c_{ij} \big( dx_{ij} + dy_{ij} \big)$$

subject to the facility containing every object (length along $y$, width
along $x$):

$$l_f \ge y_i + l_i, \quad w_f \ge x_i + w_i \quad \forall \, i \in N$$

The pipe **rack** (object 1) spans the facility length — pinned at the
origin with the facility length fixed to the rack's, so every other object
fits within $[0, l_1]$ and sits to either side of the rack:

$$y_1 = 0, \qquad l_f = l_1$$

The rectilinear edge gaps are defined by always-on constraints:

$$dx_{ij} \ge x_i - (x_j + w_j), \qquad dx_{ij} \ge x_j - (x_i + w_i)$$
$$dy_{ij} \ge y_i - (y_j + l_j), \qquad dy_{ij} \ge y_j - (y_i + l_i)$$

with worst-case position bounds $x_i, y_i \le \mathrm{UB}$ where
$\mathrm{UB} = \sum_i \max(l_i, w_i)$, plus the **non-overlap disjunction**
(one of four geometric arrangements per pair) and the **rotation
disjunction** (default vs. 90° rotated, when rotation is enabled; the rack
stays fixed).

Since $dx_{ij}, dy_{ij}$ are minimized and bounded below by both signed
gaps, each settles to the true clearance (0 when the objects overlap on
that axis). Defining them *outside* the disjunction keeps the objective
independent of which spatial relation is chosen — the disjunction's only
job is non-overlap.

### Disjunctions

For every pair $(i, j)$ with $j < i$, one of the four separations must
hold, with the minimum clearance $d_{ij}$ built in:

$$
\begin{bmatrix} Y_{ij}^1 \\ x_i + w_i + d_{ij} \le x_j \end{bmatrix}
\lor
\begin{bmatrix} Y_{ij}^2 \\ x_j + w_j + d_{ij} \le x_i \end{bmatrix}
\lor
\begin{bmatrix} Y_{ij}^3 \\ y_i + l_i + d_{ij} \le y_j \end{bmatrix}
\lor
\begin{bmatrix} Y_{ij}^4 \\ y_j + l_j + d_{ij} \le y_i \end{bmatrix}
$$

($k=1$ left, $2$ right, $3$ below, $4$ above.)

When rotation is enabled, each block additionally chooses orientation:

$$
\begin{bmatrix} Y_i^5 \\ l_i = l_i^0 \\ w_i = w_i^0 \end{bmatrix}
\;\lor\;
\begin{bmatrix} Y_i^6 \\ l_i = w_i^0 \\ w_i = l_i^0 \end{bmatrix}
$$

### Symmetry breaking

The trivial mirror symmetries make the LP relaxation eight-fold
degenerate. We anchor block 1 to be left-of-and-below block 2's center:

$$x_1 + w_1/2 \le x_2 + w_2/2 \qquad y_1 + l_1/2 \le y_2 + l_2/2$$

This kills four of eight reflective equivalences and noticeably tightens
the LP relaxation.

### Solution method

We reformulate the GDP into a MILP with the **Big-M** transformation —
one indicator per disjunct with a big constant — then solve it with
**Gurobi**. With the single-inequality disjuncts above, Big-M keeps the
model compact; the Hull (convex-hull) transformation was benchmarked too,
but it disaggregates every variable per disjunct, inflating the model for
a tighter relaxation that doesn't pay off here.

For larger instances (n > 8) the MIP can exceed the wall-clock
time limit; the app then loads the **best feasible incumbent** found
before the cutoff and reports the optimality gap. Try smaller
min-separation distances if the solver returns infeasible.

### References

[1] L. G. Papageorgiou and G. E. Rotstein, "Continuous-Domain
Mathematical Models for Optimal Process Plant Layout," *Industrial &
Engineering Chemistry Research*, vol. 37, no. 9, pp. 3631–3639, 1998.
[ACS](https://pubs.acs.org/doi/10.1021/ie980146v)

[2] J. Westerlund and L. G. Papageorgiou, "Improved Performance in
Process Plant Layout Problems Using Symmetry-Breaking Constraints,"
*Proc. FOCAPD 2004 (Foundations of Computer-Aided Process Design)*,
2004.
[PDF](https://skoge.folk.ntnu.no/prost/proceedings/focapd_2004/pdffiles/papers/075_46.pdf)

[3] N. W. Sawaya and I. E. Grossmann, "A Cutting Plane Method for
Solving Linear Generalized Disjunctive Programming Problems,"
*Computers & Chemical Engineering*, vol. 29, no. 9, pp. 1891–1913, 2005.
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0098135405000992)

[4] M. L. Bynum, G. A. Hackebeil, W. E. Hart, C. D. Laird,
B. L. Nicholson, J. D. Siirola, J.-P. Watson, and D. L. Woodruff,
*Pyomo — Optimization Modeling in Python*, 3rd ed. Cham: Springer,
2021.
[Springer](https://link.springer.com/book/10.1007/978-3-030-68928-5)
""")


def render_logs(res):
    if res is None:
        st.info("Run a solve to see Gurobi's output.")
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

# ---- Title ----
st.markdown(
    "<h2 style='margin: 0 0 0.25rem 0; padding: 0; font-size: 1.5rem; font-weight: 700;'>"
    "Facility Layout "
    "<span style='font-size: 1.15rem; font-weight: 400; color: #6b7280;'>"
    "powered by "
    "<a href='https://github.com/Pyomo/pyomo' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Pyomo</a>"
    " + "
    "<a href='https://www.gurobi.com' target='_blank' "
    "style='color: #6b7280; text-decoration: underline;'>Gurobi</a>"
    "</span>"
    "</h2>",
    unsafe_allow_html=True,
)
_caption_col, _ = st.columns([6, 3])
with _caption_col:
    st.markdown(
        "A pipe **rack** spans the facility; place the other objects on either "
        "side of it to minimize the facility's **width** plus the cost-weighted "
        "Manhattan pipe distance from each object to the rack. Edit the "
        "objects, set the options, and click **Solve**."
    )

# ---- Tabs ----
tab_opt, tab_form, tab_logs = st.tabs(
    ["▶  Optimizer", "📐  Formulation", "📋  Logs"]
)

with tab_opt:
    render_optimizer(ss)

with tab_form:
    render_formulation()

with tab_logs:
    render_logs(ss.get("res"))
