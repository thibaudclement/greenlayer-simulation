import csv

# =============================================================================
# equivalence.py
# -----------------------------------------------------------------------------
# Purpose
#   Compute an "equivalent hardware upgrade %" for each simulated sleep-cap
#   setting by matching total energy consumption. In other words: for a given
#   energy outcome achieved via node sleeping, find the upgrade percentage
#   that would yield (approximately) the same total energy.
#
# Inputs (produced by simulate.py under ./out/)
#   - out/chart1_sleep_energy.csv
#       columns: sleep_percent, total_energy_Wh, avg_latency_ms, p50_latency_ms,
#                p95_latency_ms, total_drops, avg_sleep_nodes,
#                avg_alive_capacity_fraction
#   - out/chart4_upgrade_energy.csv
#       columns: upgrade_percent, total_energy_Wh, avg_latency_ms, p50_latency_ms,
#                p95_latency_ms, total_drops
#
# Output
#   - out/equivalence.csv
#       columns: sleep_cap, saving_pct, equiv_upgrade_pct
#         • sleep_cap           : sleep percentage (from the sleep sweep)
#         • saving_pct          : % energy saved vs baseline (sleep_cap = 0)
#         • equiv_upgrade_pct   : upgrade % that interpolates to the same energy
#
# Method
#   1) Load (x=percent, y=total_energy_Wh) pairs for both sleep and upgrade runs.
#   2) Establish the baseline energy from the sleep sweep at x=0.
#   3) For each sleep_cap>0, compute the relative saving vs baseline.
#   4) Interpolate along the (upgrade_percent, total_energy_Wh) curve to find
#      the upgrade% that matches the sleep run's total energy.
#
# Notes
#   - Interpolation is linear between neighboring points (piecewise).
#   - If energy lies outside the upgrade curve’s range, no match is returned
#     for that point (None).
#   - Files are expected to exist (this script assumes simulate.py already ran).
# =============================================================================


# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------
sleep_csv = "out/chart1_sleep_energy.csv"
upg_csv   = "out/chart4_upgrade_energy.csv"
out_csv   = "out/equivalence.csv"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_pairs(path, x_key, y_key):
    """
    Load a CSV as sorted (x, y) float pairs.

    Parameters
    ----------
    path : str
        CSV path.
    x_key : str
        Column name for x-values (e.g., 'sleep_percent' or 'upgrade_percent').
    y_key : str
        Column name for y-values (e.g., 'total_energy_Wh').

    Returns
    -------
    list[tuple[float, float]]
        Sorted by x ascending.
    """
    pts=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            pts.append((float(row[x_key]), float(row[y_key])))
    return sorted(pts)

def interp_x_for_y(points, target_y):
    """
    Linear interpolation: given points = [(x0,y0), (x1,y1), ...] with x sorted,
    find x such that y(x) ≈ target_y, assuming piecewise linear segments.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Monotone in x (not necessarily monotone in y).
    target_y : float
        Target y-value to match.

    Returns
    -------
    float or None
        Interpolated x if target_y lies between a segment's endpoints; else None.
    """
    for (x1,y1),(x2,y2) in zip(points, points[1:]):
        if (y1>=target_y>=y2) or (y1<=target_y<=y2):
            if y2==y1: return x1
            t = (target_y - y1)/(y2 - y1)
            return x1 + t*(x2 - x1)
    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
sleep = load_pairs(sleep_csv, "sleep_percent", "total_energy_Wh")
upg   = load_pairs(upg_csv,   "upgrade_percent", "total_energy_Wh")
baseE = next(y for x,y in sleep if x==0.0)

rows=[]
for (cap,E) in sleep:
    if cap==0: continue
    saving=(baseE-E)/baseE*100
    eq_upg = interp_x_for_y(upg,E)
    rows.append({"sleep_cap":cap,"saving_pct":saving,"equiv_upgrade_pct":eq_upg})

with open(out_csv,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["sleep_cap","saving_pct","equiv_upgrade_pct"])
    w.writeheader(); w.writerows(rows)

print(f"Wrote {out_csv}")