import csv, math

# =============================================================================
# make_weekly_profile.py
# -----------------------------------------------------------------------------
# Purpose
#   Generate a synthetic, weekly (7 × 24 hourly bins) traffic profile that
#   separates residential and business demand into two normalized time series
#   in the range [0.0, 1.0]. The output is written to `traffic_profile_us.csv`
#   with one row per hour over a 7-day week.
#
# Why this shape?
#   - Residential ("res") load: evening prime-time peak around ~21:00, late-night
#     valley ~02:00–05:00, and a modest morning ramp.
#   - Business ("biz") load: daytime-centric with a broad peak around lunch
#     (~11:00–13:00), lower outside office hours, and reduced on weekends.
#
# Normalization & interpretation
#   - Values are normalized to [0, 1] *within each curve family* (res, biz).
#     They are not intended to be directly comparable in absolute magnitude;
#     combine them later via your own weights to form a total demand model.
#
# Output schema
#   - time_bin : integer in [0, 167], sequential hour index over the week
#   - weekday  : integer in [0, 6] where 0=Mon, 1=Tue, ..., 6=Sun
#   - hour     : integer in [0, 23] local hour of day
#   - res      : float in [0, 1] residential normalized demand
#   - biz      : float in [0, 1] business normalized demand
#
# Tuning notes
#   - Weekends are detected by weekday indices (5=Sat, 6=Sun).
#   - Small DOW tweaks are applied:
#       • Sunday residential +5% (to reflect more at-home time)
#       • Wednesday/Thursday business +10% (mid-week activity bump)
#   - All values are clamped to [0.1, 1.0] (biz) or [0.4, 1.0] (res) to avoid
#     zeros that can complicate downstream multiplicative models.
#
# Reproducibility
#   - This script is deterministic (no RNG). Re-running produces identical CSVs.
# =============================================================================


# -----------------------------------------------------------------------------
# Residential demand curve
# -----------------------------------------------------------------------------
def res_curve(hour):
    """
    Shape a residential diurnal pattern with:
      - Late-night valley (~03:00)
      - Evening prime-time peak (~21:00)
      - Soft morning/afternoon plateau

    Implementation details
    ----------------------
    - Uses a raised-cosine ramp ('eve') from 15:00 → 21:00 to emulate the
      evening climb.
    - The 'night' baseline is 0.55, rising up to 1.0 with the evening ramp.
    - Extra suppression between 02:00 and 05:00 to form a late-night valley.
    - Finally clamped to [0.4, 1.0] to avoid extremely small values.

    Parameters
    ----------
    hour : int
        Local hour of day in [0, 23].

    Returns
    -------
    float
        Normalized residential load in [0.4, 1.0].
    """
    # valley ~0.45 at 03:00, peak ~1.0 at 21:00
    # use two raised cosines to mimic morning rise & evening prime time
    t = hour

    # Evening ramp: normalized 0→1 over 15:00→21:00 using a cosine ease.
    # max(0, min((t-15)/6, 1)) bounds the ramp phase to [0,1].
    eve = 0.5*(1 - math.cos(math.pi*max(0,min((t-15)/6,1))))  # ramps 15->21

    # Base + ramp: 0.55 baseline rising by up to 0.45 → peak 1.0
    night = 0.55 + 0.45*eve

    # Late night suppression (02:00–05:00) to create a valley
    if 2 <= t <= 5:
        night *= 0.75

    # Clamp to a conservative floor/ceiling to avoid zeros
    return max(0.4, min(1.0, night))


# -----------------------------------------------------------------------------
# Business demand curve
# -----------------------------------------------------------------------------
def biz_curve(hour, is_weekend):
    """
    Shape a business diurnal pattern with:
      - Working-hours emphasis (roughly 07:00–18:00)
      - Broad lunch-time peak centered near 13:00
      - Reduced levels on weekends

    Implementation details
    ----------------------
    - Baseline is 0.1 (non-working hours floor).
    - During 07:00–18:00, apply a Gaussian bump centered at 13:00 with
      sigma≈4 hours to produce a gentle day-time hill.
    - On weekends, multiply by 0.4 to reflect lower business activity.
    - Clamp to [0.1, 1.0] to avoid zeros and keep within normalized bounds.

    Parameters
    ----------
    hour : int
        Local hour of day in [0, 23].
    is_weekend : bool
        True for Saturday/Sunday, False otherwise.

    Returns
    -------
    float
        Normalized business load in [0.1, 1.0].
    """
    # up-peak ~11:00 upstream, ~13:00 downstream; off-evening; low weekends
    base = 0.1

    # Apply a smooth daytime peak within business hours
    if 7 <= hour <= 18:
        # 0.3 floor + 0.7 * Gaussian centered at 13:00 with width ~4h
        base = 0.3 + 0.7*math.exp(-((hour-13)/4.0)**2)

    # Weekend reduction
    if is_weekend:
        base *= 0.4

    # Clamp to stable bounds
    return max(0.1, min(1.0, base))


# -----------------------------------------------------------------------------
# Main CSV generation
# -----------------------------------------------------------------------------
with open("in/traffic_profile_us.csv","w",newline="") as f:
    w = csv.writer(f)
    # Column headers (see schema in the file header docstring)
    w.writerow(["time_bin","weekday","hour","res","biz"])

    t = 0  # running 0..167 hour index across the whole week

    for d in range(7):          # d: weekday index, 0=Mon ... 6=Sun
        for h in range(24):     # h: hour of day, 0..23

            # Base curves
            res = res_curve(h)
            biz = biz_curve(h, is_weekend=(d in (5,6)))  # Sat/Sun

            # Day-of-week adjustments:
            # - Sunday residential +5% (more at-home activity)
            # - Wed/Thu business +10% (mid-week business intensity)
            if d == 6:
                res *= 1.05
            if d in (2, 3):
                biz *= 1.10

            # Emit rounded values to keep the CSV compact/readable
            w.writerow([t, d, h, round(res, 3), round(biz, 3)])
            t += 1

print("Wrote traffic_profile_us.csv")
