"""gta6-twitch-ccu-forecast-01 entry: an hour-by-hour quantile forecaster for a
game launch day's Twitch category viewers, in Python 3.13 (standard library only).

The method, fitted and checked by leave-one-out over the past launches a case
carries as its `comparables` (the program re-derives everything from the case
itself, so the final case's larger comparable set is used as-is):

  For each forecast hour h, model the log ratio r_h = ln(viewers_h / viewers_1)
  with a small ridge regression on two features read from the case at its
  checkpoint (the end of hour 1):

    x = [1, min(ln viewers_1, 11.0), ln(mean viewers of hours -3..-1 / viewers_1)]

  The size term saturates near 60k first-hour viewers: below that, a small
  first hour means the big regions have not unlocked yet and the day ramps far
  above hour 1 (the pooled past launches show log ratios falling steeply with
  log size, correlation about -0.9 in the evening hours); above it, the launch
  is already live worldwide and the rest of the day follows the diurnal cycle
  around hour 1. The pre-hours term separates launches already ramping before
  the UTC day from launches still flat at their pre-release baseline.

  The spread at each hour is the empirical distribution of the regression's
  leave-one-out residuals across the comparables, read at the case's levels
  and contracted 10% toward its own median: across the past launches the raw
  residual spread over-covers (the practice board's 90% interval covered
  about 94.6% of prints), and the contraction is what leave-one-out prefers
  on the large launches.

  For large first hours the regression's quantiles are then blended (up to
  60%, phased in between ln viewers_1 of 10.5 and 12) with the pooled quantiles
  of the "hot" comparables alone: launches whose release-day peak stayed
  within 3x their hour 1, the archetype a worldwide-live launch follows.

  Anything unexpected (too few comparables, a missing field) falls back to the
  house persistence forecast for that hour, so a case always gets a valid file.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only, no network. It reads every <id>.in in
the input directory and writes <id>.out: {"hours": [{"hour": 2, "q": [q05, q25,
q50, q75, q95]}, ...]} for hours 2..24. Outside the grader's container,
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for /input and /output.
The program is a pure function of the case bytes: two runs agree byte for byte.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file could be imported (this entry needs none).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA = 0.15
SIZE_HINGE = 11.0        # ln viewers_1 where the size slope saturates (~60k)
HOT_PEAK_MAX = 3.0       # a comparable is "hot" if its day peak <= 3x hour 1
HOT_BLEND_MAX = 0.6      # largest weight given to the hot archetype
SPREAD_SCALE = 0.90      # contraction of the residual spread toward its median
MIN_ROWS = 8             # below this many training rows, use the fallback


def parse_ts(text):
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def bucket_series(points, day_start):
    """Points of any spacing -> {hour index: (viewers, channels)}, the mean of
    the points in each hour (weighted by each point's reads when present).
    Hour index 0 is [day_start, day_start + 1h); the case's hour 1."""
    t0 = parse_ts(day_start)
    acc = {}
    for p in points or []:
        try:
            idx = int((parse_ts(p["from"]) - t0).total_seconds() // 3600)
            w = float(p.get("reads") or 1)
            v = float(p["viewers"])
            ch = float(p.get("channels") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        a = acc.setdefault(idx, [0.0, 0.0, 0.0])
        a[0] += v * w
        a[1] += ch * w
        a[2] += w
    return {i: (a[0] / a[2], a[1] / a[2]) for i, a in acc.items() if a[2] > 0}


def median(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    return vals[mid] if n % 2 else (vals[mid - 1] + vals[mid]) / 2


def mean_idx(s, lo, hi):
    vals = [s[i][0] for i in range(lo, hi + 1) if i in s]
    return sum(vals) / len(vals) if vals else None


def launch_features(s):
    """Checkpoint-legal features of one launch from its hourly series
    (indices <= 0 only), or None when hour 0 is unknown."""
    if 0 not in s:
        return None
    y1 = max(1.0, s[0][0])
    quiet = [s[i][0] for i in range(-336, -120) if i in s]
    base = median(quiet)
    if base is None:
        base = median([v for i, (v, _) in s.items() if i < 0])
    base = max(1.0, base) if base else y1
    pre3 = mean_idx(s, -3, -1)
    if pre3 is None:
        pre3 = mean_idx(s, -12, -1)
    if pre3 is None:
        pre3 = base
    return {
        "y1": y1,
        "x": [1.0, min(math.log(y1), SIZE_HINGE),
              math.log(max(1.0, pre3) / y1)],
    }


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col] / M[col][col]
                for cc in range(col, n + 1):
                    M[r][cc] -= f * M[col][cc]
    return [M[i][n] / M[i][i] for i in range(n)]


def ridge_fit(X, y, lam=1e-4):
    p = len(X[0])
    A = [[sum(X[k][i] * X[k][j] for k in range(len(X)))
          + (lam if i == j and i > 0 else 0.0)
          for j in range(p)] for i in range(p)]
    b = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(p)]
    return solve(A, b)


def quantile(sorted_vals, tau):
    if not sorted_vals:
        return 0.0
    pos = tau * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100
    rounds it; a whole number comes back as an int, as the starter writes it."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def baseline_quantiles(p1, hour, levels):
    """The house persistence forecast at the case's levels: hour 1's print,
    with a spread widening like a random walk's in log space."""
    out = []
    for tau in levels:
        # invert the normal CDF approximately at the five standard levels;
        # exact for the levels this competition uses.
        z = Z[DEFAULT_LEVELS.index(tau)] if tau in DEFAULT_LEVELS else 0.0
        out.append(p1 * math.exp(z * SIGMA * math.sqrt(hour - 1)))
    return out


def train_from_comparables(case):
    """Build the training set from the case's own comparables: for each, its
    checkpoint features and its release-day log ratios. Deterministic order."""
    game_name = (case.get("game") or {}).get("name")
    train = []
    for comp in sorted(case.get("comparables") or [], key=lambda c: c.get("name") or ""):
        if game_name and comp.get("name") == game_name:
            continue
        s = bucket_series(comp.get("hourly"), comp["day_start"])
        f = launch_features(s)
        if f is None:
            continue
        actuals = {i: s[i][0] for i in range(1, 24) if i in s}
        if not actuals:
            continue
        peak_ratio = max(actuals.values()) / f["y1"]
        train.append({"x": f["x"], "y1": f["y1"], "actuals": actuals,
                      "hot": peak_ratio <= HOT_PEAK_MAX})
    return train


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))

    points = case.get("history_hourly") or case.get("history") or []
    s = bucket_series(points, case["day_start"])
    s[0] = (p1, float(first.get("channels") or 0))
    own = launch_features(s)
    train = train_from_comparables(case)

    hours = []
    for row in case.get("forecast") or [{"hour": h} for h in range(2, 25)]:
        h = int(row["hour"])
        idx = h - 1
        qs = None
        if own is not None and len(train) >= MIN_ROWS:
            rows = [(t, math.log(max(1.0, t["actuals"][idx]) / t["y1"]))
                    for t in train if idx in t["actuals"]]
            if len(rows) >= MIN_ROWS:
                X = [t["x"] for t, _ in rows]
                Y = [y for _, y in rows]
                beta = ridge_fit(X, Y)
                if beta is not None:
                    res = []
                    for j in range(len(rows)):
                        bj = ridge_fit([r for k, r in enumerate(X) if k != j],
                                       [v for k, v in enumerate(Y) if k != j])
                        if bj is not None:
                            res.append(Y[j] - sum(a * b for a, b in zip(X[j], bj)))
                    res.sort()
                    mu = sum(a * b for a, b in zip(own["x"], beta))
                    med = quantile(res, 0.5)
                    qs = [mu + med + SPREAD_SCALE * (quantile(res, tau) - med)
                          for tau in levels]
                    # Blend toward the hot archetype for large first hours.
                    w = HOT_BLEND_MAX * min(1.0, max(0.0, (math.log(p1) - 10.5) / 1.5))
                    if w > 0.0:
                        hot_rats = sorted(
                            math.log(max(1.0, t["actuals"][idx]) / t["y1"])
                            for t in train if t["hot"] and idx in t["actuals"])
                        if len(hot_rats) >= 5:
                            qs = [(1 - w) * q + w * quantile(hot_rats, tau)
                                  for q, tau in zip(qs, levels)]
        if qs is None:
            # Log ratios of the persistence forecast (its level at p1 = 1).
            qs = [math.log(q) for q in baseline_quantiles(1.0, h, levels)]
        vals = []
        for q in sorted(qs):
            v = round2(p1 * math.exp(max(-30.0, min(30.0, q))))
            vals.append(max(0.01, min(1e9, v)))
        hours.append({"hour": h, "q": vals})
    return {"hours": hours}


def baseline_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first else 1.0
    return {"hours": [{"hour": int(r["hour"]),
                       "q": [max(0.01, round2(q)) for q in
                             baseline_quantiles(p1, int(r["hour"]), levels)]}
                      for r in (case.get("forecast") or [{"hour": h} for h in range(2, 25)])]}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case)
        except Exception:
            forecast = baseline_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
