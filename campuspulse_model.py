#!/usr/bin/env python3
"""
CAMPUSPULSE - timetable-driven crowd prediction (v2: no synthetic training data)
===============================================================================

Data used
---------
  REAL  : both timetable workbooks (all 30 divisions), cross-checked against each other
  REAL  : division strength (default 70 per division, range 60-80, or your own counts)
  REAL  : lab rule - a division with 3 parallel lab cards = 3 batches, each lab holds
          strength/3 students; with 1 lab card only 1/3 of the division is busy.
  NEEDED: real facility readings (canteen / xerox / library / parking). NEITHER workbook
          contains them, so a model that outputs a crowd % cannot be *learned* until you log some.

What the script does
--------------------
  prepare  : read + cross-check both workbooks, build the free-student grid.
             Rule: a division with NO lab and NO lecture in an hour is assumed to LEAVE campus
             (--leave-fraction 1.0, default) and is not counted as free students.
  status   : show free divisions / free students for a day (the real, timetable-only part)
  predict  : crowd % for one slot. Uses LEARNED models when real logs were trained,
             otherwise a transparent timetable RULE estimate (clearly labelled).
  day      : all slots of a day (for the twin's time slider)
  template : makes the sheet to record real readings
  train    : learns from your real readings (small-data safe, never trained on synthetic data)

Usage
-----
    pip install pandas numpy scikit-learn xgboost joblib openpyxl
    python campuspulse_model.py prepare --new CAMPUSPULSE_All_Divisions_Sorted_Timetable.xlsx \
                                        --old CAMPUSPULSE_Timetable_Sorted.xlsx
    python campuspulse_model.py status  --day Thursday
    python campuspulse_model.py predict --day Thursday --time 11:00
    python campuspulse_model.py template --start 2026-10-26 --out data/facility_log.csv
    python campuspulse_model.py train   --real-logs data/facility_log.csv

    from campuspulse_model import predict       # for the twin map
    predict("Thursday", "11:00")
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
SEED = 42
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
GRID_PATH = MODEL_DIR / "grid.joblib"
BUNDLE_PATH = MODEL_DIR / "campuspulse_bundle.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
CROSSCHECK_PATH = ROOT / "data" / "processed" / "timetable_crosscheck.csv"
STRENGTH_CSV = ROOT / "data" / "division_strength.csv"      # optional: Division,student_count

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
SLOT_MIN = 30
FIRST_HOUR, LAST_HOUR = 8, 19                  # slots 08:00 ... 18:30
DEFAULT_STRENGTH = 70                          # midpoint of the stated 60-80
STRENGTH_RANGE = (60, 80)
BATCHES_PER_DIVISION = 3
LEAVE_FRACTION = 1.0                           # share of a FULLY free division (no lab, no lecture) that leaves campus
ATTENDANCE = 0.85                              # ASSUMPTION: share of a division actually present
LEVELS = {"low": 40.0, "high": 70.0}           # <40 LOW, 40-70 MEDIUM, >70 HIGH
TARGETS = ["canteen_pct", "canteen_wait", "xerox_pct", "xerox_wait", "library_pct", "parking_pct"]
PCT_TARGETS = ["canteen_pct", "xerox_pct", "library_pct", "parking_pct"]
FACILITY_LABEL = {"canteen": "Canteen", "xerox": "Xerox", "library": "Library", "parking": "Parking"}
LEVEL_NAMES = ["LOW", "MEDIUM", "HIGH"]

# Rule parameters = ASSUMPTIONS (capacities, how many free students use each facility).
# They only drive the fallback estimate and the 'rule_*' feature; training on real logs
# calibrates / corrects them. Edit them to match your campus.
RULE = {
    "canteen_capacity": 80, "canteen_dwell_min": 25,
    "canteen_use": {8: .18, 9: .18, 10: .22, 11: .30, 12: .55, 13: .55, 14: .30, 15: .25, 16: .22, 17: .20, 18: .15},
    "xerox_use": 0.05, "xerox_service_min": 3.0,
    "library_capacity": 150, "library_free_use": 0.18, "library_stay_use": 0.03,
    "parking_capacity": 600, "parking_share": 0.28, "parking_staff": 80,
}

RULE_COLS = [f"rule_{t}" for t in TARGETS]
FEATURES = [
    "day_idx", "hour_frac", "minute", "n_div_free", "n_div_left", "n_div_partial", "free_est", "on_campus_est",
    "free_ratio", "released_now", "entering_next", "in_lab_students",
    "free_SY", "free_TY", "free_Final", "is_lunch", "is_exam", "is_event",
] + RULE_COLS


# --------------------------------------------------------------------------- #
# 1. Timetables -> hourly busy table (both workbooks) + cross-check
# --------------------------------------------------------------------------- #
def year_group(div: str) -> str:
    return "SY" if div.startswith("SY") else "TY" if div.startswith("TY") else "Final"


def hourly_busy(sessions: pd.DataFrame) -> pd.DataFrame:
    """One row per (Division, Day, Hour). Theory/Tutorial = whole division busy.
    Lab only: busy fraction = lab cards / 3 batches (3 parallel labs = every batch in a lab)."""
    rows = []
    for r in sessions.itertuples():
        for h in range(int(r.Start_Hour), max(int(r.End_Hour), int(r.Start_Hour) + 1)):
            rows.append((r.Division, r.Day, h, r.Type))
    ex = pd.DataFrame(rows, columns=["Division", "Day", "Hour", "Type"])
    out = (ex.groupby(["Division", "Day", "Hour"])
             .agg(n_cards=("Type", "size"),
                  whole=("Type", lambda s: bool(s.isin(["Theory", "Tutorial"]).any())),
                  n_lab=("Type", lambda s: int((s == "Lab").sum())))
             .reset_index())
    out["busy"] = np.where(out["whole"], 1.0, np.minimum(out["n_lab"], BATCHES_PER_DIVISION) / BATCHES_PER_DIVISION)
    return out


def load_status(new_path: str, old_path: str | None, merge: str) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    keys = ["Division", "Day", "Hour"]
    new_s = pd.read_excel(new_path, sheet_name="Sessions_Sorted")
    new_s = new_s[new_s["Day"].isin(DAYS)]
    new_h = hourly_busy(new_s)

    strengths: dict[str, int] = {}
    try:
        dm = pd.read_excel(new_path, sheet_name="Division_Master")
        cnt = pd.to_numeric(dm["Student_Count"], errors="coerce")
        strengths.update({d: int(c) for d, c in zip(dm["Division"], cnt) if pd.notna(c)})
    except Exception:
        pass
    if STRENGTH_CSV.exists():
        sc = pd.read_csv(STRENGTH_CSV)
        strengths.update(dict(zip(sc["Division"], sc["student_count"].astype(int))))

    disputes = pd.DataFrame()
    status = new_h
    if old_path:
        old_s = pd.read_excel(old_path, sheet_name="Timetable_Sorted")
        old_s = old_s[old_s["Day"].isin(DAYS)]
        old_h = hourly_busy(old_s)
        cmp = new_h.merge(old_h, on=keys, how="outer", suffixes=("_new", "_old"))
        cmp["busy_new"], cmp["busy_old"] = cmp["busy_new"].fillna(0.0), cmp["busy_old"].fillna(0.0)
        disputes = cmp[(cmp["busy_new"] - cmp["busy_old"]).abs() > 1e-9].sort_values(keys)
        if merge == "union":                       # busy if either file says busy
            cmp["busy"] = cmp[["busy_new", "busy_old"]].max(axis=1)
            cmp["whole"] = cmp["whole_new"].fillna(False).astype(bool) | cmp["whole_old"].fillna(False).astype(bool)
            status = cmp[keys + ["busy", "whole"]]
        elif merge == "intersection":              # busy only where both agree it is busy
            cmp["busy"] = cmp[["busy_new", "busy_old"]].min(axis=1)
            cmp["whole"] = cmp["whole_new"].fillna(False).astype(bool) & cmp["whole_old"].fillna(False).astype(bool)
            status = cmp[cmp["busy"] > 0][keys + ["busy", "whole"]]
        # 'primary' keeps new_h (the more complete workbook)
    return status[keys + ["busy", "whole"]], strengths, disputes


def build_grid(status: pd.DataFrame, strengths: dict, leave_fraction: float = LEAVE_FRACTION) -> pd.DataFrame:
    """One row per (division, day, hour).
    on_campus  : between the division's first and last class of the day
    fully_free : on campus window but NO lecture, NO tutorial, NO lab in this hour
    present    : share of the division still on campus (fully free divisions leave -> 1 - leave_fraction)
    free_frac  : share of the division that is free AND still on campus (counted as 'free students')"""
    divs = sorted(status["Division"].unique())
    base = pd.MultiIndex.from_product([divs, DAYS, range(FIRST_HOUR - 1, LAST_HOUR + 1)],
                                      names=["Division", "Day", "Hour"]).to_frame(index=False)
    g = base.merge(status, how="left", on=["Division", "Day", "Hour"])
    g["busy"] = g["busy"].fillna(0.0)
    g["whole"] = g["whole"].fillna(False).astype(bool)
    win = status.groupby(["Division", "Day"])["Hour"].agg(first="min", last="max").reset_index()
    g = g.merge(win, how="left", on=["Division", "Day"])
    g["on_campus"] = (g["Hour"] >= g["first"]) & (g["Hour"] <= g["last"])
    g["fully_free"] = g["on_campus"] & (g["busy"] == 0)
    g["present"] = np.where(g["on_campus"], np.where(g["fully_free"], 1.0 - leave_fraction, 1.0), 0.0)
    g["free_frac"] = np.where(g["on_campus"], (1.0 - g["busy"]) * g["present"], 0.0)
    g["year_grp"] = g["Division"].map(year_group)
    g["strength"] = g["Division"].map(lambda d: strengths.get(d, DEFAULT_STRENGTH))
    g.attrs["strength_is_default"] = not strengths
    g.attrs["leave_fraction"] = leave_fraction
    return g.drop(columns=["first", "last"])


# --------------------------------------------------------------------------- #
# 2. Features + transparent rule estimate
# --------------------------------------------------------------------------- #
def slot_features(grid: pd.DataFrame, day: str, hour: int, minute: int,
                  is_exam: int = 0, is_event: int = 0, att: float = ATTENDANCE) -> dict:
    g = grid[grid["Day"] == day]
    cur = g[g["Hour"] == hour].set_index("Division")
    prev = g[g["Hour"] == hour - 1].set_index("Division")
    nxt = g[g["Hour"] == hour + 1].set_index("Division")
    on = cur[cur["on_campus"]]

    free_est = float((on["free_frac"] * on["strength"]).sum()) * att        # free AND still on campus
    on_est = float((on["present"] * on["strength"]).sum()) * att            # students assumed on campus
    fully = on[on["fully_free"]]
    released = (prev["busy"].reindex(cur.index).fillna(0) > .99) & (cur["busy"] < .99) & cur["on_campus"]
    entering = (nxt["busy"].reindex(cur.index).fillna(0) > .99) & (cur["free_frac"] > 0)
    lab_only = on[(on["busy"] > 0) & (~on["whole"])]
    by_year = (on["free_frac"] * on["strength"]).groupby(on["year_grp"]).sum() * att

    f = {
        "day_idx": DAYS.index(day), "hour_frac": hour + minute / 60.0, "minute": minute,
        "n_div_free": int(len(fully)),                                       # timetable fact: no class at all
        "n_div_left": int((fully["present"] < 1).sum()),                     # of those, assumed to have left
        "n_div_partial": int(((on["busy"] > 0) & (on["busy"] < 1)).sum()),   # some batches in lab
        "free_est": free_est, "on_campus_est": on_est,
        "free_ratio": free_est / on_est if on_est > 0 else 0.0,
        "released_now": float(cur.loc[released[released].index, "strength"].sum() * att),
        "entering_next": float(cur.loc[entering[entering].index, "strength"].sum() * att),
        "in_lab_students": float((lab_only["busy"] * lab_only["strength"]).sum() * att),
        "free_SY": float(by_year.get("SY", 0.0)), "free_TY": float(by_year.get("TY", 0.0)),
        "free_Final": float(by_year.get("Final", 0.0)),
        "is_lunch": int(12 <= hour < 15), "is_exam": is_exam, "is_event": is_event,
    }
    f.update({f"rule_{k}": v for k, v in rule_estimate(f).items()})
    return f


def rule_estimate(f: dict) -> dict:
    """Transparent demand model: free students x usage rate x time at facility / capacity."""
    h, free, on_c = int(f["hour_frac"]), f["free_est"], f["on_campus_est"]
    exam, event = f.get("is_exam", 0), f.get("is_event", 0)
    rush = 1.25 if f["minute"] == 0 else 0.85                       # rush when classes end
    users = free * RULE["canteen_use"].get(h, .2) * rush * (0.9 if exam else 1) * (1.2 if event else 1)
    c_pct = users * RULE["canteen_dwell_min"] / 60 / RULE["canteen_capacity"] * 100
    lam = free * RULE["xerox_use"] * (1.8 if exam else 1) / 60       # jobs per minute
    rho = min(lam * RULE["xerox_service_min"], 0.98)
    x_wait = min(RULE["xerox_service_min"] * rho / (1 - rho), 30)
    l_pct = (free * RULE["library_free_use"] + on_c * RULE["library_stay_use"]) * (1.8 if exam else 1) / RULE["library_capacity"] * 100
    p_pct = (on_c * RULE["parking_share"] * (1.1 if event else 1) + RULE["parking_staff"]) / RULE["parking_capacity"] * 100
    c_pct, l_pct, p_pct = (float(np.clip(v, 0, 100)) for v in (c_pct, l_pct, p_pct))
    return {"canteen_pct": c_pct, "canteen_wait": float(np.clip(22 * (c_pct / 100) ** 2.2, 0, 40)),
            "xerox_pct": float(rho * 100), "xerox_wait": float(x_wait),
            "library_pct": l_pct, "parking_pct": p_pct}


def slot_list() -> list[tuple[int, int]]:
    return [(h, m) for h in range(FIRST_HOUR, LAST_HOUR) for m in range(0, 60, SLOT_MIN)]


# --------------------------------------------------------------------------- #
# 3. Real facility logs -> training table
# --------------------------------------------------------------------------- #
def logs_to_table(grid: pd.DataFrame, path: str) -> pd.DataFrame:
    lg = pd.read_csv(path)
    need = {"date", "time"}
    if not need <= set(lg.columns):
        sys.exit(f"Log file needs columns {sorted(need)} plus any of {TARGETS}")
    rows = []
    for r in lg.to_dict("records"):
        try:
            day = pd.Timestamp(r["date"]).day_name()
            hh, mm = (int(x) for x in str(r["time"]).split(":")[:2])
        except Exception:
            continue
        mm = (mm // SLOT_MIN) * SLOT_MIN
        if day not in DAYS or not (FIRST_HOUR <= hh < LAST_HOUR):
            continue
        f = slot_features(grid, day, hh, mm, int(r.get("is_exam", 0) or 0), int(r.get("is_event", 0) or 0))
        f["date"] = str(r["date"])[:10]
        for t in TARGETS:
            v = pd.to_numeric(r.get(t), errors="coerce")
            f[t] = float(v) if pd.notna(v) else np.nan
        rows.append(f)
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("No usable rows in the log file (check date=YYYY-MM-DD, time=HH:MM, Mon-Sat, 08:00-18:59).")
    return df


def level_of(pct) -> np.ndarray:
    p = np.asarray(pct, dtype=float)
    return np.where(p < LEVELS["low"], 0, np.where(p < LEVELS["high"], 1, 2))


def scores(y, pred, target: str) -> dict:
    out = {"MAE": round(float(mean_absolute_error(y, pred)), 3),
           "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
           "R2": round(float(r2_score(y, pred)), 4) if len(y) > 2 else None}
    if target in PCT_TARGETS:
        out["level_accuracy"] = round(float((level_of(y) == level_of(pred)).mean()), 4)
        out["level_macro_F1"] = round(float(f1_score(level_of(y), level_of(pred), average="macro")), 4)
    return out


def candidates(n_rows: int, target: str) -> dict:
    """Small-data-safe model set. Every model also sees the rule_* columns (physics prior)."""
    rule_col = f"rule_{target}"
    c = {"calibrated_rule": make_pipeline(
        ColumnTransformer([("r", "passthrough", [rule_col])]), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0))}
    if n_rows >= 60:
        c["random_forest"] = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=3,
                                                   n_jobs=-1, random_state=SEED)
    if HAVE_XGB and n_rows >= 300:
        c["xgboost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
                                    colsample_bytree=0.8, tree_method="hist", random_state=SEED, n_jobs=-1)
    return c


def train(real_logs: str, min_rows: int) -> None:
    grid = joblib.load(GRID_PATH)
    df = logs_to_table(grid, real_logs)
    print(f"Real log table: {len(df)} readings over {df['date'].nunique()} days")
    models, report = {}, {}
    for t in TARGETS:
        d = df.dropna(subset=[t])
        if len(d) < min_rows:
            print(f"  {t:13s} only {len(d)} readings (< {min_rows}) -> keeping the timetable rule")
            continue
        X, y, groups = d[FEATURES], d[t], d["date"]
        k = max(2, min(5, groups.nunique()))
        cv = GroupKFold(n_splits=k)                    # whole days held out together
        results = {}
        base_pred = cross_val_predict(DummyRegressor(strategy="mean"), X, y, groups=groups, cv=cv)
        results["baseline_mean"] = scores(y, base_pred, t)
        results["rule_uncalibrated"] = scores(y, d[f"rule_{t}"], t)
        preds = {}
        for name, mdl in candidates(len(d), t).items():
            preds[name] = cross_val_predict(mdl, X, y, groups=groups, cv=cv)
            results[name] = scores(y, preds[name], t)
        best = min(preds, key=lambda n: results[n]["MAE"])
        ref = min(results["baseline_mean"]["MAE"], results["rule_uncalibrated"]["MAE"])
        gain = 100 * (1 - results[best]["MAE"] / max(ref, 1e-9))
        promoted = gain > 0
        report[t] = {"rows": len(d), "cv_folds": k, "best_model": best, "promoted": promoted,
                     "improvement_over_best_reference_pct": round(gain, 1), "cv_scores": results}
        if promoted:
            final = candidates(len(d), t)[best].fit(X, y)
            models[t] = final
            if hasattr(final, "feature_importances_"):
                imp = sorted(zip(FEATURES, final.feature_importances_), key=lambda x: -x[1])[:6]
                report[t]["top_features"] = {a: round(float(b), 3) for a, b in imp}
        print(f"  {t:13s} rows={len(d):4d} best={best:15s} CV MAE={results[best]['MAE']:6.2f} "
              f"(mean-baseline {results['baseline_mean']['MAE']:6.2f}, rule {results['rule_uncalibrated']['MAE']:6.2f}) "
              f"{'-> LEARNED model used' if promoted else '-> does not beat references; rule kept'}")
    if not models:
        sys.exit("\nNo target had enough real data or beat the references. Log more readings and retrain.")
    margins = {t: report[t]["cv_scores"][report[t]["best_model"]]["MAE"] for t in models}
    joblib.dump({"models": models, "features": FEATURES, "grid": grid, "margins": margins}, BUNDLE_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {BUNDLE_PATH.name} and {METRICS_PATH.name} (learned targets: {sorted(models)})")


# --------------------------------------------------------------------------- #
# 4. Prediction
# --------------------------------------------------------------------------- #
_STATE: dict = {}


def _state() -> tuple[pd.DataFrame, dict | None]:
    if not _STATE:
        if BUNDLE_PATH.exists():
            b = joblib.load(BUNDLE_PATH)
            _STATE.update(grid=b["grid"], bundle=b)
        elif GRID_PATH.exists():
            _STATE.update(grid=joblib.load(GRID_PATH), bundle=None)
        else:
            raise FileNotFoundError("Run `prepare` first (it reads the timetable workbooks).")
    return _STATE["grid"], _STATE["bundle"]


def _message(fac: str, level: str, wait: float | None) -> str:
    if fac == "library":
        return {"LOW": "Plenty of seats available.", "MEDIUM": "Seats filling up.", "HIGH": "Very few seats left."}[level]
    if fac == "parking":
        return {"LOW": "Plenty of parking space.", "MEDIUM": "Parking moderately full.", "HIGH": "Parking nearly full."}[level]
    base = {"LOW": "Little to no wait", "MEDIUM": "Moderate queue", "HIGH": "Crowded"}[level]
    return base + (f" (~{round(wait)} min wait)" if level != "LOW" and wait is not None and wait >= 2 else "")


def predict(day: str, time: str, is_exam: int = 0, is_event: int = 0) -> dict:
    """Crowd prediction for one 30-min slot ('HH:MM'). JSON-ready for the twin map."""
    grid, bundle = _state()
    day = day.capitalize()
    if day not in DAYS:
        return {"day": day, "time": time, "campus_open": False, "note": "No classes on this day."}
    hh, mm = (int(x) for x in time.split(":")[:2])
    mm = (mm // SLOT_MIN) * SLOT_MIN
    if not (FIRST_HOUR <= hh < LAST_HOUR):
        return {"day": day, "time": f"{hh:02d}:{mm:02d}", "campus_open": False, "note": "Outside college hours."}

    f = slot_features(grid, day, hh, mm, is_exam, is_event)
    rule = rule_estimate(f)
    X = pd.DataFrame([f])[FEATURES]
    facilities, learned_any = {}, False
    for fac in FACILITY_LABEL:
        learned = bundle is not None and f"{fac}_pct" in bundle["models"]
        pct = float(bundle["models"][f"{fac}_pct"].predict(X)[0]) if learned else rule[f"{fac}_pct"]
        pct = float(np.clip(pct, 0, 100))
        wt = f"{fac}_wait"
        if wt in TARGETS:
            wait = float(bundle["models"][wt].predict(X)[0]) if (bundle and wt in bundle["models"]) else rule[wt]
            wait = float(np.clip(wait, 0, 40))
        else:
            wait = None
        lvl = LEVEL_NAMES[int(level_of(pct))]
        entry = {"crowd_pct": round(pct), "level": lvl, "wait_min": None if wait is None else round(wait),
                 "message": _message(fac, lvl, wait), "method": "learned_from_real_logs" if learned else "timetable_rule"}
        if learned:
            m = bundle["margins"][f"{fac}_pct"]
            entry["range_pct"] = [max(0, round(pct - m)), min(100, round(pct + m))]
        facilities[fac] = entry
        learned_any |= learned

    cur = grid[(grid["Day"] == day) & (grid["Hour"] == hh) & grid["fully_free"]]
    free = f["free_est"]
    res = {"day": day, "time": f"{hh:02d}:{mm:02d}", "campus_open": True,
           "free_divisions": f["n_div_free"], "free_division_names": sorted(cur["Division"].tolist()),
           "divisions_assumed_left": f["n_div_left"], "partly_free_divisions": f["n_div_partial"],
           "est_free_students": int(round(free)), "campus_activity": campus_activity(free),
           "facilities": facilities}
    if grid.attrs.get("strength_is_default", True):
        s_lo, s_hi = STRENGTH_RANGE[0] / DEFAULT_STRENGTH, STRENGTH_RANGE[1] / DEFAULT_STRENGTH
        res["est_free_students_range"] = [int(round(free * s_lo)), int(round(free * s_hi))]
    if not learned_any:
        res["note"] = ("Estimated from the timetable by a transparent rule (not learned). "
                       "Train on real readings with `train --real-logs` to get learned, calibrated predictions.")
    return res


def campus_activity(free_students: float) -> str:
    """LOW / MEDIUM / HIGH relative to this timetable: 40th and 75th percentile of free students."""
    lo, hi = _state()[0].attrs.get("act_thresholds", (150.0, 300.0))
    return "LOW" if free_students < lo else "MEDIUM" if free_students < hi else "HIGH"


def predict_day(day: str, **kw) -> list[dict]:
    return [predict(day, f"{h:02d}:{m:02d}", **kw) for h, m in slot_list()]


# --------------------------------------------------------------------------- #
# 5. Commands
# --------------------------------------------------------------------------- #
def cmd_prepare(new: str, old: str | None, merge: str, leave_fraction: float) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CROSSCHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    status, strengths, disputes = load_status(new, old, merge)
    grid = build_grid(status, strengths, leave_fraction)
    vals = [slot_features(grid, d, h, 0)["free_est"] for d in DAYS for h in range(FIRST_HOUR, LAST_HOUR)]
    vals = [v for v in vals if v > 0]
    grid.attrs["act_thresholds"] = (float(np.percentile(vals, 40)), float(np.percentile(vals, 75))) if vals else (150.0, 300.0)
    joblib.dump(grid, GRID_PATH)
    if BUNDLE_PATH.exists():
        BUNDLE_PATH.unlink()                      # timetable changed -> old learned models are stale
    _STATE.clear()
    n_div = grid["Division"].nunique()
    print(f"Grid built from {'both workbooks (' + merge + ')' if old else 'one workbook'}: {n_div} divisions")
    print(f"Leave rule: {leave_fraction:.0%} of a fully free division (no lab, no lecture) is assumed to leave campus")
    print("Division strength:", "default 70 (range 60-80)" if grid.attrs["strength_is_default"]
          else f"{len(strengths)} divisions have real counts, others default 70")
    if old:
        disputes.to_csv(CROSSCHECK_PATH, index=False)
        total = len(status)
        print(f"Cross-check: {len(disputes)} division-hour cells differ between the two workbooks "
              f"({100 * len(disputes) / max(total, 1):.1f}%) -> {CROSSCHECK_PATH}")
        print("  Verify those cells in the PDF; the rest are confirmed by both files.")


def cmd_status(day: str) -> None:
    for r in predict_day(day):
        if r.get("campus_open"):
            print(f"{r['time']}  fully free={r['free_divisions']:2d} (left={r['divisions_assumed_left']:2d})  "
                  f"partly free={r['partly_free_divisions']:2d}  counted free students~{r['est_free_students']:4d}  "
                  f"activity={r['campus_activity']}")


def cmd_template(start: str, days: int, times: str, out: str) -> None:
    grid, _ = _state()
    rows, d0 = [], pd.Timestamp(start)
    for i in range(days * 2):
        d = d0 + pd.Timedelta(days=i)
        if d.day_name() not in DAYS:
            continue
        for t in times.split(","):
            hh, mm = (int(x) for x in t.split(":"))
            f = slot_features(grid, d.day_name(), hh, mm)
            rows.append({"date": d.date().isoformat(), "time": t, "day": d.day_name(),
                         "free_divisions_expected": f["n_div_free"], "est_free_students_expected": round(f["free_est"]),
                         "canteen_pct": "", "canteen_wait": "", "xerox_pct": "", "xerox_wait": "",
                         "library_pct": "", "parking_pct": "", "is_exam": 0, "is_event": 0, "notes": ""})
        if len({r["date"] for r in rows}) >= days:
            break
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} blank rows -> {out}. Fill the 6 reading columns, then run train --real-logs {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="CAMPUSPULSE")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--new", required=True, help="CAMPUSPULSE_All_Divisions_Sorted_Timetable.xlsx")
    p.add_argument("--old", help="CAMPUSPULSE_Timetable_Sorted.xlsx (used to cross-check)")
    p.add_argument("--leave-fraction", type=float, default=LEAVE_FRACTION,
                   help="share of a fully free division assumed to leave campus (1.0 = all leave, 0 = all stay)")
    p.add_argument("--merge", choices=["primary", "union", "intersection"], default="primary",
                   help="primary = trust --new (default); union = busy if either file says busy")
    s = sub.add_parser("status"); s.add_argument("--day", required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--day", required=True); pr.add_argument("--time", required=True)
    pr.add_argument("--exam", type=int, default=0); pr.add_argument("--event", type=int, default=0)
    d = sub.add_parser("day"); d.add_argument("--day", required=True)
    t = sub.add_parser("template")
    t.add_argument("--start", required=True); t.add_argument("--days", type=int, default=6)
    t.add_argument("--times", default="09:30,11:30,12:30,13:30,15:30,17:00")
    t.add_argument("--out", default="data/facility_log.csv")
    tr = sub.add_parser("train")
    tr.add_argument("--real-logs", required=True)
    tr.add_argument("--min-rows", type=int, default=30)
    a = ap.parse_args()
    if a.cmd == "prepare":
        cmd_prepare(a.new, a.old, a.merge, a.leave_fraction)
    elif a.cmd == "status":
        cmd_status(a.day.capitalize())
    elif a.cmd == "predict":
        print(json.dumps(predict(a.day, a.time, a.exam, a.event), indent=2))
    elif a.cmd == "day":
        print(json.dumps(predict_day(a.day.capitalize()), indent=1))
    elif a.cmd == "template":
        cmd_template(a.start, a.days, a.times, a.out)
    else:
        train(a.real_logs, a.min_rows)


if __name__ == "__main__":
    main()
