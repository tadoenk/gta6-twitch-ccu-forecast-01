#!/usr/bin/env python3
"""
GTA6 Twitch CCU Forecast program.
Deterministic program forecasting launch-day viewers (hours 2 to 24) using anchor points,
comparable pool filtering, OLS regression, and Huber Iteratively Reweighted Least Squares (IRLS)
refinement. The Huber IRLS procedure iteratively reweights residuals (using weight 1 when
absolute residual <= 1.5*s, and 1.5*s/|e_i| otherwise, with s as the root mean square deviation
divided by n) over up to two iterations, stopping early if residual spread s <= 1e-12,
to produce robust median forecasts and residual spreads unaffected by outliers.

Fallback rules:
- Starter baseline formula applies when there are no usable members (n = 0) or when the case cannot be parsed / is not a JSON object.
- Median log-ratio method applies when there are 1 or 2 usable members (n = 1 or n = 2).
"""

import os
import glob
import json
import math

Z_VALS = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
TAU_VALS = [0.05, 0.25, 0.50, 0.75, 0.95]

def median(lst):
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    else:
        return (s[mid - 1] + s[mid]) / 2.0

def get_viewers(item):
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for k in ["viewers", "viewers_count"]:
            if k in item and item[k] is not None:
                try:
                    v = float(item[k])
                    if not math.isnan(v) and not math.isinf(v):
                        return v
                except (ValueError, TypeError):
                    pass
    return None

def extract_anchor(case):
    # 1. Check "known" - if list, use LAST element
    known = case.get("known")
    if isinstance(known, list) and len(known) > 0:
        v = get_viewers(known[-1])
        if v is not None and v > 0:
            return v
    elif isinstance(known, dict):
        v = get_viewers(known)
        if v is not None and v > 0:
            return v

    # 2. Check last entry of "history_hourly"
    history = case.get("history_hourly")
    if isinstance(history, list) and len(history) > 0:
        v = get_viewers(history[-1])
        if v is not None and v > 0:
            return v

    # 3. Fallback
    return 1.0

def get_game_name(case):
    game = case.get("game")
    if isinstance(game, dict):
        return str(game.get("name", "")).strip().lower()
    elif isinstance(game, str):
        return game.strip().lower()
    return ""

def get_unlock_text(comp):
    for field in ["unlock", "unlock_description", "unlock_text", "description"]:
        val = comp.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return ""

def baseline_forecast(p, h):
    # q_tau = p * exp(z_tau * 0.15 * sqrt(h - 1))
    res = []
    for z in Z_VALS:
        val = p * math.exp(z * 0.15 * math.sqrt(h - 1))
        val = max(1.0, min(1000000000.0, val))
        res.append(val)
    # Enforce non-decreasing
    res.sort()
    return res

def process_case(case):
    p = extract_anchor(case)
    if p <= 0 or math.isnan(p) or math.isinf(p):
        p = 1.0

    own_game = get_game_name(case)
    comparables = case.get("comparables", [])
    if not isinstance(comparables, list):
        comparables = []

    # Filter out own game
    pool = []
    for comp in comparables:
        if not isinstance(comp, dict):
            continue
        c_name = ""
        c_game = comp.get("game")
        if isinstance(c_game, dict):
            c_name = str(c_game.get("name", "")).strip().lower()
        elif isinstance(c_game, str):
            c_name = c_game.strip().lower()
        if not c_name:
            c_name = str(comp.get("name", "")).strip().lower()
        
        if own_game and c_name == own_game:
            continue
        pool.append(comp)

    # Filter by unlock text mentions "midnight local" or "rolling"
    filtered_pool = []
    for comp in pool:
        ut = get_unlock_text(comp)
        if "midnight local" in ut or "rolling" in ut:
            filtered_pool.append(comp)

    if len(filtered_pool) < 6:
        # use all comparables except own game
        filtered_pool = pool

    # Kept members
    kept_members = []
    for comp in filtered_pool:
        day_start = str(comp.get("day_start", ""))[:10]
        hourly_data = comp.get("hourly") or comp.get("hourly_points") or comp.get("points")
        if not isinstance(hourly_data, list):
            continue
        
        hours_dict = {}
        for pt in hourly_data:
            if not isinstance(pt, dict):
                continue
            # Read timestamp from 'from' field (or fallback to 'timestamp')
            ts_val = pt.get("from")
            if not ts_val:
                ts_val = pt.get("timestamp", "")
            ts = str(ts_val)
            if len(ts) < 10:
                continue
            t_date = ts[:10]
            if t_date != day_start:
                continue
            
            hour_val = None
            if "T" in ts:
                time_part = ts.split("T")[1]
                try:
                    h_part = int(time_part[:2])
                    hour_val = h_part + 1
                except:
                    pass
            if hour_val is None:
                if "hour" in pt:
                    try:
                        hour_val = int(pt["hour"])
                    except:
                        pass
            
            if hour_val is not None and 1 <= hour_val <= 24:
                v = get_viewers(pt)
                if v is not None:
                    v = max(1.0, v)
                    hours_dict[hour_val] = v

        if 1 in hours_dict and len(hours_dict) >= 20:
            kept_members.append(hours_dict)

    hours_output = []
    ln_p = math.log(p)

    for h in range(2, 25):
        # Gather points (x = ln(hour-1 viewers), y = ln(hour-h viewers))
        xs = []
        ys = []
        for km in kept_members:
            if 1 in km and h in km:
                xs.append(math.log(km[1]))
                ys.append(math.log(km[h]))
        
        n = len(xs)
        if n == 0:
            q_vals = baseline_forecast(p, h)
        elif n in (1, 2):
            ratios = [ys[i] - xs[i] for i in range(n)]
            m = median(ratios)
            var_sum = sum((r - m) ** 2 for r in ratios)
            s = math.sqrt(var_sum / n)
            q_vals = []
            median_val = p * math.exp(m)
            for z in Z_VALS:
                val = median_val * math.exp(z * s)
                val = max(1.0, min(1000000000.0, val))
                q_vals.append(val)
            q_vals.sort()
        else:
            # n >= 3: OLS then up to two Huber IRLS iterations
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            sxx = sum((x - mean_x) ** 2 for x in xs)
            sxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))

            if sxx < 1e-12:
                b = 0.0
                a = mean_y
            else:
                b = sxy / sxx
                a = mean_y - b * mean_x

            # Huber IRLS refinement: at most two iterations, recomputing residuals and s at each step
            for _ in range(2):
                residuals = [ys[i] - (a + b * xs[i]) for i in range(n)]
                var_e = sum(e ** 2 for e in residuals) / n
                s = math.sqrt(var_e)
                if s <= 1e-12:
                    break
                
                weights = []
                for e in residuals:
                    abs_e = abs(e)
                    if abs_e <= 1.5 * s:
                        weights.append(1.0)
                    else:
                        weights.append(1.5 * s / abs_e)

                sum_w = sum(weights)
                if sum_w <= 1e-12:
                    break
                mean_xw = sum(weights[i] * xs[i] for i in range(n)) / sum_w
                mean_yw = sum(weights[i] * ys[i] for i in range(n)) / sum_w

                sxx_w = sum(weights[i] * (xs[i] - mean_xw) ** 2 for i in range(n))
                sxy_w = sum(weights[i] * (xs[i] - mean_xw) * (ys[i] - mean_yw) for i in range(n))

                if sxx_w < 1e-12:
                    b = 0.0
                    a = mean_yw
                else:
                    b = sxy_w / sxx_w
                    a = mean_yw - b * mean_xw

            # Final median forecast and residual spread recomputed from final residuals after loop
            median_val = math.exp(a + b * ln_p)
            final_residuals = [ys[i] - (a + b * xs[i]) for i in range(n)]
            final_s = math.sqrt(sum(e ** 2 for e in final_residuals) / n)

            q_vals = []
            for z in Z_VALS:
                val = median_val * math.exp(z * final_s)
                val = max(1.0, min(1000000000.0, val))
                q_vals.append(val)
            q_vals.sort()

        hours_output.append({
            "hour": h,
            "q": [v for v in q_vals]  # Full float precision, no rounding
        })

    return {"hours": hours_output}

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR")
    if not input_dir or not os.path.exists(input_dir):
        if os.path.exists("/input"):
            input_dir = "/input"
        elif os.path.exists("./input"):
            input_dir = "./input"
        else:
            input_dir = "."

    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)

    in_files = glob.glob(os.path.join(input_dir, "*.in"))

    for in_path in in_files:
        filename = os.path.basename(in_path)
        stem = os.path.splitext(filename)[0]
        
        case_id = stem
        parsed_ok = False
        case_data = None

        try:
            with open(in_path, "r", encoding="utf-8") as f:
                case_data = json.load(f)
            if isinstance(case_data, dict):
                parsed_ok = True
                if "id" in case_data and case_data["id"] is not None:
                    case_id = str(case_data["id"])
        except Exception:
            pass

        if not parsed_ok or not isinstance(case_data, dict):
            # Fallback forecast
            hours_output = []
            for h in range(2, 25):
                q_vals = baseline_forecast(1.0, h)
                hours_output.append({
                    "hour": h,
                    "q": [v for v in q_vals]
                })
            output_data = {"hours": hours_output}
        else:
            output_data = process_case(case_data)

        out_path = os.path.join(output_dir, f"{case_id}.out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, separators=(',', ':'))

if __name__ == "__main__":
    main()
