#!/usr/bin/env python3
"""
This program forecasts GTA VI's launch-day Twitch viewers for hours 2 to 24.
It anchors on the known hour-1 viewers and pools past launches whose unlock text mentions midnight local or rolling.
For each hour it fits an ordinary least-squares regression of log hour-h viewers on log hour-1 viewers across the pool, takes the fitted value at the case's hour-1 count as the median, and spreads the five quantiles by the population standard deviation of the regression residuals.
Hours with fewer than 3 usable pool members fall back to a median log-ratio, and hours with none fall back to the starter baseline formula.
The program is deterministic and uses only the standard library.
"""
import os
import json
import math
import glob

Z_LEVELS = [
    (0.05, -1.644854),
    (0.25, -0.674490),
    (0.50, 0.0),
    (0.75, 0.674490),
    (0.95, 1.644854)
]

def parse_hour(ts):
    if not ts or not isinstance(ts, str):
        return None
    try:
        if len(ts) >= 13 and ts[10] in ('T', ' '):
            return int(ts[11:13])
        parts = ts.replace('T', ' ').split()
        if len(parts) > 1:
            time_part = parts[1]
            hour_str = time_part.split(':')[0]
            return int(hour_str)
    except Exception:
        pass
    return None

def compute_forecast(case):
    # Anchor p
    p = 1.0
    try:
        known = case.get('known')
        if isinstance(known, list) and len(known) > 0:
            k0 = known[0]
            if isinstance(k0, dict):
                v = k0.get('viewers') or k0.get('viewers_count')
                if v is not None:
                    p = float(v)
            elif isinstance(k0, (int, float)):
                p = float(k0)
        elif isinstance(known, dict):
            v = known.get('viewers') or known.get('viewers_count')
            if v is not None:
                p = float(v)
        
        if p <= 0:
            history_hourly = case.get('history_hourly')
            if isinstance(history_hourly, list) and len(history_hourly) > 0:
                last_h = history_hourly[-1]
                if isinstance(last_h, dict):
                    v = last_h.get('viewers') or last_h.get('viewers_count')
                    if v is not None:
                        p = float(v)
    except Exception:
        p = 1.0

    if p <= 0 or math.isnan(p) or math.isinf(p):
        p = 1.0

    game_obj = case.get('game', {})
    game_name = ""
    if isinstance(game_obj, dict):
        game_name = str(game_obj.get('name', ''))
    elif isinstance(game_obj, str):
        game_name = game_obj

    comparables = case.get('comparables', [])
    if not isinstance(comparables, list):
        comparables = []

    def get_unlock_text(comp):
        for key in ['unlock', 'unlock_description', 'unlock_text', 'description']:
            val = comp.get(key)
            if val:
                return str(val)
        return ''

    def get_platforms(comp):
        pform = comp.get('platforms') or comp.get('platform') or []
        if isinstance(pform, list):
            return " ".join(str(x) for x in pform)
        return str(pform)

    filtered_pool = []
    for comp in comparables:
        if not isinstance(comp, dict):
            continue
        c_name = str(comp.get('name', ''))
        if c_name.lower() == game_name.lower():
            continue
        unlock_lower = get_unlock_text(comp).lower()

        has_unlock = ("midnight local" in unlock_lower) or ("rolling" in unlock_lower)

        if has_unlock:
            filtered_pool.append(comp)

    pool = filtered_pool
    if len(pool) < 6:
        pool = []
        for comp in comparables:
            if not isinstance(comp, dict):
                continue
            c_name = str(comp.get('name', ''))
            if c_name.lower() == game_name.lower():
                continue
            pool.append(comp)

    kept_members = []
    for comp in pool:
        day_start = str(comp.get('day_start', ''))
        if len(day_start) < 10:
            continue
        date_part = day_start[:10]

        hourly_pts = comp.get('hourly') or comp.get('hourly_points') or comp.get('points') or []
        if not isinstance(hourly_pts, list):
            continue

        hour_map = {}
        for pt in hourly_pts:
            if not isinstance(pt, dict):
                continue
            from_ts = str(pt.get('from') or pt.get('from_timestamp') or pt.get('timestamp') or '')
            if len(from_ts) < 10 or from_ts[:10] != date_part:
                continue
            viewers = pt.get('viewers') or pt.get('viewers_count')
            if viewers is None:
                continue
            try:
                v_float = float(viewers)
            except Exception:
                continue
            h_num = parse_hour(from_ts)
            if h_num is not None:
                hour_num = h_num + 1
                if 1 <= hour_num <= 24:
                    hour_map[hour_num] = max(1.0, v_float)

        if hour_map.get(1, 0) > 0 and len(hour_map) >= 20:
            kept_members.append(hour_map)

    hours_output = []

    for h in range(2, 25):
        q_list = []
        xs = []
        ys = []
        for hm in kept_members:
            if 1 in hm and h in hm:
                xs.append(math.log(hm[1]))
                ys.append(math.log(hm[h]))
        n_both = len(xs)

        if len(kept_members) == 0 or n_both == 0:
            for tau, z in Z_LEVELS:
                val = p * math.exp(z * 0.15 * math.sqrt(h - 1))
                val = max(1.0, min(1000000000.0, val))
                q_list.append(val)
        elif n_both < 3:
            log_ratios = [ys[i] - xs[i] for i in range(n_both)]
            sorted_lr = sorted(log_ratios)
            n_lr = len(sorted_lr)
            if n_lr % 2 == 1:
                m = sorted_lr[n_lr // 2]
            else:
                m = 0.5 * (sorted_lr[n_lr // 2 - 1] + sorted_lr[n_lr // 2])

            mean_sq_dev = sum((lr - m) ** 2 for lr in sorted_lr) / n_lr
            s = math.sqrt(mean_sq_dev)

            for tau, z in Z_LEVELS:
                val = p * math.exp(m + z * s)
                val = max(1.0, min(1000000000.0, val))
                q_list.append(val)
        else:
            mean_x = sum(xs) / n_both
            mean_y = sum(ys) / n_both
            s_xx = sum((x - mean_x) ** 2 for x in xs)
            if s_xx == 0 or abs(s_xx) < 1e-12:
                b = 0.0
                a = mean_y
            else:
                s_xy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n_both))
                b = s_xy / s_xx
                a = mean_y - b * mean_x

            median_forecast = math.exp(a + b * math.log(p))
            residuals = [ys[i] - (a + b * xs[i]) for i in range(n_both)]
            mean_sq_res = sum(e ** 2 for e in residuals) / n_both
            s = math.sqrt(mean_sq_res)

            for tau, z in Z_LEVELS:
                val = median_forecast * math.exp(z * s)
                val = max(1.0, min(1000000000.0, val))
                q_list.append(val)

        for i in range(1, len(q_list)):
            if q_list[i] < q_list[i - 1]:
                q_list[i] = q_list[i - 1]

        hours_output.append({
            "hour": h,
            "q": q_list
        })

    return {"hours": hours_output}

def fallback_forecast():
    hours_output = []
    p = 1.0
    for h in range(2, 25):
        q_list = []
        for tau, z in Z_LEVELS:
            val = p * math.exp(z * 0.15 * math.sqrt(h - 1))
            val = max(1.0, min(1000000000.0, val))
            q_list.append(val)
        for i in range(1, len(q_list)):
            if q_list[i] < q_list[i - 1]:
                q_list[i] = q_list[i - 1]
        hours_output.append({
            "hour": h,
            "q": q_list
        })
    return {"hours": hours_output}

def main():
    input_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    output_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')

    if not os.path.exists(input_dir):
        if os.path.exists('input'):
            input_dir = 'input'
        else:
            input_dir = '.'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    in_files = sorted(glob.glob(os.path.join(input_dir, '*.in')))
    if not in_files:
        in_files = sorted(glob.glob('*.in'))

    for filepath in in_files:
        filename = os.path.basename(filepath)
        case_id = os.path.splitext(filename)[0]
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                case_data = json.load(f)
            if isinstance(case_data, dict):
                if 'id' in case_data:
                    case_id = str(case_data['id'])
                result = compute_forecast(case_data)
            else:
                result = fallback_forecast()
        except Exception:
            result = fallback_forecast()

        out_filepath = os.path.join(output_dir, f"{case_id}.out")
        try:
            with open(out_filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, separators=(',', ':'))
        except Exception:
            out_filepath = f"{case_id}.out"
            with open(out_filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, separators=(',', ':'))

if __name__ == '__main__':
    main()
