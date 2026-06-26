# -*- coding: utf-8 -*-
"""
cycles.py — 时间周期择时模型核心
=================================================
把三种经典周期方法实现为可计算的函数，并叠加成"时间共振"评分：

1. 合成周期模型 (Composite Cycle)
   - 对（对数）价格做线性去趋势 -> FFT 频谱分解 -> 挑选主导周期分量
   - 用最小二乘把这些正弦波拟合回历史，再向前外推得到一条"合成曲线"
   - 合成曲线（剔除趋势部分）的未来波谷 = 投影出的时间拐点

2. 周期嵌套模型 (Hurst Cycle Nesting)
   - 用带通振荡器探测历史波谷
   - 估计某条"名义周期"的平均实测长度，从最近一个波谷向前投影
   - 同时投影一组 2:1 嵌套的名义周期（季/半年/年），体现 harmonicity + synchronicity

3. 斐波那契时间周期 (Fibonacci Time Cycles)
   - 从一个关键拐点(pivot)起，按斐波那契交易日间隔向前数，投影时间拐点

时间共振 (Time Confluence)
   - 把上述方法投影出的未来拐点日期聚类，落在同一窗口的方法越多，确信度越高

注意：这是技术/周期择时启发式工具，只负责"时间到位"与大盘方向判断，
不构成投资建议。基本面/板块 overlay 不在本模型范围内。
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# 斐波那契序列（交易日），跳过过小的项
FIB = [13, 21, 34, 55, 89, 144, 233, 377, 610, 987]


# ----------------------------------------------------------------------
# 基础工具
# ----------------------------------------------------------------------
def prep_series(close, use_log=True):
    """清洗价格序列，返回 (DatetimeIndex, ndarray)。yields/负值不取对数。"""
    s = pd.Series(close).dropna().astype(float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    vals = s.values
    if use_log and np.all(vals > 0):
        y = np.log(vals)
    else:
        y = vals.astype(float)
    return s.index, y


def linear_detrend(y):
    """去线性趋势，返回 (残差, 趋势多项式系数)。"""
    t = np.arange(len(y))
    coef = np.polyfit(t, y, 1)
    return y - np.polyval(coef, t), coef


def future_bdates(last_date, n):
    """从 last_date 之后生成 n 个工作日日期索引。"""
    if n <= 0:
        return pd.DatetimeIndex([])
    start = pd.Timestamp(last_date) + pd.Timedelta(days=1)
    return pd.bdate_range(start=start, periods=n)


def pos_to_date(pos, hist_index, fut_index):
    """把"从 0 起算的 bar 位置"映射成日期。pos 可越过历史末端进入投影区。"""
    N = len(hist_index)
    if pos < 0:
        pos = 0
    if pos < N:
        return pd.Timestamp(hist_index[int(round(pos))])
    fpos = int(round(pos)) - N
    if fpos < len(fut_index):
        return pd.Timestamp(fut_index[fpos])
    # 超出投影区则线性外推工作日
    extra = fpos - len(fut_index) + 1
    base = fut_index[-1] if len(fut_index) else hist_index[-1]
    return pd.Timestamp(base) + pd.tseries.offsets.BDay(extra)


# ----------------------------------------------------------------------
# 1. 合成周期模型
# ----------------------------------------------------------------------
def spectrum_periods(y_detr, n=5, min_p=20, max_p=None):
    """对去趋势序列做加窗 FFT，返回主导周期列表（按能量降序、去重）。"""
    N = len(y_detr)
    if max_p is None:
        max_p = max(min_p + 1, N // 2)
    w = np.hanning(N)
    Y = np.fft.rfft((y_detr - y_detr.mean()) * w)
    freqs = np.fft.rfftfreq(N, d=1.0)
    power = np.abs(Y)
    with np.errstate(divide="ignore"):
        periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
    band = (periods >= min_p) & (periods <= max_p)
    idx = np.where(band)[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(power[idx])[::-1]]
    chosen = []
    for i in order:
        if all(abs(periods[i] - periods[j]) > 0.12 * periods[j] for j in chosen):
            chosen.append(i)
        if len(chosen) >= n:
            break
    tot = power[band].sum() + 1e-12
    return [
        {"period": float(periods[i]), "power": float(power[i]),
         "weight": float(power[i] / tot)}
        for i in chosen
    ]


def fit_sines(y_detr, periods):
    """最小二乘拟合给定周期的正余弦分量，返回系数 [c0, a1,b1, a2,b2, ...]。"""
    N = len(y_detr)
    t = np.arange(N)
    cols = [np.ones(N)]
    for p in periods:
        w = 2 * np.pi / p
        cols += [np.cos(w * t), np.sin(w * t)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y_detr, rcond=None)
    return coef


def composite_curve(periods, coef, N, project):
    """重构并外推合成周期（纯周期分量，不含趋势）。返回 (t, comp)。"""
    t = np.arange(N + project)
    comp = np.full(len(t), coef[0], dtype=float)
    k = 1
    for p in periods:
        w = 2 * np.pi / p
        comp += coef[k] * np.cos(w * t) + coef[k + 1] * np.sin(w * t)
        k += 2
    return t, comp


# ----------------------------------------------------------------------
# 2. 周期嵌套模型 (Hurst)
# ----------------------------------------------------------------------
def cycle_troughs(y, period):
    """用带通振荡器探测周期波谷，返回 (波谷位置数组, 振荡器序列)。"""
    s = pd.Series(y)
    longw = int(max(10, round(2 * period)))
    shortw = int(max(3, round(period * 0.15)))
    osc = (s.rolling(shortw, min_periods=1, center=True).mean()
           - s.rolling(longw, min_periods=1, center=True).mean())
    dist = int(max(5, round(period * 0.55)))
    troughs, _ = find_peaks(-osc.values, distance=dist)
    return troughs, osc.values


def project_nominal(troughs, length, N, project, horizon):
    """从最近波谷按名义长度向前投影波谷位置（仅返回未来 horizon 内的）。"""
    if len(troughs) == 0:
        return []
    last = troughs[-1]
    out = []
    k = 1
    while True:
        pos = last + k * length
        if pos > N + project + 5:
            break
        if pos >= N - 5:  # 含刚发生与未来
            out.append(pos)
        k += 1
        if k > 12:
            break
    return out


# ----------------------------------------------------------------------
# 3. 斐波那契时间周期
# ----------------------------------------------------------------------
def auto_pivot(detr, recent=520):
    """
    自动挑选斐波那契关键拐点：在最近 ~recent 个交易日(默认约2年)内，
    取 prominence 最大的摆动极值（高点或低点皆可）。
    锚定近端拐点，未来的斐波时间点才会落在可观测的投影窗口内。
    """
    N = len(detr)
    lo = max(0, N - recent)
    seg = detr[lo:]
    prom = np.std(detr) * 0.4
    tr, tp = find_peaks(-seg, prominence=prom)
    pk, pp = find_peaks(seg, prominence=prom)
    cand = [(lo + t, p) for t, p in zip(tr, tp["prominences"])] + \
           [(lo + t, p) for t, p in zip(pk, pp["prominences"])]
    if not cand:
        # 退化：用全样本最低点
        return int(np.argmin(detr))
    return int(max(cand, key=lambda x: x[1])[0])


def fib_projection(pivot_pos, N, project, horizon_pos):
    """从 pivot 起按斐波那契交易日投影，返回 [(位置, fib数)]。"""
    out = []
    for f in FIB:
        pos = pivot_pos + f
        if pos > horizon_pos:
            break
        if pos >= N - 20:  # 近端 + 未来
            out.append((pos, f))
    return out


# ----------------------------------------------------------------------
# 时间共振聚类
# ----------------------------------------------------------------------
# 各方法在共振评分中的权重
METHOD_WEIGHT = {"合成周期": 1.0, "嵌套微周期": 1.0, "斐波那契": 1.0, "名义周期": 0.5}


def cluster_confluence(events, tol_days=4):
    """
    events: [{date, method, label}]  -> 聚类成共振窗口
    返回按分值降序的 [{date, score, methods:set, items:[...]}]
    """
    if not events:
        return []
    ev = sorted(events, key=lambda e: e["date"])
    clusters = []
    cur = [ev[0]]
    for e in ev[1:]:
        if (e["date"] - cur[-1]["date"]).days <= tol_days:
            cur.append(e)
        else:
            clusters.append(cur)
            cur = [e]
    clusters.append(cur)

    out = []
    for c in clusters:
        methods = {}
        for e in c:
            methods[e["method"]] = max(methods.get(e["method"], 0),
                                       METHOD_WEIGHT.get(e["method"], 0.5))
        score = sum(methods.values())
        center = pd.Timestamp(int(np.mean([x["date"].value for x in c])))
        out.append({
            "date": center.normalize(),
            "score": round(score, 2),
            "methods": set(methods.keys()),
            "n_methods": len(methods),
            "items": c,
        })
    out.sort(key=lambda x: (x["score"], -abs((x["date"] - pd.Timestamp.now()).days)),
             reverse=True)
    return out


def confidence_label(score):
    if score >= 2.5:
        return "高", "🟢"
    if score >= 1.5:
        return "中", "🟡"
    return "低", "⚪"


# ----------------------------------------------------------------------
# 顶层分析
# ----------------------------------------------------------------------
def analyze(close, name="", ticker="", use_log=True, project=200,
            n_harmonics=5, fib_pivot=None, nominal=(65, 130, 260),
            micro_band=(40, 150), tol_days=4):
    """
    对单个标的运行三套周期模型 + 共振评分，返回一个完整 dict 供 UI 使用。
    fib_pivot: None=自动探测；或 'YYYY-MM-DD' 手动指定关键拐点。
    nominal: 名义周期长度（交易日），默认 季/半年/年。
    """
    idx, y = prep_series(close, use_log)
    N = len(y)
    res = {"name": name, "ticker": ticker, "use_log": use_log, "N": N,
           "ok": N >= 120}
    if not res["ok"]:
        res["error"] = "历史数据不足（<120 根），无法可靠估计周期。"
        return res

    detr, lin = linear_detrend(y)
    fut_index = future_bdates(idx[-1], project)
    all_dates = idx.append(fut_index)
    horizon_pos = N + project - 1

    res["index"] = idx
    res["price"] = np.exp(y) if use_log else y
    res["last_date"] = pd.Timestamp(idx[-1])
    res["all_dates"] = all_dates

    # ---- 1. 合成周期 ----
    periods_info = spectrum_periods(detr, n=n_harmonics, min_p=20,
                                    max_p=min(500, N // 2))
    res["periods"] = periods_info
    composite_events = []
    if periods_info:
        periods = [p["period"] for p in periods_info]
        coef = fit_sines(detr, periods)
        t, comp = composite_curve(periods, coef, N, project)
        trend_full = np.polyval(lin, t)
        comp_full = comp + trend_full
        res["composite_t"] = t
        res["composite"] = comp                      # 纯周期分量（用于拐点）
        res["composite_overlay"] = np.exp(comp_full) if use_log else comp_full
        # 拐点：在纯周期分量上找极值
        c_tr, _ = find_peaks(-comp, distance=int(max(8, min(periods) * 0.5)))
        c_pk, _ = find_peaks(comp, distance=int(max(8, min(periods) * 0.5)))
        res["comp_troughs"] = c_tr
        res["comp_peaks"] = c_pk
        # 未来波谷事件
        for p in c_tr:
            if p >= N - 1:
                d = pos_to_date(p, idx, fut_index)
                composite_events.append({"date": d, "method": "合成周期",
                                         "label": "合成波谷"})
        # 当前相位
        slope = comp[N - 1] - comp[N - 2]
        res["phase_rising"] = bool(slope > 0)
        nxt_tr = [p for p in c_tr if p >= N - 1]
        nxt_pk = [p for p in c_pk if p >= N - 1]
        res["next_trough_date"] = pos_to_date(nxt_tr[0], idx, fut_index) if nxt_tr else None
        res["next_peak_date"] = pos_to_date(nxt_pk[0], idx, fut_index) if nxt_pk else None
        prev_tr = [p for p in c_tr if p <= N - 1]
        res["last_trough_date"] = pos_to_date(prev_tr[-1], idx, fut_index) if prev_tr else None
        res["bars_since_trough"] = int(N - 1 - prev_tr[-1]) if prev_tr else None
    else:
        res["phase_rising"] = None
        res["next_trough_date"] = res["next_peak_date"] = None

    # ---- 2. Hurst 嵌套 ----
    # 微周期：频谱中落在 micro_band 的最强周期；否则用 nominal[0]
    micro = None
    for p in periods_info:
        if micro_band[0] <= p["period"] <= micro_band[1]:
            micro = p["period"]
            break
    if micro is None:
        micro = nominal[0]
    res["micro_period"] = float(micro)

    m_tr, osc = cycle_troughs(y, micro)
    res["hurst_osc"] = osc
    res["hurst_troughs"] = m_tr
    if len(m_tr) >= 2:
        res["micro_obs"] = float(np.median(np.diff(m_tr)))
    else:
        res["micro_obs"] = float(micro)
    res["last_micro_trough_date"] = (pos_to_date(m_tr[-1], idx, fut_index)
                                     if len(m_tr) else None)

    hurst_events = []
    micro_proj = project_nominal(m_tr, res["micro_obs"], N, project, horizon_pos)
    res["micro_proj_dates"] = []
    for pos in micro_proj:
        d = pos_to_date(pos, idx, fut_index)
        res["micro_proj_dates"].append(d)
        if pos >= N - 3:
            hurst_events.append({"date": d, "method": "嵌套微周期",
                                 "label": f"微周期({res['micro_obs']:.0f}d)波谷"})

    # 名义嵌套（季/半年/年）
    res["nominal"] = []
    nominal_events = []
    for L in nominal:
        tr_L, _ = cycle_troughs(y, L)
        proj = project_nominal(tr_L, L, N, project, horizon_pos)
        dates = [pos_to_date(p, idx, fut_index) for p in proj]
        res["nominal"].append({"length": L, "dates": dates,
                               "last_trough": (pos_to_date(tr_L[-1], idx, fut_index)
                                               if len(tr_L) else None)})
        for p, d in zip(proj, dates):
            if p >= N - 3:
                nominal_events.append({"date": d, "method": "名义周期",
                                       "label": f"{L}d名义波谷"})

    # ---- 3. 斐波那契时间周期 ----
    if fib_pivot:
        try:
            pv = pd.Timestamp(fib_pivot)
            pivot_pos = int(idx.get_indexer([pv], method="nearest")[0])
        except Exception:
            pivot_pos = auto_pivot(detr)
    else:
        pivot_pos = auto_pivot(detr)
    res["fib_pivot_date"] = pos_to_date(pivot_pos, idx, fut_index)
    res["fib_pivot_pos"] = int(pivot_pos)
    fib_pts = fib_projection(pivot_pos, N, project, horizon_pos)
    res["fib"] = []
    fib_events = []
    for pos, f in fib_pts:
        d = pos_to_date(pos, idx, fut_index)
        res["fib"].append({"n": f, "date": d, "future": pos >= N - 1})
        if pos >= N - 3:
            fib_events.append({"date": d, "method": "斐波那契",
                               "label": f"Fib {f}"})

    # ---- 时间共振 ----
    today = pd.Timestamp.now().normalize()
    horizon_date = res["last_date"] + pd.tseries.offsets.BDay(project)
    all_events = composite_events + hurst_events + nominal_events + fib_events
    all_events = [e for e in all_events
                  if today - pd.Timedelta(days=10) <= e["date"] <= horizon_date]
    res["events"] = all_events
    res["confluence"] = cluster_confluence(all_events, tol_days=tol_days)

    # ---- 趋势 & 指引 ----
    price = res["price"]
    s_price = pd.Series(price, index=idx)
    ma = s_price.rolling(min(200, max(20, N // 3)), min_periods=10).mean()
    res["ma_long"] = ma
    res["above_ma"] = bool(price[-1] >= ma.values[-1]) if not np.isnan(ma.values[-1]) else None
    res["chg_1d"] = float(price[-1] / price[-2] - 1) if N >= 2 else 0.0
    res["chg_5d"] = float(price[-1] / price[-6] - 1) if N >= 6 else 0.0
    res["chg_20d"] = float(price[-1] / price[-21] - 1) if N >= 21 else 0.0

    res.update(_make_guidance(res))
    return res


def _bdays_between(d0, d1):
    """两个日期间的工作日数（带符号）。"""
    if d0 is None or d1 is None:
        return None
    sign = 1 if d1 >= d0 else -1
    n = np.busday_count(min(d0, d1).date(), max(d0, d1).date())
    return int(sign * n)


def _make_guidance(res):
    """根据周期相位 + 趋势 + 共振，生成相位标签、操作建议与中文指引。"""
    today = pd.Timestamp.now().normalize()
    rising = res.get("phase_rising")
    nt = res.get("next_trough_date")
    npk = res.get("next_peak_date")
    days_to_trough = _bdays_between(today, nt) if nt else None
    days_to_peak = _bdays_between(today, npk) if npk else None
    bars_since = res.get("bars_since_trough")
    micro = res.get("micro_obs", 65)
    above = res.get("above_ma")

    # 最近的共振窗口
    conf = res.get("confluence", [])
    best = conf[0] if conf else None
    best_days = _bdays_between(today, best["date"]) if best else None

    # 相位标签
    if rising is None:
        phase = "数据不足"
    elif rising:
        phase = "上行段 ▲"
    else:
        phase = "下行段 ▼"

    # 操作分类
    action, color = "中性 · 观望", "gray"
    reasons = []

    if days_to_trough is not None and 0 <= days_to_trough <= max(8, micro * 0.12):
        action, color = "抄底 / 分批建仓窗口", "green"
        reasons.append(f"临近合成周期低点（约{days_to_trough}个交易日后，"
                       f"{nt.date()}）")
    elif rising and bars_since is not None and bars_since <= micro * 0.4:
        action, color = "上行初段 · 持有 / 逢回吸纳", "green"
        reasons.append(f"周期波谷已现（{res.get('last_trough_date').date() if res.get('last_trough_date') else '近期'}），"
                       f"上行展开中")
    elif days_to_peak is not None and 0 <= days_to_peak <= max(8, micro * 0.12):
        action, color = "临近周期高点 · 减仓 / 兑现", "red"
        reasons.append(f"临近合成周期高点（约{days_to_peak}个交易日后，"
                       f"{npk.date()}）")
    elif rising is False:
        action, color = "下行段 · 观望 / 等待时间到位", "orange"
        if nt:
            reasons.append(f"下一周期低点投影在 {nt.date()}"
                           + (f"（约{days_to_trough}个交易日后）" if days_to_trough else ""))
    elif rising:
        action, color = "上行中段 · 持有", "green"

    # 趋势叠加
    if above is True:
        reasons.append("价在长期均线上方（多头结构）")
    elif above is False:
        reasons.append("价在长期均线下方（空头/筑底结构）")

    # 共振叠加
    if best and best_days is not None and -5 <= best_days <= 60:
        lvl, _ = confidence_label(best["score"])
        reasons.append(f"时间共振窗口 {best['date'].date()}"
                       f"（{best['n_methods']}法共振·确信度{lvl}）")
        res["near_confluence"] = best
    else:
        res["near_confluence"] = None

    text = "；".join(reasons) if reasons else "周期信号中性，等待时间窗口。"
    return {"phase": phase, "action": action, "action_color": color,
            "guidance": text,
            "days_to_trough": days_to_trough,
            "days_to_peak": days_to_peak}
