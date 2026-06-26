# -*- coding: utf-8 -*-
"""
analysis.py — 把周期模型结果 + 价格结构，合成一份"完整交易建议 + 分析总结"
=================================================
交易建议结构：方向偏向 / 时间窗口 / 触发条件 / 关键价位 / 参考入场 / 止损失效 /
目标 / 仓位提示 / 一段总结。可叠加宏观倾向(macro_tilt)。

所有价位都来自价格结构（近端摆动高低点、长期均线、ATR），是**模型推导的参考**，
不是精确点位，更不是投资建议。利率(美债收益率)为收益率口径，方向语义与价格相反，已注明。
"""
import numpy as np
import pandas as pd


def fmt_num(x, is_yield=False):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if is_yield:
        return f"{x:.3f}%"
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.1f}"
    if ax >= 10:
        return f"{x:.2f}"
    return f"{x:.3f}"


def month_seasonality(index, price):
    """该标的『当前日历月』的价格季节性：历史同月月度收益均值 + 上涨年数/胜率。"""
    try:
        s = pd.Series(price, index=pd.DatetimeIndex(index)).dropna()
        m = s.resample("ME").last()
        ret = (m.pct_change() * 100).dropna()
        cm = pd.Timestamp.now().month
        same = ret[ret.index.month == cm]
        if len(same) == 0:
            return None
        return {"month": cm, "avg": float(same.mean()), "n": int(len(same)),
                "pos": int((same > 0).sum()), "winrate": float((same > 0).mean() * 100)}
    except Exception:
        return None


def forward_view(r, name, is_yield=False, tol=6):
    """
    生成自然语言『后续走势观点』，模仿示例口吻：
      合成周期 → 反弹/见顶时间窗 + 操作难度；
      周期嵌套 → 当前第几轮微周期、平均多少周、本轮何时结束 + 与斐波/共振的时间对应。
    返回 dict：text(段落) / window(起止日期) / window_kind('low'|'high') / micro_end / round。
    """
    from cycles import confidence_label
    idx = r.get("index")
    last_date = r.get("last_date")
    micro = float(r.get("micro_obs", 65) or 65)
    micro_w = micro / 5.0
    rising = r.get("phase_rising")
    above = r.get("above_ma")
    nt = r.get("next_trough_date")
    npk = r.get("next_peak_date")
    pivot_date = r.get("fib_pivot_date")
    pivot_pos = r.get("fib_pivot_pos")

    # --- 合成周期：选最近的拐点作为头条时间窗 ---
    cand = []
    if nt is not None:
        cand.append((pd.Timestamp(nt), "low"))
    if npk is not None:
        cand.append((pd.Timestamp(npk), "high"))
    window = None
    window_kind = None
    comp_sentence = "合成周期暂无清晰的近端拐点投影。"
    if cand:
        turning, window_kind = min(cand, key=lambda x: x[0])
        w = max(3, int(round(micro * 0.06)))
        ws = (turning - pd.tseries.offsets.BDay(w)).normalize()
        we = (turning + pd.tseries.offsets.BDay(w)).normalize()
        window = (ws, we)
        if window_kind == "low":
            turn_word = "收益率阶段性低点" if is_yield else "阶段性低点、企稳反弹"
            if (above is False) or (rising is False):
                op = "属左侧布局，操作难度偏高，需轻仓试探"
            else:
                op = "为顺势回踩，可按计划分批"
        else:
            turn_word = "收益率阶段性高点" if is_yield else "周期高点、面临冲高回落"
            op = "宜高位兑现、避免追高"
        comp_sentence = (f"根据合成周期(composite cycle)模型显示，{name} 有望在 "
                         f"{ws.strftime('%m.%d')}–{we.strftime('%m.%d')} 附近迎来{turn_word}"
                         f"（{op}）。")

    # --- 周期嵌套：第几轮 + 本轮何时结束 ---
    troughs = list(r.get("hurst_troughs", []))
    if pivot_pos is not None:
        after = [t for t in troughs if t >= pivot_pos - 3]
    else:
        after = troughs
    round_k = max(1, len(after))
    micro_end = None
    for dd in r.get("micro_proj_dates", []):
        if pd.Timestamp(dd) >= pd.Timestamp(last_date):
            micro_end = pd.Timestamp(dd)
            break

    # Windows 上 strftime 不支持 %-m，手工拼接日期串
    if pivot_date is not None:
        pv_str = f"{pivot_date.year % 100}年{pivot_date.month}.{pivot_date.day}"
    else:
        pv_str = "关键拐点"

    if micro_end is not None:
        end_str = f"{micro_end.month}月{micro_end.day}日"
        nest_sentence = (f"根据周期嵌套模型显示，目前正经历自 {pv_str} 起算的第 {round_k} 轮、"
                         f"平均时间约 {micro_w:.0f} 周的微周期，本轮微周期大概率将于 "
                         f"{end_str} 前后结束")
        # 与斐波/共振的时间对应
        notes = []
        for x in r.get("fib", []):
            if x.get("future") and abs((pd.Timestamp(x["date"]) - micro_end).days) <= tol:
                notes.append(f"恰对应自 {pv_str} 起算的斐波那契时间点(F{x['n']})")
                break
        for c in r.get("confluence", []):
            if c.get("n_methods", 0) >= 2 and abs((pd.Timestamp(c["date"]) - micro_end).days) <= tol:
                lvl, _ = confidence_label(c["score"])
                notes.append(f"构成 {c['n_methods']} 法时间共振（确信度{lvl}）")
                break
        if notes:
            nest_sentence += f"，{end_str} 同样" + "、".join(notes) + "。"
        else:
            nest_sentence += "，为本轮微周期的时间到位点。"
    else:
        nest_sentence = (f"根据周期嵌套模型显示，目前正经历自 {pv_str} 起算的第 {round_k} 轮、"
                         f"平均约 {micro_w:.0f} 周的微周期，本轮仍在运行中。")

    text = comp_sentence + nest_sentence

    # ---- 本月价格季节性（顺/逆风）----
    seas = month_seasonality(idx, r.get("price"))
    if seas is not None:
        what = "收益率" if is_yield else "价格"
        if seas["avg"] > 0.05:
            tone = "季节顺风(历史本月多偏强)"
        elif seas["avg"] < -0.05:
            tone = "季节逆风(历史本月多偏弱)"
        else:
            tone = "季节中性"
        text += (f" 季节性：{seas['month']}月{what}历史同月均值 {seas['avg']:+.1f}%"
                 f"（近{seas['n']}年{seas['pos']}年上涨·胜率{seas['winrate']:.0f}%），{tone}。")

    return {"text": text, "window": window, "window_kind": window_kind,
            "micro_end": micro_end, "round": round_k, "micro_weeks": micro_w,
            "seasonality": seas}


def atr(df, n=14):
    """从 OHLC 计算 ATR(n)。df 需含 High/Low/Close。"""
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n, min_periods=2).mean().iloc[-1])


def reference_levels(r, is_yield=False, swing=30):
    """从近端价格结构提取支撑/阻力/中枢/ATR。"""
    df = r.get("ohlc")
    last = float(r["price"][-1])
    ma = None
    if "ma_long" in r and len(r["ma_long"]):
        v = r["ma_long"].values[-1]
        ma = None if np.isnan(v) else float(v)
    if df is not None and len(df) >= 5 and {"High", "Low", "Close"}.issubset(df.columns):
        support = float(df["Low"].tail(swing).min())
        resistance = float(df["High"].tail(swing).max())
        a = atr(df)
    else:  # 退化：只用收盘
        c = pd.Series(r["price"])
        support = float(c.tail(swing).min())
        resistance = float(c.tail(swing).max())
        a = float(c.pct_change().std() * last) if last else None
    return {"last": last, "support": support, "resistance": resistance,
            "ma": ma, "atr": a, "is_yield": is_yield}


def _pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def build_trade_plan(r, is_yield=False, macro_tilt_text=None):
    """返回一份结构化交易建议 dict。"""
    lv = reference_levels(r, is_yield)
    last, sup, res, ma, a = lv["last"], lv["support"], lv["resistance"], lv["ma"], lv["atr"]
    rising = r.get("phase_rising")
    above = r.get("above_ma")
    action = r.get("action", "")
    color = r.get("action_color", "gray")
    d2t = r.get("days_to_trough")
    d2p = r.get("days_to_peak")
    nt = r.get("next_trough_date")
    npk = r.get("next_peak_date")
    nc = r.get("near_confluence")

    def f(x):
        return fmt_num(x, is_yield)

    # ---- 方向偏向 ----
    if rising and above:
        bias, bcolor = "偏多 ▲（周期上行 + 站上均线）", "green"
    elif (rising is False) and (above is False):
        bias, bcolor = "偏空 ▼（周期下行 + 跌破均线）", "red"
    elif rising and above is False:
        bias, bcolor = "筑底偏多（周期转上行，但仍在均线下）", "orange"
    elif (rising is False) and above:
        bias, bcolor = "高位偏谨慎（周期下行，但仍在均线上）", "orange"
    else:
        bias, bcolor = "中性 · 区间", "gray"

    # ---- 时间窗口 ----
    hz = []
    if nt is not None:
        hz.append(f"下一周期低点(抄底时间锚)：**{nt.date()}**"
                  + (f"（约{d2t}个交易日后）" if d2t is not None else ""))
    if npk is not None:
        hz.append(f"下一周期高点(减仓时间锚)：**{npk.date()}**"
                  + (f"（约{d2p}个交易日后）" if d2p is not None else ""))
    if nc is not None:
        from cycles import confidence_label
        lvl, icon = confidence_label(nc["score"])
        hz.append(f"最近时间共振：**{nc['date'].date()}**（{nc['n_methods']}法·确信度{icon}{lvl}）")
    horizon = "；".join(hz) if hz else "暂无明确时间锚。"

    # ---- 触发 / 入场 / 止损 / 目标 ----
    sup_pct = _pct(sup, last)
    res_pct = _pct(res, last)
    buf = a if a else (0.02 * last)

    if color == "green" or (rising and d2t is not None and d2t <= 8):
        setup = (f"价格回踩 **{f(sup)}** 一带（近端支撑）、且合成周期翻为上行段时，"
                 f"为分批做多触发；临近上方『下一周期低点』时间锚同步确认。")
        entry = f"参考分批区：**{f(sup)} ~ {f(last)}**（现价附近至支撑）"
        stop = f"止损/失效：有效跌破 **{f(sup - buf)}**（支撑下方约1×ATR，{f(buf)}）→ 周期低点判断失效"
        target = f"目标：上看 **{f(res)}**（近端阻力，约{res_pct:+.1f}%）→ 站稳后看周期高点时间窗"
    elif color == "red" or (d2p is not None and d2p <= 8):
        setup = (f"临近周期高点时间窗，**不追高**；价格冲向 **{f(res)}** 上方滞涨即减仓/兑现。"
                 f"激进者可在阻力附近轻仓试空。")
        entry = f"减仓/试空参考：**{f(res)}** 附近"
        stop = f"（试空）止损：放量站上 **{f(res + buf)}**（阻力上方约1×ATR）"
        target = f"下看回补：**{f(sup)}**（近端支撑，约{sup_pct:+.1f}%）或下一周期低点"
    elif color == "orange" and rising is False:
        setup = (f"周期下行段，**以观望为主**，等待『下一周期低点』时间锚到位再找做多触发；"
                 f"不左侧重仓。")
        entry = f"等待区：接近 **{f(sup)}** 且时间锚到位再分批"
        stop = f"做多触发后止损：跌破 **{f(sup - buf)}**"
        target = f"反弹目标：**{f(ma) if ma else f(res)}**（趋势中枢/阻力）"
    else:
        setup = f"区间震荡，**低吸高抛**：靠近 **{f(sup)}** 偏多、靠近 **{f(res)}** 偏空。"
        entry = f"区间下沿 **{f(sup)}** 试多 / 上沿 **{f(res)}** 试空"
        stop = f"破区间：跌破 **{f(sup - buf)}** 或升破 **{f(res + buf)}** 离场"
        target = f"对侧边界：**{f(res)}** / **{f(sup)}**"

    # ---- 仓位 ----
    if nc is not None and nc["score"] >= 2.5:
        sizing = "确信度高（多法共振）→ 可按计划分 2-3 批建仓。"
    elif nc is not None and nc["score"] >= 1.5:
        sizing = "确信度中 → 先试仓 1/3，时间窗确认后再加。"
    else:
        sizing = "确信度偏低 → 轻仓试探为主，等共振增强再放量。"

    # ---- 关键价位表 ----
    levels = {
        "现价": f(last),
        "近端支撑": f(sup) + (f"（{sup_pct:+.1f}%）" if sup_pct is not None else ""),
        "近端阻力": f(res) + (f"（{res_pct:+.1f}%）" if res_pct is not None else ""),
        "趋势中枢(均线)": f(ma) if ma else "—",
        "ATR(14)": f(a) if a else "—",
    }

    # ---- 总结 ----
    yld_note = "（注：本标的为收益率口径，收益率↑=债价↓，做多/做空语义对应收益率方向）" if is_yield else ""
    parts = [
        f"**方向**：{bias}{yld_note}。",
        f"**节奏**：{action}；{r.get('guidance','')}。",
        f"**结构**：现价 {f(last)}，下方支撑 {f(sup)}、上方阻力 {f(res)}，"
        f"趋势中枢 {f(ma) if ma else '—'}。",
        f"**策略**：{setup}",
        f"**风控**：{stop}。仓位—{sizing}",
    ]
    if macro_tilt_text:
        parts.append(f"**宏观叠加**：{macro_tilt_text}")
    summary = " ".join(parts)

    return {"bias": bias, "bias_color": bcolor, "horizon": horizon,
            "setup": setup, "entry": entry, "stop": stop, "target": target,
            "sizing": sizing, "levels": levels, "summary": summary,
            "action": action, "action_color": color}
