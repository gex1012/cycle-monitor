# -*- coding: utf-8 -*-
"""
rotation.py — A股行业轮动 (RRG + 扩散指标) + 规则模拟盘
=================================================
复刻 西部证券『相对旋转图(RRG) + 扩散指标』行业轮动思路：
  - RRG：RS-Ratio(相对强弱趋势, 回看 220 日) + RS-Momentum(其变化率, 回看 60 日)，MA20 平滑，
    四象限：领先(+/+) / 改善(-/+) / 滞后(-/-) / 走弱(+/-)。
  - 策略：每月末取『扩散指标』最高的 6 个行业，再剔除处于三/四象限(滞后/走弱)者，等权持有。
  - 模拟盘：按上述规则从历史价格滚动重算净值(规则驱动，无需持久存储，随行情自动更新)。

诚实声明（务必对用户展示）：
  * 原文『扩散指标』是行业内部成分股广度(需成分股数据)。本处用 **ETF 自身多周期趋势强度做代理**，
    非原文口径，仅方向近似。
  * 行业池用流动性最好的 ~19 只行业 ETF 代理中信一级行业(非全 30 个)。
  * RS-Ratio/RS-Momentum 用标准 z-score 中心化到 100 的公开复现(JdK 原式为专有)。
  * 回测/模拟净值不含交易成本与冲击，属研究用途，不构成投资建议。
"""
import numpy as np
import pandas as pd
from data import fetch_close

BENCHMARK = ("510300.SS", "沪深300")
SECTORS = [
    ("512480.SS", "半导体"), ("512760.SS", "芯片"), ("512720.SS", "计算机"),
    ("515050.SS", "5G通信"), ("512980.SS", "传媒"), ("512660.SS", "军工"),
    ("512400.SS", "有色金属"), ("515220.SS", "煤炭"), ("512800.SS", "银行"),
    ("512880.SS", "证券"), ("512200.SS", "房地产"), ("516970.SS", "基建"),
    ("159611.SZ", "电力"), ("515790.SS", "光伏"), ("515030.SS", "新能源车"),
    ("512690.SS", "白酒"), ("159928.SZ", "消费"), ("512170.SS", "医疗"),
    ("512010.SS", "医药"),
]
NAME = {t: n for t, n in SECTORS}
NAME[BENCHMARK[0]] = BENCHMARK[1]

RS_W = 220      # RS-Ratio 回看
MOM_W = 60      # RS-Momentum 回看
SMOOTH = 20     # MA20 平滑
TOPN = 6        # 扩散指标 top N
Q_KEEP = ("领先", "改善")   # 保留象限（剔除 滞后/走弱）


# ----------------------------------------------------------------------
def fetch_universe(period="8y"):
    cols = {}
    b = fetch_close(BENCHMARK[0], period)
    if b is not None:
        cols[BENCHMARK[0]] = b
    for t, _ in SECTORS:
        s = fetch_close(t, period)
        if s is not None:
            cols[t] = s
    px = pd.DataFrame(cols).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    return px


def compute_rrg(px):
    """对每个行业算 RS-Ratio / RS-Momentum(中心化到 100)。返回 {ticker: DataFrame}。"""
    bench = px[BENCHMARK[0]]
    out = {}
    for t, _ in SECTORS:
        if t not in px:
            continue
        rs = (px[t] / bench).dropna()
        if len(rs) < RS_W + SMOOTH:
            continue
        ratio = 100 + (rs - rs.rolling(RS_W).mean()) / rs.rolling(RS_W).std()
        ratio = ratio.rolling(SMOOTH).mean()
        mom_raw = ratio - ratio.shift(MOM_W)
        mom = 100 + (mom_raw - mom_raw.rolling(RS_W).mean()) / mom_raw.rolling(RS_W).std()
        mom = mom.rolling(SMOOTH).mean()
        out[t] = pd.DataFrame({"rs_ratio": ratio, "rs_mom": mom}).dropna()
    return out


def quadrant(ratio, mom):
    if ratio is None or mom is None or np.isnan(ratio) or np.isnan(mom):
        return None
    if ratio >= 100 and mom >= 100:
        return "领先"
    if ratio < 100 and mom >= 100:
        return "改善"
    if ratio < 100 and mom < 100:
        return "滞后"
    return "走弱"


def compute_diffusion(px):
    """扩散指标代理：ETF 自身多周期趋势强度(0..1)。列=ticker，行=日期。"""
    out = {}
    for t, _ in SECTORS:
        if t not in px:
            continue
        s = px[t].dropna()
        if len(s) < 130:
            continue
        ma = {w: s.rolling(w).mean() for w in (5, 10, 20, 60, 120)}
        conds = pd.concat([
            s > ma[5], s > ma[10], s > ma[20], s > ma[60], s > ma[120],
            ma[5] > ma[20], ma[20] > ma[60], ma[60] > ma[120],
            s.pct_change(20) > 0, s.pct_change(60) > 0,
        ], axis=1)
        # 注意：s>NaN 在 pandas 里得 False（非 NaN），故 120 日均线未成形前置为 NaN，避免早期垃圾值
        out[t] = conds.mean(axis=1).where(ma[120].notna())
    return pd.DataFrame(out)


def select_holdings(rrg, diff, date):
    """在 date 依规则选行业：扩散 top6 → 剔除三四象限 → 返回 [ticker]。"""
    sub = diff.loc[:date]
    if len(sub) == 0:
        return []
    d = sub.iloc[-1].dropna().sort_values(ascending=False)
    top = list(d.index[:TOPN])
    keep = []
    for t in top:
        rr = rrg.get(t)
        if rr is None:
            continue
        rsub = rr.loc[:date]
        if len(rsub) == 0:
            continue
        q = quadrant(rsub["rs_ratio"].iloc[-1], rsub["rs_mom"].iloc[-1])
        if q in Q_KEEP:
            keep.append(t)
    return keep


def backtest(px, rrg, diff, market_filter=False):
    """
    按月末调仓滚动模拟净值。market_filter=True 时叠加市场择时：
    沪深300 跌破 200 日线则当日空仓(规避震荡市回撤，针对原文提示的弱点)。
    返回 navdf, holdings_log, current_holdings, start_date。
    """
    bt = BENCHMARK[0]
    rets = px.pct_change()
    dates = px.index
    month_end = set(pd.Series(dates, index=dates).resample("ME").last().dropna().tolist())

    # 起点：首个『>=8 个行业有有效 RRG 信号』的日期（RRG 需 ~300 日预热，是真正的约束）
    rrg_ratio = pd.DataFrame({t: rrg[t]["rs_ratio"] for t in rrg}).reindex(dates)
    rcount = rrg_ratio.notna().sum(axis=1)
    cand = rcount[rcount >= 8].index
    start = cand[0] if len(cand) else dates[0]

    bench_ma = px[bt].rolling(200).mean()
    risk_on = px[bt] > bench_ma

    nav, bnav = 1.0, 1.0
    cur = []
    rows, log = [], []
    for dt in dates:
        if dt < start:
            continue
        off = market_filter and not bool(risk_on.get(dt, True))
        if cur and not off:
            r = rets.loc[dt, cur].mean(skipna=True)
            r = 0.0 if pd.isna(r) else r
        else:
            r = 0.0
        br = rets.loc[dt, bt]
        br = 0.0 if pd.isna(br) else br
        nav *= (1 + r)
        bnav *= (1 + br)
        rows.append((dt, nav, bnav, 0 if off else len(cur)))
        if dt in month_end:   # 月末收盘后调仓，次日生效
            newh = select_holdings(rrg, diff, dt)
            if newh != cur:
                log.append({"date": dt, "holdings": [NAME[x] for x in newh]})
            cur = newh
    navdf = pd.DataFrame(rows, columns=["date", "strategy", "benchmark", "n_hold"]).set_index("date")
    return navdf, log, cur, start


def perf_metrics(navdf):
    strat = navdf["strategy"]
    bench = navdf["benchmark"]
    n = len(strat)
    yrs = max(n / 252.0, 1e-6)
    cagr = strat.iloc[-1] ** (1 / yrs) - 1
    bcagr = bench.iloc[-1] ** (1 / yrs) - 1
    dret = strat.pct_change().dropna()
    vol = dret.std() * np.sqrt(252)
    sharpe = (dret.mean() * 252) / (vol + 1e-9)
    mdd = (strat / strat.cummax() - 1).min()
    bmdd = (bench / bench.cummax() - 1).min()
    # 月度胜率
    sm = strat.resample("ME").last().pct_change().dropna()
    bm = bench.resample("ME").last().pct_change().dropna()
    common = sm.index.intersection(bm.index)
    win = float((sm.loc[common] > bm.loc[common]).mean() * 100) if len(common) else np.nan
    return {"年化收益": cagr, "基准年化": bcagr, "年化超额": cagr - bcagr,
            "总收益": strat.iloc[-1] - 1, "基准总收益": bench.iloc[-1] - 1,
            "年化波动": vol, "夏普": sharpe, "最大回撤": mdd, "基准最大回撤": bmdd,
            "月度胜率": win}


def rrg_points(rrg, tail=8, step=5):
    """取每个行业最新点 + 尾迹(近 tail 个、每 step 交易日采样一点)供 RRG 散点图。"""
    pts = {}
    for t, rr in rrg.items():
        if rr is None or len(rr) == 0:
            continue
        sampled = rr.iloc[::step]
        tailrows = sampled.tail(tail)
        if len(tailrows) == 0:
            continue
        pts[t] = {
            "name": NAME[t],
            "ratio": float(rr["rs_ratio"].iloc[-1]),
            "mom": float(rr["rs_mom"].iloc[-1]),
            "quadrant": quadrant(rr["rs_ratio"].iloc[-1], rr["rs_mom"].iloc[-1]),
            "tail_ratio": list(tailrows["rs_ratio"].values),
            "tail_mom": list(tailrows["rs_mom"].values),
        }
    return pts


def next_rebalance(px):
    """下一次调仓日（下一个月末）。"""
    last = px.index[-1]
    nxt = (last + pd.offsets.MonthEnd(1))
    if nxt <= last:
        nxt = last + pd.offsets.MonthEnd(1)
    return nxt


def snapshot(px, rrg, diff, cur):
    """当前持仓明细 + 扩散排名全表（供 UI）。"""
    last_date = px.index[-1]
    d_last = diff.loc[:last_date].iloc[-1] if len(diff) else pd.Series(dtype=float)
    rank = d_last.dropna().sort_values(ascending=False)
    rank_pos = {t: i + 1 for i, t in enumerate(rank.index)}

    cur_rows = []
    for t in cur:
        rr = rrg.get(t)
        ratio = float(rr["rs_ratio"].iloc[-1]) if rr is not None else np.nan
        mom = float(rr["rs_mom"].iloc[-1]) if rr is not None else np.nan
        r1m = float(px[t].iloc[-1] / px[t].iloc[-22] - 1) if len(px[t].dropna()) > 22 else np.nan
        cur_rows.append({"行业": NAME[t], "象限": quadrant(ratio, mom),
                         "扩散排名": rank_pos.get(t, None),
                         "扩散分": float(d_last.get(t, np.nan)),
                         "RS-Ratio": ratio, "RS-Mom": mom, "近1月": r1m})

    diff_rows = []
    for t in rank.index:
        rr = rrg.get(t)
        ratio = float(rr["rs_ratio"].iloc[-1]) if rr is not None else np.nan
        mom = float(rr["rs_mom"].iloc[-1]) if rr is not None else np.nan
        q = quadrant(ratio, mom)
        rank_i = rank_pos.get(t)
        selected = (rank_i is not None and rank_i <= TOPN and q in Q_KEEP)
        diff_rows.append({"行业": NAME[t], "扩散分": float(rank.get(t, np.nan)),
                          "象限": q, "RS-Ratio": ratio, "RS-Mom": mom,
                          "扩散Top6": "✅" if (rank_i and rank_i <= TOPN) else "",
                          "入选": "🟢持有" if selected else ""})
    return cur_rows, diff_rows, last_date


def run(period="8y", market_filter=False):
    """一次性算完，返回给 UI 的完整结果 dict（也供命令行测试）。"""
    px = fetch_universe(period)
    if px is None or px.empty or BENCHMARK[0] not in px:
        return None
    rrg = compute_rrg(px)
    diff = compute_diffusion(px)
    navdf, log, cur, start = backtest(px, rrg, diff, market_filter=market_filter)
    cur_rows, diff_rows, last_date = snapshot(px, rrg, diff, cur)
    return {"px": px, "navdf": navdf, "log": log, "current": cur, "start": start,
            "metrics": perf_metrics(navdf), "rrg_points": rrg_points(rrg),
            "current_rows": cur_rows, "diff_rows": diff_rows, "last_date": last_date,
            "next_rebalance": next_rebalance(px), "n_sectors": len(rrg)}
