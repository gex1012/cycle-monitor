# -*- coding: utf-8 -*-
"""
app.py — 时间周期择时监测平台 (Streamlit)
=================================================
运行：  python -m streamlit run app.py
首页：  全标的信号板 + 近期"时间共振"日历
详情：  单标的的合成周期/Hurst嵌套/斐波那契时间周期图与分析
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from data import fetch_ohlc
from cycles import analyze, confidence_label
import analysis
import macro

st.set_page_config(page_title="周期择时监测平台", layout="wide",
                   initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# 数据加载（带缓存）
# ----------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def load_one(ticker, use_log, period, project, n_harmonics, tol_days,
             nominal, micro_band, pivot):
    df = fetch_ohlc(ticker, period=period)
    if df is None or len(df) < 120:
        return None
    r = analyze(df["Close"].dropna(), ticker=ticker, use_log=use_log, project=project,
                n_harmonics=n_harmonics, fib_pivot=pivot,
                nominal=nominal, micro_band=micro_band, tol_days=tol_days)
    if r is not None and r.get("ok"):
        # 附带近端 OHLC 供"交易建议"计算 ATR/摆动高低点（限长以控缓存体积）
        r["ohlc"] = df.tail(220).copy()
    return r


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_reading(mid):
    cfg = macro.by_id(mid)
    return macro.latest(cfg)


@st.cache_data(ttl=3600, show_spinner=False)
def load_macro_raw(mid):
    return macro.fetch_raw(mid)


def all_macro_readings():
    """全部宏观最新读数（按定义顺序）。"""
    out = []
    for cfg in macro.MACRO:
        r = load_macro_reading(cfg["id"])
        if r is not None:
            out.append(r)
    return out


def fresh_macro_readings(readings):
    fr = [r for r in readings if macro.is_fresh(r)]
    # 最新参考月在前；同月按变动幅度排序
    fr.sort(key=lambda x: (x["ref_date"], abs(x["chg"])), reverse=True)
    return fr


def macro_tilt_text_for(key, fresh):
    """针对单个标的，把最新宏观批次的净倾向写成一句话。"""
    if not fresh:
        return None
    tilt = macro.instrument_tilt(fresh)
    score = tilt.get(key, 0)
    lbl, _ = macro.tilt_label(score)
    bucket = macro.BUCKET_CN.get(macro.INSTR_BUCKET.get(key, ""), "")
    return f"本月已公布数据对【{bucket}】净倾向 {lbl}（规则启发评分 {score:+d}，仅方向参考）。"


def load_all(params):
    out = {}
    prog = st.progress(0.0, text="抓取行情并计算周期…")
    n = len(config.INSTRUMENTS)
    for i, it in enumerate(config.INSTRUMENTS):
        r = load_one(it["ticker"], it["log"], params["period"], params["project"],
                     params["n_harmonics"], params["tol_days"],
                     tuple(params["nominal"]), tuple(params["micro_band"]),
                     it["pivot"])
        if r is not None:
            r["name"] = it["name"]
            r["group"] = it["group"]
            r["key"] = it["key"]
            r["note"] = it["note"]
        out[it["key"]] = r
        prog.progress((i + 1) / n, text=f"加载 {it['name']} ({i+1}/{n})")
    prog.empty()
    return out


# ----------------------------------------------------------------------
# 格式化工具
# ----------------------------------------------------------------------
def fmt_price(x, is_yield=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
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


def fmt_pct(x):
    if x is None or np.isnan(x):
        return "—"
    return f"{x*100:+.2f}%"


def fmt_date(d):
    return "—" if d is None else pd.Timestamp(d).strftime("%Y-%m-%d")


ACTION_BG = {"green": "#0b6b3a", "red": "#8b1a1a", "orange": "#7a4a00",
             "gray": "#3a3a3a"}


# ----------------------------------------------------------------------
# 侧边栏
# ----------------------------------------------------------------------
st.sidebar.title("⏱ 周期择时监测")
st.sidebar.caption(config.DATA_SOURCE)

view = st.sidebar.radio("视图", ["🏠 首页 · 信号板", "🔬 标的详情",
                                 "🌍 宏观数据 · 季节性", "📖 帮助 · 信号解读"],
                        index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("模型参数")
st.sidebar.caption("不确定怎么调？看『📖 帮助 · 信号解读』页。")
d = config.DEFAULTS
period = st.sidebar.selectbox(
    "历史长度", ["5y", "8y", "10y", "max"],
    index=["5y", "8y", "10y", "max"].index(d["period"]),
    help="拉多长的历史做频谱分解。越长→能识别的最长周期越长、估计越稳，但会掺入旧市场状态、"
         "对近期变化更钝；越短→更灵敏但噪声大、长周期测不准。常用 8y。")
project = st.sidebar.slider(
    "向前投影(交易日)", 90, 360, d["project"], 10,
    help="周期曲线向未来外推多少个交易日（200≈9-10个月）。只决定你能看多远的时间窗口，"
         "不改变模型本身。看得越远，远端越不可靠。")
n_harmonics = st.sidebar.slider(
    "合成周期分量数", 3, 8, d["n_harmonics"],
    help="合成周期叠加几个能量最大的主导周期。少(3)→只抓大节奏、平滑；"
         "多(7-8)→拟合更贴历史但易过拟合、混入小周期噪声。常用 4-5。")
tol_days = st.sidebar.slider(
    "共振聚类容差(日)", 2, 8, d["tol_days"],
    help="几条投影拐点相距多少天内算落在『同一时间窗口』。大→更容易凑成共振(确信度可能虚高)；"
         "小→更严格、共振更少更硬。常用 3-5。")

params = {"period": period, "project": project, "n_harmonics": n_harmonics,
          "tol_days": tol_days, "nominal": d["nominal"], "micro_band": d["micro_band"]}

if st.sidebar.button("🔄 刷新行情(清缓存)"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ 周期/技术择时启发式工具，仅供研究，不构成投资建议。"
                   "基本面与板块 overlay 不在模型内。")


# ----------------------------------------------------------------------
# 首页
# ----------------------------------------------------------------------
def render_home(DATA):
    st.title("🏠 时间周期择时 · 信号板")
    valid = [r for r in DATA.values() if r is not None]
    if valid:
        last = max(r["last_date"] for r in valid)
        st.caption(f"数据来源：{config.DATA_SOURCE}　|　最新数据：{fmt_date(last)}"
                   f"　|　标的数：{len(valid)}/{len(config.INSTRUMENTS)}")

    with st.expander("📖 模型说明：三法叠加 + 时间共振（点击展开）"):
        st.markdown("""
**本平台只做"时间到位"的择时与方向判断，不做估值/基本面。** 三种经典周期方法叠加：

1. **合成周期 (Composite Cycle)** — 对（对数）价格去趋势后做 FFT 频谱分解，挑出主导周期分量，
   最小二乘重构成一条合成曲线并向前外推；其**未来波谷**=投影的时间拐点。（源流：Dewey / Timing Solution）
2. **周期嵌套 (Hurst Nesting)** — 带通振荡器探测历史波谷，估计微周期（默认~13周）平均实测长度后向前投影；
   并叠加 2:1 嵌套的名义周期（季/半年/年），体现 harmonicity + synchronicity。（源流：J.M. Hurst 1970）
3. **斐波那契时间周期** — 从关键拐点(pivot)起按斐波那契交易日间隔向前数，投影下一时间拐点。

> **时间共振 (Time Confluence)**：不同方法投影的拐点落在同一窗口时，确信度抬升（🟢高 / 🟡中 / ⚪低）。
> 共振点越多、信号越强 —— 这是这类框架抬高确信度的标准逻辑。
        """)

    # ---- 最新宏观更新 + 对各标的影响 ----
    render_home_macro(DATA)

    # ---- 近期时间共振日历（全标的聚合）----
    st.subheader("📅 近期时间窗口（全标的共振聚合）")
    today = pd.Timestamp.now().normalize()
    rows = []
    for r in valid:
        for c in r["confluence"][:4]:
            dd = pd.Timestamp(c["date"])
            ndays = np.busday_count(today.date(), dd.date()) if dd >= today else -np.busday_count(dd.date(), today.date())
            if -10 <= ndays <= project:
                lvl, icon = confidence_label(c["score"])
                rows.append({"日期": dd, "标的": r["name"], "共振分": c["score"],
                             "方法数": c["n_methods"], "确信度": f"{icon}{lvl}",
                             "方法": "·".join(sorted(c["methods"])),
                             "距今(交易日)": int(ndays)})
    if rows:
        cal = pd.DataFrame(rows).sort_values(["日期", "共振分"], ascending=[True, False])
        cal_show = cal.copy()
        cal_show["日期"] = cal_show["日期"].dt.strftime("%Y-%m-%d")
        st.dataframe(cal_show, hide_index=True, width="stretch",
                     column_config={"共振分": st.column_config.NumberColumn(format="%.1f")})
    else:
        st.info("当前投影窗口内暂无显著共振点。")

    # ---- 信号板 ----
    st.subheader("📊 标的信号板")
    board = []
    for it in config.INSTRUMENTS:
        r = DATA.get(it["key"])
        if r is None:
            board.append({"标的": it["name"], "最新价": "—", "日涨跌": "—",
                          "5日": "—", "趋势": "—", "周期相位": "数据缺失",
                          "下一波谷": "—", "距波谷(d)": None, "确信度": "—",
                          "操作指引": "无数据", "_color": "gray"})
            continue
        is_y = (it["key"] == "us10y")
        nc = r.get("near_confluence")
        if nc is not None:
            lvl, icon = confidence_label(nc["score"])
            conf_s = f"{icon}{lvl}({fmt_date(nc['date'])})"
        else:
            conf_s = "—"
        trend = "▲多头" if r.get("above_ma") else ("▼空头" if r.get("above_ma") is False else "—")
        board.append({
            "标的": r["name"],
            "最新价": fmt_price(float(r["price"][-1]), is_y),
            "日涨跌": fmt_pct(r["chg_1d"]),
            "5日": fmt_pct(r["chg_5d"]),
            "趋势": trend,
            "周期相位": r["phase"],
            "下一波谷": fmt_date(r["next_trough_date"]),
            "距波谷(d)": r.get("days_to_trough"),
            "确信度": conf_s,
            "操作指引": r["action"],
            "_color": r["action_color"],
        })
    bdf = pd.DataFrame(board)
    colors = bdf.pop("_color")

    def _style(row):
        c = colors.loc[row.name]
        base = [""] * len(row)
        idx = list(bdf.columns).index("操作指引")
        base[idx] = f"background-color:{ACTION_BG.get(c,'#3a3a3a')};color:#fff;font-weight:600"
        return base

    sty = bdf.style.apply(_style, axis=1)
    st.dataframe(
        sty, hide_index=True, width="stretch",
        column_config={
            "日涨跌": st.column_config.Column(help="最新一根日线 vs 前一日收盘的涨跌幅"),
            "5日": st.column_config.Column(help="最近 5 个交易日累计涨跌幅"),
            "趋势": st.column_config.Column(help="价格相对长期均线(≈200日)的位置：▲多头=在均线上方，▼空头=下方。方向过滤器。"),
            "周期相位": st.column_config.Column(help="合成周期曲线当前的斜率方向：▲上行段 / ▼下行段。判断现在是涨势还是跌势的时段。"),
            "下一波谷": st.column_config.Column(help="合成周期投影出的下一个时间低点（拐点）日期。"),
            "距波谷(d)": st.column_config.NumberColumn(format="%d", help="距上面那个波谷还有几个交易日。数值小(≤约8)=临近抄底时间窗口；负数=波谷可能刚发生。"),
            "最近共振": st.column_config.Column(help="离今天最近的『时间共振』窗口及确信度(🟢高/🟡中/⚪低)。多法重合的日期更可信。"),
            "操作指引": st.column_config.Column(help="综合相位+趋势+共振给出的时间窗口提示。绿=偏多/建仓窗口，红=减仓，橙=下行观望，灰=中性。非投资建议。"),
        })

    st.caption("『距波谷(d)』为合成周期投影的下一波谷距今交易日数；负数表示波谷可能刚发生。"
               "绿=偏多/建仓窗口，红=减仓，橙=下行观望，灰=中性。")

    # ---- 各标的 · 后续走势观点（自然语言）----
    st.subheader("📝 各标的 · 后续走势观点")
    st.caption("由 合成周期(反弹/见顶时间窗) + 周期嵌套(当前第几轮微周期·何时结束) 自动生成，"
               "进入『🔬 标的详情』可在图上看到对应的预计窗口标注。非投资建议。")
    cols = st.columns(2)
    j = 0
    for it in config.INSTRUMENTS:
        r = DATA.get(it["key"])
        if r is None:
            continue
        v = analysis.forward_view(r, it["name"], is_yield=(it["key"] == "us10y"))
        wk = v.get("window_kind")
        dot = "🟢" if wk == "low" else ("🔴" if wk == "high" else "⚪")
        with cols[j % 2]:
            st.markdown(f"**{dot} {it['name']}**")
            st.caption(v["text"])
        j += 1


# ----------------------------------------------------------------------
# 详情图
# ----------------------------------------------------------------------
def build_chart(r, is_yield=False, view=None):
    fig = go.Figure()
    idx = r["index"]
    price = r["price"]
    all_dates = r["all_dates"]

    # 价格
    fig.add_trace(go.Scatter(x=idx, y=price, name="价格", mode="lines",
                             line=dict(color="#4da6ff", width=1.6)))
    # 长期均线
    if "ma_long" in r:
        fig.add_trace(go.Scatter(x=idx, y=r["ma_long"].values, name="长期均线",
                                 mode="lines", line=dict(color="#888", width=1, dash="dot")))
    # 合成周期曲线（含投影）
    if "composite_overlay" in r:
        ov = r["composite_overlay"]
        n = len(idx)
        fig.add_trace(go.Scatter(x=all_dates[:n], y=ov[:n], name="合成周期(拟合)",
                                 mode="lines", line=dict(color="#ffb84d", width=1.4)))
        fig.add_trace(go.Scatter(x=all_dates[n:], y=ov[n:], name="合成周期(投影)",
                                 mode="lines", line=dict(color="#ff7043", width=2, dash="dash")))
        # 波谷/波峰标记
        tr = r.get("comp_troughs", [])
        pk = r.get("comp_peaks", [])
        tr = [p for p in tr if p < len(all_dates)]
        pk = [p for p in pk if p < len(all_dates)]
        fig.add_trace(go.Scatter(x=[all_dates[p] for p in tr], y=[ov[p] for p in tr],
                                 name="周期波谷", mode="markers",
                                 marker=dict(color="#2ecc71", size=8, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=[all_dates[p] for p in pk], y=[ov[p] for p in pk],
                                 name="周期波峰", mode="markers",
                                 marker=dict(color="#e74c3c", size=8, symbol="triangle-down")))

    # 斐波那契 pivot
    if r.get("fib_pivot_date") is not None:
        fig.add_vline(x=r["fib_pivot_date"], line=dict(color="#9b59b6", width=1.5),
                      annotation_text="Fib锚点", annotation_position="top")
    # 斐波未来时间线
    for f in [x for x in r.get("fib", []) if x["future"]][:5]:
        fig.add_vline(x=f["date"], line=dict(color="#9b59b6", width=1, dash="dot"),
                      annotation_text=f"F{f['n']}", annotation_position="top")
    # 微周期投影
    for dproj in r.get("micro_proj_dates", [])[:5]:
        if dproj > r["last_date"]:
            fig.add_vline(x=dproj, line=dict(color="#1abc9c", width=1, dash="dashdot"))

    # 时间共振窗口（阴影）
    for c in r.get("confluence", [])[:5]:
        cd = pd.Timestamp(c["date"])
        op = min(0.28, 0.08 + 0.08 * c["score"])
        col = "#2ecc71" if c["score"] >= 2.5 else ("#f1c40f" if c["score"] >= 1.5 else "#7f8c8d")
        fig.add_vrect(x0=cd - pd.Timedelta(days=tol_days), x1=cd + pd.Timedelta(days=tol_days),
                      fillcolor=col, opacity=op, line_width=0)

    # ---- 后续走势观点：预计反弹/见顶窗口（带标注）----
    if view and view.get("window"):
        ws, we = view["window"]
        is_low = view.get("window_kind") == "low"
        label = "🟢 预计反弹/低点窗口" if is_low else "🔴 预计见顶窗口"
        fig.add_vrect(x0=ws, x1=we,
                      fillcolor="#2ecc71" if is_low else "#e74c3c", opacity=0.16,
                      line_width=1.2, line_dash="dot",
                      line_color="#2ecc71" if is_low else "#e74c3c",
                      annotation_text=label, annotation_position="top left",
                      annotation=dict(font=dict(size=12)))
    if view and view.get("micro_end") is not None:
        fig.add_vline(x=view["micro_end"], line=dict(color="#f39c12", width=1.6, dash="dash"),
                      annotation_text=f"微周期末端·第{view.get('round','?')}轮",
                      annotation_position="bottom left")

    fig.add_vline(x=r["last_date"], line=dict(color="#ffffff", width=1),
                  annotation_text="今日", annotation_position="bottom right")

    fig.update_layout(template="plotly_dark", height=560, hovermode="x unified",
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    if not is_yield:
        fig.update_yaxes(type="log")
    return fig


def render_detail(DATA):
    names = [it["name"] for it in config.INSTRUMENTS]
    keys = [it["key"] for it in config.INSTRUMENTS]
    sel_name = st.selectbox("选择标的", names, index=0)
    key = keys[names.index(sel_name)]
    r = DATA.get(key)
    it = config.by_key(key)
    if r is None:
        st.error("该标的数据加载失败，稍后点击侧栏『刷新行情』重试。")
        return
    is_y = (key == "us10y")

    st.title(f"🔬 {r['name']}")
    st.caption(f"{it['ticker']}　|　{it['note']}　|　截至 {fmt_date(r['last_date'])}"
               f"（共 {r['N']} 根日线）")

    # 指标行
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新价", fmt_price(float(r["price"][-1]), is_y), fmt_pct(r["chg_1d"]))
    c2.metric("5日", fmt_pct(r["chg_5d"]))
    c3.metric("20日", fmt_pct(r["chg_20d"]))
    c4.metric("周期相位", r["phase"])
    nc = r.get("near_confluence")
    if nc is not None:
        lvl, icon = confidence_label(nc["score"])
        c5.metric("最近共振", fmt_date(nc["date"]), f"{icon}{lvl} · {nc['n_methods']}法")
    else:
        c5.metric("最近共振", "—")

    # 指引
    color = {"green": "🟢", "red": "🔴", "orange": "🟠", "gray": "⚪"}.get(r["action_color"], "⚪")
    st.markdown(f"### {color} 操作指引：**{r['action']}**")
    st.info(r["guidance"])

    # 后续走势观点（合成周期 + 周期嵌套自然语言）
    view = analysis.forward_view(r, r["name"], is_yield=is_y)
    st.markdown("### 📌 后续走势观点")
    st.warning(view["text"])

    # 图（带『预计反弹/见顶窗口』标注）
    st.plotly_chart(build_chart(r, is_y, view), width="stretch")

    # ---- 完整交易建议（基于上图周期结构 + 价格结构 + 宏观叠加）----
    st.markdown("## 📋 完整交易建议")
    st.caption("基于上图：周期相位/时间窗口 + 近端摆动支撑阻力/ATR + 本月宏观倾向。"
               "价位为模型推导参考，非精确点位，更非投资建议。")
    fresh = fresh_macro_readings(all_macro_readings())
    tilt_text = macro_tilt_text_for(key, fresh)
    plan = analysis.build_trade_plan(r, is_yield=is_y, macro_tilt_text=tilt_text)

    pcolor = {"green": "🟢", "red": "🔴", "orange": "🟠", "gray": "⚪"}.get(plan["bias_color"], "⚪")
    pc1, pc2 = st.columns([1, 1])
    with pc1:
        st.markdown(f"**{pcolor} 方向偏向**：{plan['bias']}")
        st.markdown(f"**🕒 时间窗口**：{plan['horizon']}")
        st.markdown(f"**🎯 触发/入场**：{plan['setup']}")
        st.markdown(f"**🛑 止损/失效**：{plan['stop']}")
        st.markdown(f"**📈 目标**：{plan['target']}")
        st.markdown(f"**⚖️ 仓位**：{plan['sizing']}")
    with pc2:
        st.markdown("**关键价位**")
        lv = plan["levels"]
        st.dataframe(pd.DataFrame({"项目": list(lv.keys()), "数值": list(lv.values())}),
                     hide_index=True, width="stretch")
        if tilt_text:
            st.markdown("**宏观叠加**")
            st.info(tilt_text)

    st.markdown("### 🧠 分析总结")
    st.success(plan["summary"])

    # 三法明细
    st.markdown("## 🔧 三法模型明细")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1️⃣ 合成周期 · 主导分量")
        if r.get("periods"):
            pdf = pd.DataFrame([{"周期(交易日)": round(p["period"], 1),
                                 "周期(周)": round(p["period"] / 5, 1),
                                 "能量占比": f"{p['weight']*100:.0f}%"} for p in r["periods"]])
            st.dataframe(pdf, hide_index=True, width="stretch")
        st.caption(f"下一合成波谷：**{fmt_date(r.get('next_trough_date'))}**　|　"
                   f"下一合成波峰：**{fmt_date(r.get('next_peak_date'))}**")

        st.markdown("#### 2️⃣ Hurst 周期嵌套")
        st.caption(f"微周期实测长度：**{r.get('micro_obs',0):.0f}** 交易日"
                   f"（≈{r.get('micro_obs',0)/5:.1f} 周）　最近微周期波谷：{fmt_date(r.get('last_micro_trough_date'))}")
        nest_rows = [{"名义周期": "微周期", "长度(交易日)": round(r.get("micro_obs", 0)),
                      "投影波谷": "、".join(fmt_date(x) for x in r.get("micro_proj_dates", [])[:3]) or "—"}]
        for nm in r.get("nominal", []):
            nest_rows.append({"名义周期": f"{nm['length']}d(≈{nm['length']//5}周)",
                              "长度(交易日)": nm["length"],
                              "投影波谷": "、".join(fmt_date(x) for x in nm["dates"][:3]) or "—"})
        st.dataframe(pd.DataFrame(nest_rows), hide_index=True, width="stretch")

    with col2:
        st.markdown("#### 3️⃣ 斐波那契时间周期")
        st.caption(f"关键拐点(pivot)：**{fmt_date(r.get('fib_pivot_date'))}**"
                   + ("（手动锚定）" if it["pivot"] else "（自动探测近端摆动极值）"))
        if r.get("fib"):
            fdf = pd.DataFrame([{"Fib数": x["n"], "投影日期": fmt_date(x["date"]),
                                 "状态": "未来" if x["future"] else "已过"} for x in r["fib"]])
            st.dataframe(fdf, hide_index=True, width="stretch")

        st.markdown("#### ⭐ 时间共振窗口")
        if r.get("confluence"):
            crows = []
            for c in r["confluence"][:8]:
                lvl, icon = confidence_label(c["score"])
                crows.append({"窗口日期": fmt_date(c["date"]), "共振分": c["score"],
                              "方法数": c["n_methods"], "确信度": f"{icon}{lvl}",
                              "方法": "·".join(sorted(c["methods"]))})
            st.dataframe(pd.DataFrame(crows), hide_index=True, width="stretch")
        else:
            st.info("投影窗口内暂无共振点。")


# ----------------------------------------------------------------------
# 帮助 / 信号解读
# ----------------------------------------------------------------------
def render_help(params):
    st.title("📖 帮助 · 信号怎么读、参数怎么调")
    st.caption("一句话：这套模型赌的是『时间到位』而不是『价格到位』——到点了就容易出拐点，"
               "而拐点信号越多方法重合越可信。")

    tabs = st.tabs(["① 信号板逐列", "② 详情图例", "③ 三法在说什么",
                    "④ 时间共振·确信度", "⑤ 参数怎么调", "⑥ 案例：7.10 怎么数出来的"])

    # ---------- ① 信号板逐列 ----------
    with tabs[0]:
        st.markdown("""
#### 首页信号板，每一列怎么读

| 列 | 含义 | 怎么用 |
|---|---|---|
| **最新价 / 日涨跌 / 5日** | 最新收盘与近期涨跌 | 现状参考，不是周期信号本身 |
| **趋势** | 价 vs 长期均线(≈200日)：▲多头在上 / ▼空头在下 | **方向过滤器**。周期给"何时"，趋势给"顺不顺势"。下行段+空头=更干净的下跌；上行段+多头=更顺的上涨 |
| **周期相位** | 合成周期曲线当前斜率：▲上行段 / ▼下行段 | 现在处在涨势时段还是跌势时段。下行段里别急着满仓，等波谷 |
| **下一波谷** | 合成周期投影的下一个**时间低点** | 计划"第一轮抄底"的时间锚 |
| **距波谷(d)** | 离那个波谷还有几个交易日 | **≤约8 = 临近抄底时间窗**；负数 = 波谷可能刚发生（看是否转上行段确认） |
| **最近共振** | 离今天最近的共振窗口 + 确信度🟢🟡⚪ | 多法重合的日期，比单一方法更值得盯 |
| **操作指引** | 相位+趋势+共振综合出的时间提示 | 见下方配色 |

**操作指引配色**：🟢绿=偏多/建仓窗口 · 🔴红=临近高点/减仓 · 🟠橙=下行段观望等低点 · ⚪灰=中性。

> 它给的是**时间窗口与方向倾向**，不是买卖点位，更不是投资建议。具体进出、仓位、止损要你自己叠加风控。
        """)

    # ---------- ② 详情图例 ----------
    with tabs[1]:
        st.markdown("""
#### 标的详情页，图上每个元素

| 图元 | 是什么 |
|---|---|
| 🔵 **蓝色实线** | 实际价格（对数坐标，利率类除外） |
| ⚪ **灰色点线** | 长期均线（趋势基准） |
| 🟠 **橙色实线** | 合成周期**拟合**段（落在历史上，看贴合度） |
| 🔴 **红色虚线** | 合成周期**投影**段（未来外推，越远越虚） |
| 🔺 **绿色上三角** | 周期**波谷**（时间低点 → 潜在做多/抄底点） |
| 🔻 **红色下三角** | 周期**波峰**（时间高点 → 潜在减仓点） |
| 🟣 **紫色竖线(Fib锚点)** | 斐波那契起算的关键拐点 pivot |
| 🟣 **紫色竖点线 F-xx** | 从 pivot 数出的斐波那契时间点 |
| 🟢 **青色点划竖线** | Hurst 微周期投影的未来波谷 |
| ▮ **彩色阴影竖带** | **时间共振窗口**：绿=强(🟢) / 黄=中(🟡) / 灰=弱(⚪)，越宽越浓越值得关注 |
| ⬜ **白色竖线** | 今天 |

**最该盯的画面**：几条不同颜色的竖线/三角，挤在同一条阴影带里——那就是多法共振的高确信时间窗。
        """)

    # ---------- ③ 三法 ----------
    with tabs[2]:
        st.markdown("""
#### 三种方法各自在说什么

**1. 合成周期 (Composite Cycle)** — 频谱分解+重构
对数价格去趋势后做 FFT，找出能量最大的几个周期分量，最小二乘重构成一条合成正弦曲线再外推。
*详情页「主导分量」表*里，"周期(周)"列就是这些主导节奏（如 ~40周年度级、~13周季度级）。
合成曲线的未来波谷 = 投影的时间拐点。源流：Dewey / Timing Solution。

**2. 周期嵌套 (Hurst Nesting)** — 短周期套在长周期里
- *和谐性 harmonicity*：相邻周期长度多呈约 2:1（本平台名义周期取 季65d / 半年130d / 年260d）。
- *同步性 synchronicity*：不同长度的周期倾向**同时见底** → 这正是抬高确信度的物理直觉。
- *名义周期表*：用带通振荡器测出"微周期"的平均实测长度，从最近波谷往前投影。
你说的"均长约13周的微周期"就是这里的微周期。源流：J.M. Hurst (1970)。

**3. 斐波那契时间周期** — 从关键拐点按斐波那契**交易日**(13,21,34,55,89,144,233,377,610…)往前数，
每个落点是一个候选时间拐点。pivot 默认自动取近~2年最显著摆动极值；科创50手动锚定 24/9/24。

> 三法相互独立（一个看频谱、一个看实测周期间距、一个看斐波那契计数），所以它们**指向同一天**才有意义。
        """)

    # ---------- ④ 共振 ----------
    with tabs[3]:
        st.markdown("""
#### 时间共振 (Time Confluence) 与确信度

把三法投影出的所有未来拐点日期，按**容差**（侧栏可调，默认±4天）聚成簇。一个簇里出现的方法越多、越独立，分值越高：

| 方法 | 计分权重 |
|---|---|
| 合成周期波谷 | 1.0 |
| 嵌套微周期波谷 | 1.0 |
| 斐波那契时间点 | 1.0 |
| 名义周期(季/半年/年)波谷 | 0.5 |

**确信度分级**：🟢 高 (分值≥2.5) · 🟡 中 (≥1.5) · ⚪ 低 (<1.5)。

> 逻辑就是你说的：**共振点越多、信号越强**。注意这是"时间上的巧合度"，不是胜率保证——
> 它只提高"这个时间窗口容易出拐点"的把握，方向还要结合趋势/相位判断。
        """)

    # ---------- ⑤ 参数 ----------
    with tabs[4]:
        st.markdown("""
#### 侧栏四个参数怎么调

| 参数 | 作用 | 调大 ↑ | 调小 ↓ | 建议 |
|---|---|---|---|---|
| **历史长度** | 喂给频谱的历史窗 | 能识别更长的周期、估计更稳 | 更灵敏、但长周期测不准、噪声大 | `8y`；想抓更长的多年周期用 `10y/max` |
| **向前投影** | 曲线外推多远 | 看得更远 | 只看近端 | `200`(≈9-10月)；只盯短期可降到 120 |
| **合成周期分量数** | 叠几个主导周期 | 更贴历史、但易过拟合/混噪 | 更平滑、只抓大节奏 | `4-5` |
| **共振聚类容差** | 多近算"同一窗口" | 更易凑成共振(确信度可能虚高) | 更严格、共振更硬 | `3-5` 天 |

**调参小法则**
- 想要**更硬的共振信号** → 容差调小(2-3) + 分量数适中(4)。共振少但每个更可信。
- 想**多看候选窗口** → 容差调大(6-8) + 分量数(5-6)。会冒出更多🟡中等窗口供筛。
- 某标的合成曲线和价格**明显不贴** → 多半是历史太短或太长：先试把"历史长度"在 5y/8y/10y 间切换。
- 改任何参数会**自动重算**；换行情数据要点侧栏 🔄 刷新。
        """)

    # ---------- ⑥ 7.10 案例 ----------
    with tabs[5]:
        st.markdown("""
#### 案例：原文里的"7.10附近"是怎么数出来的

> 背景：原文写于 2025 年年中，把 **7.10** 作为"较高确信度"的第一轮抄底时间窗。它**不是**靠单一方法，
> 而是两条**相互独立**的时间方法同时指向同一天 —— 这就是"时间共振"。

**两条独立路径（都从 24/9/24 这个 A 股大底起算）**

1. **微周期末端（Hurst 同步性）**：从 9.24 大底起，按约 **13 周（≈一个季度）的名义微周期**连续数波谷，
   一轮接一轮……数到**第 7 轮**的末端，落在 ≈ **7.10**。依据是 synchronicity：短周期倾向在节点处同步见底。

2. **斐波那契时间底**：同样从 9.24 起，按**斐波那契时间间隔**向前数，其中一个斐波点也落在 **7.10 附近**。

**关键不在某一条多准，而在两条独立的线"撞"在同一天** → 时间共振 → 确信度被抬高。
共振的方法越多越靠近，这一天出拐点的把握越大。这正是本平台 `最近共振 / 确信度` 那一列的逻辑。

---

**诚实声明**：原文没有公开它精确的周期拟合参数，所以上面讲的是**方法论**，不是逐位复刻；
本平台用公开、固定的规则自动复算，得到的日期可能与原文差几天（节假日折算、锚点选择、周期长度差异都会影响）。
而且 7.10 指的是 **2025-07-10，现已成为历史** —— 你可以在『科创50』详情页回看当时那个窗口前后是否真的出现了低点。

下面用**本平台的同款机器**（科创50，锚点锁定 9.24）实时算一遍给你看：
        """)
        if st.button("▶ 用『科创50』(Fib锚点=2024-09-24) 实时复算", type="primary"):
            it = config.by_key("star50")
            with st.spinner("抓取科创50并按 9.24 锚点复算…"):
                r = load_one(it["ticker"], it["log"], params["period"], params["project"],
                             params["n_harmonics"], params["tol_days"],
                             tuple(params["nominal"]), tuple(params["micro_band"]), it["pivot"])
            if r is None:
                st.error("科创50 数据加载失败，请稍后在侧栏点 🔄 刷新后重试。")
            else:
                st.success(f"Fib 关键拐点(pivot) = {fmt_date(r['fib_pivot_date'])}　|　"
                           f"实测微周期长度 ≈ {r.get('micro_obs',0):.0f} 交易日"
                           f"（≈{r.get('micro_obs',0)/5:.1f} 周）")
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**① 从 9.24 数出来的斐波那契时间点**")
                    if r.get("fib"):
                        st.dataframe(
                            pd.DataFrame([{"Fib(交易日)": x["n"], "落点日期": fmt_date(x["date"]),
                                           "状态": "未来" if x["future"] else "已过"} for x in r["fib"]]),
                            hide_index=True, width="stretch")
                    st.markdown("**② Hurst 微周期投影的未来波谷**")
                    md = r.get("micro_proj_dates", [])
                    st.write("、".join(fmt_date(x) for x in md[:5]) if md else "—")
                with cc2:
                    st.markdown("**③ 三法叠加后的『时间共振』窗口**")
                    if r.get("confluence"):
                        crows = []
                        for c in r["confluence"][:6]:
                            lvl, icon = confidence_label(c["score"])
                            crows.append({"窗口日期": fmt_date(c["date"]), "共振分": c["score"],
                                          "方法数": c["n_methods"], "确信度": f"{icon}{lvl}",
                                          "方法": "·".join(sorted(c["methods"]))})
                        st.dataframe(pd.DataFrame(crows), hide_index=True, width="stretch")
                    st.caption("把②③和『科创50』详情页的阴影窗口对照看——这就是当年推出 7.10 用的同一套机器，"
                               "只是现在锚点不变、时间推进到了今天。")


# ----------------------------------------------------------------------
# 宏观：首页面板 + 独立页
# ----------------------------------------------------------------------
def render_home_macro(DATA):
    st.subheader("📡 最新宏观更新 · 重点数据优先")
    readings = all_macro_readings()
    if not readings:
        st.info("宏观数据暂不可用（FRED 抓取失败），稍后点侧栏 🔄 刷新重试。")
        return
    fresh = fresh_macro_readings(readings)
    if not fresh:
        st.caption("近期暂无新公布的重要宏观数据。完整数据见『🌍 宏观数据』页。")
        return

    today = pd.Timestamp.now().normalize()
    def _ago(r):
        return (today - macro.est_release_date(r)).days
    # 近 3 日内"刚公布"的（排除日频/准实时序列，去重保序）
    just = [r for r in sorted(fresh, key=_ago)
            if -1 <= _ago(r) <= 3
            and not (r["cfg"].get("monthly_last") or r["cfg"]["freq"] == "D")]
    just_names = list(dict.fromkeys(r["cfg"]["name"] for r in just))
    line = f"🔥 **刚公布（近几日）**：{('、'.join(just_names[:6]))}。" if just_names else ""
    st.caption(line + f"本批次共 {len(fresh)} 项；卡片按『最新公布在前』排列，⭐=市场最关注（CPI/PCE/非农等）。")

    # 最新公布在前；同日按重点(tier1)优先、变动大者在前
    cards = sorted(fresh, key=lambda x: (-macro.est_release_date(x).value,
                                         macro.tier(x["cfg"]), -abs(x["chg"])))
    cols = st.columns(3)
    arr = {"up": "▲", "down": "▼", "flat": "▬"}
    for i, rr in enumerate(cards[:9]):
        cfg = rr["cfg"]
        star = "⭐ " if macro.tier(cfg) == 1 else ""
        with cols[i % 3]:
            st.metric(f"{star}{cfg['name']}",
                      macro.fmt_val(cfg, rr["value"]),
                      f"{arr[rr['dir']]} 前值 {macro.fmt_val(cfg, rr['prior'])}")
            st.caption(f"🗓 {macro.release_hint(rr)}　·　{macro.impact_comment(cfg, rr)}")

    st.markdown("**最新宏观批次 → 各标的方向倾向**（规则启发，非投资建议）")
    tilt = macro.instrument_tilt(fresh)
    rows = []
    for it in config.INSTRUMENTS:
        lbl, _ = macro.tilt_label(tilt[it["key"]])
        rows.append({"标的": it["name"], "宏观净评分": tilt[it["key"]], "方向倾向": lbl,
                     "所属资产桶": macro.BUCKET_CN.get(macro.INSTR_BUCKET.get(it["key"], ""), "")})
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={"宏观净评分": st.column_config.NumberColumn(
            format="%+d", help="本月已公布宏观对该标的的方向净评分：正=偏多，负=偏空，0=中性。只表方向不表幅度。")})

    with st.expander("ℹ️ 这些倾向怎么算的 / 查看全部本月数据"):
        st.caption("算法：每条宏观按其『较前值上行/下行』方向，对相关资产桶记 +1/-1（见 macro.py 的 eff 表），"
                   "再按标的所属资产桶汇总成净评分。只表方向倾向，不构成投资建议。")
        full = [{"指标": r["cfg"]["name"], "参考月": r["ref_date"].strftime("%Y-%m"),
                 "最新": macro.fmt_val(r["cfg"], r["value"]),
                 "前值": macro.fmt_val(r["cfg"], r["prior"]),
                 "方向": {"up": "▲上行", "down": "▼下行", "flat": "持平"}[r["dir"]],
                 "影响评论": macro.impact_comment(r["cfg"], r)} for r in fresh]
        st.dataframe(pd.DataFrame(full), hide_index=True, width="stretch")


def render_macro():
    st.title("🌍 宏观数据 · 月度总结与季节性")
    st.caption("来源：FRED 美联储经济数据（免密钥）。这是宏观 overlay 评论层，"
               "用于研判最新数据对各标的的方向影响，**不属于周期择时模型本身，也非投资建议**。")
    readings = all_macro_readings()
    if not readings:
        st.error("FRED 数据加载失败，稍后点侧栏 🔄 刷新重试。")
        return

    # ---- 月度总结表 ----
    st.subheader("📋 重要宏观数据 · 总结")
    arr = {"up": "▲", "down": "▼", "flat": "▬"}
    rows = []
    for r in readings:
        cfg = r["cfg"]
        meaning = cfg["up"] if r["dir"] == "up" else (cfg["down"] if r["dir"] == "down" else "持平")
        rows.append({"新": "🔵" if macro.is_fresh(r) else "",
                     "指标": cfg["name"], "频率": cfg["freq"],
                     "参考期": r["ref_date"].strftime("%Y-%m"),
                     "最新": macro.fmt_val(cfg, r["value"]),
                     "前值": macro.fmt_val(cfg, r["prior"]),
                     "方向": arr[r["dir"]], "主题": cfg["theme"], "读数含义": meaning})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("🔵 = 近期已公布批次（约70天内参考期）。")

    # ---- 季节性（过去 N 年 · 按日历月叠加 + 季节均线）----
    st.subheader("📈 季节性图（过去 N 年 · 按日历月叠加 + 季节均线）")
    themes = []
    for r in readings:
        if r["cfg"]["theme"] not in themes:
            themes.append(r["cfg"]["theme"])
    fc1, fc2, fc3 = st.columns([1.1, 2.2, 1])
    with fc1:
        theme_sel = st.selectbox("大类", ["全部"] + themes, index=0)
    pool = [r for r in readings if theme_sel == "全部" or r["cfg"]["theme"] == theme_sel]
    with fc2:
        sel = st.selectbox("指标（含分项）", [r["cfg"]["name"] for r in pool], index=0)
    with fc3:
        years = st.slider("回看年数", 3, 8, 5)
    rr = next(r for r in pool if r["cfg"]["name"] == sel)
    cfg = rr["cfg"]
    raw = load_macro_raw(cfg["id"])
    months = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
    piv, ly = macro.seasonal_by_year(raw, cfg["kind"], years) if raw is not None else (None, None)
    kind_unit = {"yoy": "同比 %", "mom": "环比 %", "mom_diff": "环比变化", "level": "水平/指数"}.get(cfg["kind"], "")
    if piv is not None:
        fig = go.Figure()
        for col in [c for c in piv.columns if c != "平均"]:
            is_cur = (col == ly)
            fig.add_trace(go.Scatter(
                x=months, y=piv[col].values, name=str(col), mode="lines+markers",
                line=dict(width=3 if is_cur else 1.4, color="#ff7043" if is_cur else None),
                opacity=1.0 if is_cur else 0.5))
        fig.add_trace(go.Scatter(
            x=months, y=piv["平均"].values, name=f"{years}年季节均线",
            mode="lines", line=dict(width=3.5, color="#ffffff", dash="dash")))
        fig.update_layout(template="plotly_dark", height=430, hovermode="x unified",
                          margin=dict(l=10, r=10, t=30, b=10), yaxis_title=kind_unit,
                          legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig, width="stretch")
        cm = pd.Timestamp.now().month
        cmv = piv.loc[cm, "平均"] if cm in piv.index else None
        st.caption(f"读图：每条细线 = 某一年的 12 个月路径，**橙色加粗 = 今年({ly})**，"
                   f"**白色虚线 = 过去{years}年季节均线**。同一月份看各年高低、以及今年相对均线的偏离，"
                   f"即该月的季节性强弱与今年的季节顺/逆风。当前 {cm} 月季节均值 ≈ "
                   f"{('%.2f' % cmv) if (cmv is not None and not pd.isna(cmv)) else '—'} {kind_unit}。"
                   f"口径：{kind_unit}（随指标而定）。季节性只是统计倾向，会被宏观大势/政策覆盖。")

        # 今年 vs 季节均线 的偏离柱（>0 季节顺风 / <0 逆风）
        if ly in piv.columns:
            dev = (piv[ly] - piv["平均"])
            colors = []
            for v in dev.values:
                if v is None or pd.isna(v):
                    colors.append("rgba(120,120,120,0.3)")
                else:
                    colors.append("#2ecc71" if v >= 0 else "#e74c3c")
            figd = go.Figure(go.Bar(x=months, y=dev.values, marker_color=colors,
                                    name="今年-均线"))
            figd.update_layout(template="plotly_dark", height=240,
                               margin=dict(l=10, r=10, t=40, b=10),
                               yaxis_title=f"今年偏离均线（{kind_unit}）",
                               title=f"今年({ly}) vs {years}年季节均线：🟢>0 季节顺风 / 🔴<0 逆风")
            st.plotly_chart(figd, width="stretch")
            st.caption("绿柱=今年该月高于季节常态（顺风），红柱=低于常态（逆风）；"
                       "只在今年已公布的月份有柱子。")
    else:
        st.info("该指标季节性数据不足。")

    # ---- 该指标走势 + 影响 ----
    st.subheader("🧭 最新方向 → 对各标的影响")
    st.info(macro.impact_comment(cfg, rr))
    ser = rr["series"]
    fig2 = go.Figure(go.Scatter(x=ser.index, y=ser.values, mode="lines",
                                line=dict(color="#ffb84d")))
    fig2.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=40, b=10),
                       title=f"{cfg['name']} 历史走势")
    st.plotly_chart(fig2, width="stretch")


# ----------------------------------------------------------------------
if view.startswith("🏠"):
    render_home(load_all(params))
elif view.startswith("🔬"):
    render_detail(load_all(params))
elif view.startswith("🌍"):
    render_macro()
else:
    render_help(params)
