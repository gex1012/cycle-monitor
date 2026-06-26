# -*- coding: utf-8 -*-
"""
macro.py — 宏观数据层 (FRED, 免 API Key)
=================================================
- 抓取美国关键月度/季度宏观序列（来源：FRED fredgraph.csv，无需密钥）
- 计算最新值 / 前值 / 变动方向、同比/环比变换
- 季节性图（按日历月平均的环比变化）
- "影响规则"：把宏观变动方向映射到各标的（启发式宏观 overlay，非周期模型）

声明：这是宏观启发式评论层，用于"最新数据对各标的方向倾向"的快速研判，
不构成投资建议，也不属于周期择时模型本身。
"""
import datetime as dt
import numpy as np
import pandas as pd

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"

# 资产桶
BUCKETS = ["usd", "rates", "gold", "silver", "us_eq", "em_eq", "semi", "copper", "oil"]
BUCKET_CN = {"usd": "美元", "rates": "美债收益率", "gold": "黄金", "silver": "白银",
             "us_eq": "美欧股", "em_eq": "新兴/亚太股", "semi": "半导体",
             "copper": "铜", "oil": "原油"}

# 标的 key -> 资产桶
INSTR_BUCKET = {
    "spx": "us_eq", "ndx": "us_eq", "soxx": "semi", "kospi": "em_eq",
    "nikkei": "em_eq", "star50": "em_eq", "sx5e": "us_eq", "bvsp": "em_eq",
    "dxy": "usd", "us10y": "rates", "gold": "gold", "silver": "silver",
    "brent": "oil", "copper": "copper",
}

# 宏观指标定义
#   kind:  yoy=同比% / mom=环比% / mom_diff=环比绝对差 / level=水平值
#   season: pct=对原序列取环比百分比做季节性 / diff=取环比绝对差
#   eff:   当该指标"上行"时，对各资产桶的方向(+利好/-利空/0中性)
MACRO = [
    {"id": "CPIAUCSL", "name": "CPI 同比", "kind": "yoy", "freq": "M", "fmt": "{:.1f}%",
     "season": "pct", "theme": "通胀", "up": "通胀升温→偏鹰", "down": "通胀降温→偏鸽",
     "eff": {"usd": +1, "rates": +1, "gold": -1, "silver": -1, "us_eq": -1, "em_eq": -1, "semi": -1, "copper": -1, "oil": +1}},
    {"id": "CPILFESL", "name": "核心CPI 同比", "kind": "yoy", "freq": "M", "fmt": "{:.1f}%",
     "season": "pct", "theme": "通胀", "up": "核心粘性通胀升温→偏鹰", "down": "核心通胀回落→偏鸽",
     "eff": {"usd": +1, "rates": +1, "gold": -1, "silver": -1, "us_eq": -1, "em_eq": -1, "semi": -1, "copper": 0, "oil": 0}},
    {"id": "PCEPILFE", "name": "核心PCE 同比", "kind": "yoy", "freq": "M", "fmt": "{:.1f}%",
     "season": "pct", "theme": "通胀(美联储口径)", "up": "美联储最看重的通胀升温→偏鹰", "down": "核心PCE回落→偏鸽",
     "eff": {"usd": +1, "rates": +1, "gold": -1, "silver": -1, "us_eq": -1, "em_eq": -1, "semi": -1, "copper": 0, "oil": 0}},
    {"id": "UNRATE", "name": "失业率", "kind": "level", "freq": "M", "fmt": "{:.1f}%",
     "season": "diff", "theme": "就业", "up": "失业上行→劳动力转弱→偏鸽/衰退担忧", "down": "失业回落→就业强→偏鹰",
     "eff": {"usd": -1, "rates": -1, "gold": +1, "silver": +1, "us_eq": -1, "em_eq": -1, "semi": 0, "copper": -1, "oil": -1}},
    {"id": "PAYEMS", "name": "非农就业(月增·千)", "kind": "mom_diff", "freq": "M", "fmt": "{:+,.0f}",
     "season": "diff", "theme": "就业", "up": "非农强劲→增长好但偏鹰", "down": "非农走弱→增长忧/偏鸽",
     "eff": {"usd": +1, "rates": +1, "gold": -1, "silver": -1, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "RSAFS", "name": "零售销售 同比", "kind": "yoy", "freq": "M", "fmt": "{:.1f}%",
     "season": "pct", "theme": "消费", "up": "消费需求走强→增长+", "down": "消费走弱→需求忧",
     "eff": {"usd": +1, "rates": +1, "gold": 0, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "INDPRO", "name": "工业产出 同比", "kind": "yoy", "freq": "M", "fmt": "{:.1f}%",
     "season": "pct", "theme": "工业", "up": "工业走强→周期/商品+", "down": "工业走弱→周期承压",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "UMCSENT", "name": "密歇根消费者信心", "kind": "level", "freq": "M", "fmt": "{:.1f}",
     "season": "diff", "theme": "信心", "up": "消费者信心回升→风险偏好+", "down": "信心走弱→避险",
     "eff": {"usd": -1, "rates": 0, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "FEDFUNDS", "name": "联邦基金有效利率", "kind": "level", "freq": "M", "fmt": "{:.2f}%",
     "season": "diff", "theme": "货币政策", "up": "政策利率上行→收紧", "down": "政策利率下行→宽松",
     "eff": {"usd": +1, "rates": +1, "gold": -1, "silver": -1, "us_eq": -1, "em_eq": -1, "semi": -1, "copper": -1, "oil": -1}},
    {"id": "HOUST", "name": "新屋开工(千)", "kind": "level", "freq": "M", "fmt": "{:,.0f}",
     "season": "pct", "theme": "地产", "up": "地产走强→利率敏感需求+", "down": "地产走弱→需求忧",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": 0, "copper": +1, "oil": +1}},
    {"id": "A191RL1Q225SBEA", "name": "实际GDP 季环比(折年)", "kind": "level", "freq": "Q", "fmt": "{:.1f}%",
     "season": "diff", "theme": "增长", "up": "增长加速→风险+", "down": "增长放缓→避险",
     "eff": {"usd": +1, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "T10Y2Y", "name": "10Y-2Y 利差", "kind": "level", "freq": "D", "fmt": "{:+.2f}", "monthly_last": True,
     "season": "diff", "theme": "收益率曲线", "up": "曲线变陡(脱离倒挂)→衰退预警缓解", "down": "曲线趋平/倒挂→衰退预警",
     "eff": {"usd": 0, "rates": 0, "gold": 0, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": 0}},

    # ===== 耐用品订单 + 分项（环比%）=====
    {"id": "DGORDER", "name": "耐用品订单·总(环比)", "kind": "mom", "freq": "M", "fmt": "{:+.1f}%",
     "season": "pct", "theme": "耐用品·资本开支", "up": "耐用品订单回升→制造/投资需求+", "down": "耐用品订单走弱→需求忧",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "ADXTNO", "name": "耐用品订单·ex运输(环比·分项)", "kind": "mom", "freq": "M", "fmt": "{:+.1f}%",
     "season": "pct", "theme": "耐用品·资本开支", "up": "剔除运输后核心需求走强", "down": "核心耐用品走弱",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "NEWORDER", "name": "核心资本品订单·ex飞机(环比·分项)", "kind": "mom", "freq": "M", "fmt": "{:+.1f}%",
     "season": "pct", "theme": "耐用品·资本开支", "up": "企业资本开支(capex)回升→设备/半导体需求+", "down": "capex走弱→设备投资忧",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},

    # ===== PMI 类（ISM 有版权，改用联储制造业调查·扩散指数·含分项）=====
    {"id": "GACDFSA066MSFRBPHI", "name": "费城联储制造业·总体(PMI类)", "kind": "level", "freq": "M", "fmt": "{:.1f}",
     "season": "diff", "theme": "制造业景气(PMI类)", "up": "制造业景气扩张(>0)→风险偏好+", "down": "制造业景气收缩→避险",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "NOCDFSA066MSFRBPHI", "name": "费城联储·新订单(分项)", "kind": "level", "freq": "M", "fmt": "{:.1f}",
     "season": "diff", "theme": "制造业景气(PMI类)", "up": "新订单走强→未来生产/景气+", "down": "新订单走弱→需求转弱",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "SHCDFSA066MSFRBPHI", "name": "费城联储·出货(分项)", "kind": "level", "freq": "M", "fmt": "{:.1f}",
     "season": "diff", "theme": "制造业景气(PMI类)", "up": "出货走强→当期产出+", "down": "出货走弱",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "NECDFSA066MSFRBPHI", "name": "费城联储·就业(分项)", "kind": "level", "freq": "M", "fmt": "{:.1f}",
     "season": "diff", "theme": "制造业景气(PMI类)", "up": "制造业招工走强→景气+", "down": "制造业用工收缩",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
    {"id": "PPCDFSA066MSFRBPHI", "name": "费城联储·支付价格(分项·通胀)", "kind": "level", "freq": "M", "fmt": "{:.1f}",
     "season": "diff", "theme": "制造业景气(PMI类)", "up": "投入价格上行→通胀压力(偏鹰)", "down": "投入价格回落→通胀缓和",
     "eff": {"usd": +1, "rates": +1, "gold": -1, "silver": -1, "us_eq": -1, "em_eq": -1, "semi": -1, "copper": +1, "oil": +1}},
    {"id": "GACDISA066MSFRBNY", "name": "纽约联储Empire制造业·总体(PMI类)", "kind": "level", "freq": "M", "fmt": "{:.1f}",
     "season": "diff", "theme": "制造业景气(PMI类)", "up": "Empire景气扩张→风险偏好+", "down": "Empire景气收缩→避险",
     "eff": {"usd": 0, "rates": +1, "gold": -1, "silver": 0, "us_eq": +1, "em_eq": +1, "semi": +1, "copper": +1, "oil": +1}},
]


def by_id(mid):
    for m in MACRO:
        if m["id"] == mid:
            return m
    return None


# ----------------------------------------------------------------------
def fetch_raw(mid):
    """抓取单个 FRED 序列，返回 Series(date->value)。失败返回 None。"""
    try:
        df = pd.read_csv(FRED_CSV.format(mid))
        c0, c1 = df.columns[0], df.columns[1]
        df[c0] = pd.to_datetime(df[c0])
        df[c1] = pd.to_numeric(df[c1], errors="coerce")
        s = df.dropna().set_index(c0)[c1].sort_index()
        return s
    except Exception as e:  # noqa
        print(f"[macro] {mid} 抓取失败: {e}")
        return None


def transform(raw, kind):
    """按 kind 把原序列转成展示口径。"""
    if raw is None or len(raw) == 0:
        return raw
    if kind == "yoy":
        return (raw.pct_change(12) * 100).dropna()
    if kind == "mom":
        return (raw.pct_change(1) * 100).dropna()
    if kind == "mom_diff":
        return raw.diff(1).dropna()
    return raw  # level


def latest(cfg, raw=None):
    """返回最新读数 dict：value/prior/chg/dir/ref_date/series(变换后)。"""
    if raw is None:
        raw = fetch_raw(cfg["id"])
    if raw is None or len(raw) < 2:
        return None
    # 对于日频(利差)取月末值，便于和月度宏观并列
    if cfg.get("monthly_last"):
        raw = raw.resample("ME").last().dropna()
    ser = transform(raw, cfg["kind"])
    if ser is None or len(ser) < 2:
        return None
    value = float(ser.iloc[-1])
    prior = float(ser.iloc[-2])
    chg = value - prior
    direction = "up" if chg > 1e-9 else ("down" if chg < -1e-9 else "flat")
    return {"cfg": cfg, "value": value, "prior": prior, "chg": chg,
            "dir": direction, "ref_date": pd.Timestamp(ser.index[-1]),
            "series": ser, "raw": raw}


def fmt_val(cfg, v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    try:
        return cfg["fmt"].format(v)
    except Exception:
        return f"{v:.2f}"


def seasonal_profile(raw, season):
    """按日历月平均的环比变化（季节性）。返回 (月份Series平均, 最近12个月实际)。"""
    s = raw.dropna()
    ch = (s.pct_change() * 100) if season == "pct" else s.diff()
    ch = ch.dropna()
    if len(ch) == 0:
        return None, None
    dfm = pd.DataFrame({"v": ch.values}, index=ch.index)
    dfm["m"] = dfm.index.month
    avg = dfm.groupby("m")["v"].mean().reindex(range(1, 13))
    # 最近 12 个月实际（按月对齐）
    recent = ch.iloc[-12:]
    cur = pd.Series(index=range(1, 13), dtype=float)
    for idx, val in recent.items():
        cur[idx.month] = val
    return avg, cur


def seasonal_by_year(raw, kind, years=5):
    """
    过去 years 年、按日历月的季节性叠加：返回 (DataFrame[index=月1..12, columns=各年份 + '平均'], 最新年份)。
    数值口径按指标 kind（同比/环比/水平）变换，便于同月跨年对比 + 画季节均线。
    """
    if raw is None or len(raw) == 0:
        return None, None
    s = transform(raw, kind).dropna()
    if len(s) == 0:
        return None, None
    df = pd.DataFrame({"v": s.values}, index=pd.DatetimeIndex(s.index))
    df["year"] = df.index.year
    df["month"] = df.index.month
    ly = int(df["year"].max())
    keep = list(range(ly - years + 1, ly + 1))
    df = df[df["year"].isin(keep)]
    piv = df.pivot_table(index="month", columns="year", values="v", aggfunc="last").reindex(range(1, 13))
    piv["平均"] = piv.mean(axis=1, skipna=True)
    return piv, ly


# ----------------------------------------------------------------------
def impact_signs(cfg, direction):
    """按方向返回各资产桶的影响符号(+/-/0)。direction=='down' 时翻转。"""
    sgn = 1 if direction == "up" else (-1 if direction == "down" else 0)
    return {b: cfg["eff"].get(b, 0) * sgn for b in BUCKETS}


def impact_comment(cfg, reading):
    """生成一句"最新数据→各资产"影响评论。"""
    d = reading["dir"]
    head = cfg["up"] if d == "up" else (cfg["down"] if d == "down" else "持平→影响中性")
    if d == "flat":
        return f"{cfg['name']} 持平，方向性影响有限。"
    signs = impact_signs(cfg, d)
    pos = [BUCKET_CN[b] for b, v in signs.items() if v > 0]
    neg = [BUCKET_CN[b] for b, v in signs.items() if v < 0]
    parts = [head]
    if pos:
        parts.append("利好 " + "、".join(pos))
    if neg:
        parts.append("利空 " + "、".join(neg))
    return "；".join(parts) + "。"


def instrument_tilt(fresh_readings):
    """聚合若干"最新宏观"对每个标的的净方向倾向。
    返回 {instr_key: {'score':int,'pos':int,'neg':int}}。"""
    out = {}
    for key, bucket in INSTR_BUCKET.items():
        score = 0
        for r in fresh_readings:
            sgn = 1 if r["dir"] == "up" else (-1 if r["dir"] == "down" else 0)
            score += r["cfg"]["eff"].get(bucket, 0) * sgn
        out[key] = score
    return out


def tilt_label(score):
    # 指标数较多，阈值放宽以反映"净倾向"
    if score >= 3:
        return "偏多 🟢", "green"
    if score >= 1:
        return "略偏多 🟢", "green"
    if score <= -3:
        return "偏空 🔴", "red"
    if score <= -1:
        return "略偏空 🔴", "red"
    return "中性 ⚪", "gray"


def is_fresh(reading, days=70):
    """
    最新观测是否属于"近期已公布批次"（用于首页『最新更新』高亮）。
    月度数据(参考月M)通常在 M+1 月公布，故用 ~70 天窗口覆盖当月发布的上一月批次。
    """
    if reading is None:
        return False
    return (pd.Timestamp.now().normalize() - reading["ref_date"]).days <= days
