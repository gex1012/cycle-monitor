# -*- coding: utf-8 -*-
"""
config.py — 监测标的清单
=================================================
每个标的字段：
  key     唯一标识
  name    中文名
  ticker  Yahoo Finance 代码（数据来源 https://finance.yahoo.com ）
  group   分组（用于首页归类）
  log     周期分析是否取对数（利率类用 False）
  pivot   斐波那契关键拐点；None=自动探测，或 'YYYY-MM-DD' 手动锚定
  note    指引备注
"""

# 数据来源说明（首页展示）
DATA_SOURCE = "Yahoo Finance (finance.yahoo.com) · 日线收盘"

INSTRUMENTS = [
    # ---- 股票指数 ----
    {"key": "spx",   "name": "美股·标普500",    "ticker": "^GSPC",     "group": "股票指数", "log": True,  "pivot": None,
     "note": "美股大盘基准"},
    {"key": "ndx",   "name": "美股·纳指100",    "ticker": "^NDX",      "group": "股票指数", "log": True,  "pivot": None,
     "note": "成长/科技权重"},
    {"key": "soxx",  "name": "费城半导体·SOXX", "ticker": "SOXX",      "group": "股票指数", "log": True,  "pivot": None,
     "note": "半导体设备/材料板块 overlay 的代理"},
    {"key": "kospi", "name": "韩国·KOSPI",      "ticker": "^KS11",     "group": "股票指数", "log": True,  "pivot": None,
     "note": "亚洲科技/出口周期"},
    {"key": "nikkei","name": "日本·日经225",    "ticker": "^N225",     "group": "股票指数", "log": True,  "pivot": None,
     "note": "日股大盘"},
    {"key": "star50","name": "中国·科创50",     "ticker": "588000.SS", "group": "股票指数", "log": True,  "pivot": "2024-09-24",
     "note": "用华夏科创50ETF代理；斐波拐点锚定 24/9/24 大底"},
    {"key": "sx5e",  "name": "欧洲·斯托克50",   "ticker": "^STOXX50E", "group": "股票指数", "log": True,  "pivot": None,
     "note": "欧股大盘"},
    {"key": "bvsp",  "name": "巴西·IBOVESPA",   "ticker": "^BVSP",     "group": "股票指数", "log": True,  "pivot": None,
     "note": "新兴市场/商品货币关联"},

    # ---- 外汇 / 利率 ----
    {"key": "dxy",   "name": "美元指数 DXY",    "ticker": "DX-Y.NYB",  "group": "外汇·利率", "log": True,  "pivot": None,
     "note": "全球风险/流动性总开关"},
    {"key": "us10y", "name": "10年期美债收益率","ticker": "^TNX",      "group": "外汇·利率", "log": False, "pivot": None,
     "note": "收益率（%），非价格；上行=债价下跌"},

    # ---- 贵金属 / 商品 ----
    {"key": "gold",  "name": "黄金 (COMEX)",    "ticker": "GC=F",      "group": "贵金属·商品", "log": True, "pivot": None,
     "note": "避险/实际利率反向"},
    {"key": "silver","name": "白银 (COMEX)",    "ticker": "SI=F",      "group": "贵金属·商品", "log": True, "pivot": None,
     "note": "弹性大于黄金"},
    {"key": "brent", "name": "布伦特原油",      "ticker": "BZ=F",      "group": "贵金属·商品", "log": True, "pivot": None,
     "note": "宏观对冲组的核心（WTI 68 建仓思路代理）"},
    {"key": "copper","name": "铜 (COMEX)",      "ticker": "HG=F",      "group": "贵金属·商品", "log": True, "pivot": None,
     "note": "全球需求/再通胀晴雨表"},
]

# 默认参数
DEFAULTS = {
    "period": "8y",        # 拉取历史长度
    "project": 200,        # 向前投影交易日（约 9-10 个月）
    "n_harmonics": 5,      # 合成周期使用的主导分量数
    "tol_days": 4,         # 时间共振聚类容差（日历日）
    "nominal": (65, 130, 260),   # 名义周期：≈13周/26周/52周
    "micro_band": (40, 150),     # 微周期搜索带（交易日）
}


def by_key(key):
    for it in INSTRUMENTS:
        if it["key"] == key:
            return it
    return None
