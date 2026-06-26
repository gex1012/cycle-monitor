# -*- coding: utf-8 -*-
"""
data.py — 行情抓取层 (Yahoo Finance / yfinance)
"""
import time
import pandas as pd
import yfinance as yf


def _flatten(df):
    if df is None or df.empty:
        return None
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_one(ticker, period="8y", interval="1d", retries=2):
    """抓单个标的日线，返回 DataFrame 或 None。"""
    last_err = None
    for _ in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             progress=False, auto_adjust=True, threads=False)
            df = _flatten(df)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:  # noqa
            last_err = e
            time.sleep(1.0)
    if last_err:
        print(f"[data] {ticker} 抓取失败: {last_err}")
    return None


def fetch_close(ticker, period="8y"):
    """返回收盘价 Series（index=日期），失败返回 None。"""
    df = fetch_one(ticker, period=period)
    if df is None or "Close" not in df:
        return None
    return df["Close"].dropna()


def fetch_ohlc(ticker, period="8y"):
    """返回完整 OHLC DataFrame（用于交易建议的 ATR/摆动高低点），失败返回 None。"""
    df = fetch_one(ticker, period=period)
    if df is None or "Close" not in df:
        return None
    return df.dropna(subset=["Close"])


def last_quote(ticker):
    """尽量取最新一笔报价（含盘中），返回 (price, time) 或 (None, None)。"""
    try:
        df = yf.download(ticker, period="5d", interval="1m",
                         progress=False, auto_adjust=True, threads=False)
        df = _flatten(df)
        if df is not None and len(df):
            return float(df["Close"].dropna().iloc[-1]), df.index[-1]
    except Exception:
        pass
    return None, None
