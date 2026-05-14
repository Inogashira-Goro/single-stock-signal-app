# -*- coding: utf-8 -*-
import math
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import urllib3
import yfinance as yf

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="個股技術面快評", page_icon="📊", layout="centered")
st.title("📊 個股技術面快評")
st.caption("輸入台股代號後，抓取股價資料並依技術面規則產生簡短評語。這是輔助觀察工具，不是投資建議。")


@st.cache_data(ttl=60 * 60 * 24)
def get_tw_stock_list():
    stock_map = {}
    headers = {"User-Agent": "Mozilla/5.0"}
    for mode, suffix, market_name in [(2, ".TW", "上市"), (4, ".TWO", "上櫃")]:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        res = requests.get(url, headers=headers, verify=False, timeout=20)
        res.encoding = "big5"
        df = pd.read_html(StringIO(res.text))[0].iloc[1:].copy()
        for _, row in df.iterrows():
            try:
                code_name = str(row[0]).split()
                if len(code_name) != 2:
                    continue
                code, name = code_name
                category = str(row[4])
                if len(code) == 4 and category not in ["權證", "牛熊證", "認購(售)權證"]:
                    stock_map[code] = {"name": name, "ticker": f"{code}{suffix}", "market": market_name, "category": category}
            except Exception:
                continue
    return stock_map


def normalize_yfinance_df(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))
        if "Close" in level0:
            df.columns = df.columns.get_level_values(0)
        elif ticker in level0:
            df = df[ticker]
        elif "Close" in level1:
            df.columns = df.columns.get_level_values(1)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df.dropna(how="all")


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_price(ticker, period):
    try:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False, multi_level_index=False)
    except TypeError:
        data = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
    return normalize_yfinance_df(data, ticker)


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def analyze(df):
    if df is None or df.empty or len(df) < 65:
        return None
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    if "Close" not in df.columns:
        return None
    if "Volume" not in df.columns:
        df["Volume"] = 0

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    volume = pd.to_numeric(df["Volume"], errors="coerce").reindex(close.index).fillna(0)
    if len(close) < 65:
        return None

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    rsi = calc_rsi(close, 14)
    std20 = close.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    last = close.iloc[-1]
    ma5_last = ma5.iloc[-1]
    ma20_last = ma20.iloc[-1]
    ma60_last = ma60.iloc[-1]
    rsi_last = rsi.iloc[-1]
    upper_last = upper.iloc[-1]
    lower_last = lower.iloc[-1]
    roc5 = (last / close.iloc[-6] - 1) * 100
    roc10 = (last / close.iloc[-11] - 1) * 100
    roc20 = (last / close.iloc[-21] - 1) * 100
    dist_ma20 = (last / ma20_last - 1) * 100
    dist_ma60 = (last / ma60_last - 1) * 100
    vol20 = volume.tail(20).mean()
    vol_ratio = volume.tail(5).mean() / vol20 if vol20 else np.nan
    bb_pos = (last - lower_last) / (upper_last - lower_last) if upper_last != lower_last else np.nan

    score = 0
    good, bad = [], []
    if last > ma5_last:
        score += 10; good.append("收盤價站上 MA5，短線沒有明顯轉弱")
    else:
        score -= 15; bad.append("收盤價跌破 MA5，短線轉弱")
    if last > ma20_last:
        score += 15; good.append("收盤價站上 MA20")
    else:
        score -= 20; bad.append("收盤價跌破 MA20")
    if last > ma60_last:
        score += 15; good.append("收盤價站上 MA60")
    else:
        score -= 20; bad.append("收盤價跌破 MA60")
    if ma5_last > ma20_last > ma60_last:
        score += 20; good.append("MA5 > MA20 > MA60，多頭排列")
    elif ma5_last < ma20_last < ma60_last:
        score -= 25; bad.append("MA5 < MA20 < MA60，空頭排列")
    else:
        bad.append("均線排列仍不乾淨")
    if roc10 > 3:
        score += 15; good.append(f"10 日漲跌幅為 {roc10:.2f}%，短線動能偏強")
    elif roc10 < -3:
        score -= 15; bad.append(f"10 日漲跌幅為 {roc10:.2f}%，短線動能偏弱")
    else:
        score += 3; good.append(f"10 日漲跌幅為 {roc10:.2f}%，短線變化不大")
    if not math.isnan(rsi_last):
        if 50 <= rsi_last <= 70:
            score += 12; good.append(f"RSI 約 {rsi_last:.1f}，偏強但尚未極端過熱")
        elif rsi_last > 78:
            score -= 12; bad.append(f"RSI 約 {rsi_last:.1f}，短線過熱風險較高")
        elif rsi_last < 45:
            score -= 8; bad.append(f"RSI 約 {rsi_last:.1f}，動能偏弱")
    if not math.isnan(vol_ratio):
        if vol_ratio >= 1.2:
            score += 8; good.append(f"近 5 日量能約為 20 日均量的 {vol_ratio:.2f} 倍，量能有放大")
        elif vol_ratio < 0.7:
            score -= 5; bad.append(f"近 5 日量能約為 20 日均量的 {vol_ratio:.2f} 倍，量能偏弱")
    if dist_ma20 > 18:
        score -= 15; bad.append(f"距 MA20 約 {dist_ma20:.2f}%，可能已有追高風險")
    elif dist_ma20 < -8:
        score -= 10; bad.append(f"低於 MA20 約 {abs(dist_ma20):.2f}%，型態偏弱")

    if score >= 60:
        verdict, tone = "建議買入", "技術面偏多，但仍建議小部位、設停損，不宜無腦追高。"
    elif score >= 25:
        verdict, tone = "可觀望", "訊號尚未一致，可以列入觀察，等突破或拉回不破關鍵均線再說。"
    else:
        verdict, tone = "不建議進場", "目前技術面條件不足，追進去的風險大於優勢。"

    return {"verdict": verdict, "tone": tone, "score": score, "last": last, "ma5": ma5_last, "ma20": ma20_last, "ma60": ma60_last, "rsi": rsi_last, "roc5": roc5, "roc10": roc10, "roc20": roc20, "dist_ma20": dist_ma20, "dist_ma60": dist_ma60, "vol_ratio": vol_ratio, "bb_pos": bb_pos, "rows": len(close), "reasons_good": good, "reasons_bad": bad}


with st.form("input_form"):
    code = st.text_input("股票代號", placeholder="例如：2330、2317、0050").strip()
    market_mode = st.radio("市場判斷", ["自動判斷", "上市 .TW", "上櫃 .TWO"], horizontal=True)
    period = st.selectbox("股價資料期間", ["6mo", "1y", "2y"], index=0)
    show_debug = st.checkbox("顯示除錯資訊", value=False)
    submitted = st.form_submit_button("分析", use_container_width=True, type="primary")

if submitted:
    if not code:
        st.warning("請先輸入股票代號。")
        st.stop()
    code = code.replace(".TW", "").replace(".TWO", "").strip()
    with st.spinner("正在抓取資料並分析..."):
        try:
            stock_map = get_tw_stock_list()
        except Exception:
            stock_map = {}
        if market_mode == "上市 .TW":
            ticker, name, market = f"{code}.TW", code, "上市"
        elif market_mode == "上櫃 .TWO":
            ticker, name, market = f"{code}.TWO", code, "上櫃"
        elif code in stock_map:
            ticker, name, market = stock_map[code]["ticker"], stock_map[code]["name"], stock_map[code]["market"]
        else:
            ticker, name, market = f"{code}.TW", code, "未自動辨識，先以上市代號嘗試"
        df = fetch_price(ticker, period)
        result = analyze(df)

    if show_debug:
        st.write("嘗試抓取代號：", ticker)
        st.write("抓到資料列數：", 0 if df is None else len(df))
        if df is not None and not df.empty:
            st.write("欄位：", [str(c) for c in df.columns])
            st.dataframe(df.tail(), use_container_width=True)

    if result is None:
        st.error("資料不足或抓取失敗。請確認股票代號是否正確，或手動切換上市/上櫃後再試。")
        st.stop()

    st.subheader(f"{code} {name}｜{market}")
    st.metric("技術面快評", result["verdict"], delta=f"分數 {result['score']:.1f}")
    st.write(result["tone"])
    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("收盤價", f"{result['last']:.2f}")
    c2.metric("10日漲跌幅", f"{result['roc10']:.2f}%")
    c3.metric("RSI", "N/A" if math.isnan(result["rsi"]) else f"{result['rsi']:.1f}")
    c4, c5, c6 = st.columns(3)
    c4.metric("MA5", f"{result['ma5']:.2f}")
    c5.metric("MA20", f"{result['ma20']:.2f}")
    c6.metric("MA60", f"{result['ma60']:.2f}")
    with st.expander("為什麼是這個評語？", expanded=True):
        if result["reasons_good"]:
            st.markdown("**偏多因素**")
            for r in result["reasons_good"][:4]:
                st.write(f"✅ {r}")
        if result["reasons_bad"]:
            st.markdown("**風險因素**")
            for r in result["reasons_bad"][:4]:
                st.write(f"⚠️ {r}")
    with st.expander("原始指標"):
        raw = pd.DataFrame([
            ["收盤價", result["last"]], ["MA5", result["ma5"]], ["MA20", result["ma20"]], ["MA60", result["ma60"]],
            ["5日漲跌幅%", result["roc5"]], ["10日漲跌幅%", result["roc10"]], ["20日漲跌幅%", result["roc20"]],
            ["距MA20%", result["dist_ma20"]], ["距MA60%", result["dist_ma60"]], ["近5日/20日均量", result["vol_ratio"]],
            ["布林區間位置", result["bb_pos"]], ["有效資料筆數", result["rows"]],
        ], columns=["指標", "數值"])
        st.dataframe(raw, use_container_width=True, hide_index=True)
    st.warning("提醒：本工具只是依固定技術規則產生快評，不含基本面、新聞、籌碼、法說、財報與重大事件分析。請不要把它當作保證獲利的買賣建議。")
else:
    st.info("輸入股票代號後按「分析」。例如 2330、2317、0050。")
