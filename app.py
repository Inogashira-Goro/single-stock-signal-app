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
    """抓上市與上櫃清單，用於自動判斷 .TW / .TWO 與顯示股票名稱。"""
    stock_map = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for mode, suffix, market_name in [(2, ".TW", "上市"), (4, ".TWO", "上櫃")]:
        url = f"https://isin.twse.com.tw/isin/C_public.jsp?strMode={mode}"
        try:
            res = requests.get(url, headers=headers, verify=False, timeout=20)
            res.encoding = "big5"
            tables = pd.read_html(StringIO(res.text))
            df = tables[0].iloc[1:].copy()
        except Exception:
            continue

        for _, row in df.iterrows():
            try:
                # 第一欄通常長這樣：「2330　台積電」
                raw = str(row[0]).strip()
                parts = raw.split()
                if len(parts) < 2:
                    continue

                code = parts[0].strip()
                name = "".join(parts[1:]).strip()
                category = str(row[4]) if len(row) > 4 else ""

                if len(code) == 4 and code.isdigit() and category not in ["權證", "牛熊證", "認購(售)權證"]:
                    stock_map[code] = {
                        "name": name,
                        "ticker": f"{code}{suffix}",
                        "market": market_name,
                        "category": category,
                    }
            except Exception:
                continue

    return stock_map


def normalize_yfinance_df(data, ticker):
    """把 yfinance 回傳的各種欄位格式統一成 Open/High/Low/Close/Volume。"""
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()

    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))

        # 格式：('Close', '2330.TW')
        if "Close" in level0:
            df.columns = df.columns.get_level_values(0)
        # 格式：('2330.TW', 'Close')
        elif ticker in level0:
            df = df[ticker]
        # 保底：第二層含 Close
        elif "Close" in level1:
            df.columns = df.columns.get_level_values(1)

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    return df.dropna(how="all")


@st.cache_data(ttl=60 * 20, show_spinner=False)
def fetch_price_once(ticker, period):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            multi_level_index=False,
        )
    except TypeError:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

    df = normalize_yfinance_df(data, ticker)

    # yfinance download 偶爾抽風，改用 Ticker.history 再試一次
    if df.empty:
        try:
            df2 = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
            df = normalize_yfinance_df(df2, ticker)
        except Exception:
            pass

    return df


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

    mid = ma20
    std20 = close.rolling(20).std()
    upper = mid + 2 * std20
    lower = mid - 2 * std20

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
    reasons_good = []
    reasons_bad = []

    if last > ma5_last:
        score += 10
        reasons_good.append("收盤價站上 MA5，短線沒有明顯轉弱")
    else:
        score -= 15
        reasons_bad.append("收盤價跌破 MA5，短線轉弱")

    if last > ma20_last:
        score += 15
        reasons_good.append("收盤價站上 MA20")
    else:
        score -= 20
        reasons_bad.append("收盤價跌破 MA20")

    if last > ma60_last:
        score += 15
        reasons_good.append("收盤價站上 MA60")
    else:
        score -= 20
        reasons_bad.append("收盤價跌破 MA60")

    if ma5_last > ma20_last > ma60_last:
        score += 20
        reasons_good.append("MA5 > MA20 > MA60，多頭排列")
    elif ma5_last < ma20_last < ma60_last:
        score -= 25
        reasons_bad.append("MA5 < MA20 < MA60，空頭排列")
    else:
        reasons_bad.append("均線排列仍不乾淨")

    if roc10 > 3:
        score += 15
        reasons_good.append(f"10 日漲跌幅為 {roc10:.2f}%，短線動能偏強")
    elif roc10 < -3:
        score -= 15
        reasons_bad.append(f"10 日漲跌幅為 {roc10:.2f}%，短線動能偏弱")
    else:
        score += 3
        reasons_good.append(f"10 日漲跌幅為 {roc10:.2f}%，短線變化不大")

    if not math.isnan(rsi_last):
        if 50 <= rsi_last <= 70:
            score += 12
            reasons_good.append(f"RSI 約 {rsi_last:.1f}，偏強但尚未極端過熱")
        elif rsi_last > 78:
            score -= 12
            reasons_bad.append(f"RSI 約 {rsi_last:.1f}，短線過熱風險較高")
        elif rsi_last < 45:
            score -= 8
            reasons_bad.append(f"RSI 約 {rsi_last:.1f}，動能偏弱")

    if not math.isnan(vol_ratio):
        if vol_ratio >= 1.2:
            score += 8
            reasons_good.append(f"近 5 日量能約為 20 日均量的 {vol_ratio:.2f} 倍，量能有放大")
        elif vol_ratio < 0.7:
            score -= 5
            reasons_bad.append(f"近 5 日量能約為 20 日均量的 {vol_ratio:.2f} 倍，量能偏弱")

    if dist_ma20 > 18:
        score -= 15
        reasons_bad.append(f"距 MA20 約 {dist_ma20:.2f}%，可能已有追高風險")
    elif dist_ma20 < -8:
        score -= 10
        reasons_bad.append(f"低於 MA20 約 {abs(dist_ma20):.2f}%，型態偏弱")

    if score >= 60:
        verdict = "建議買入"
        tone = "技術面偏多，但仍建議小部位、設停損，不宜無腦追高。"
        color = "#16a34a"  # green
        bg = "#dcfce7"
    elif score >= 25:
        verdict = "可觀望"
        tone = "訊號尚未一致，可以列入觀察，等突破或拉回不破關鍵均線再說。"
        color = "#ca8a04"  # yellow/amber
        bg = "#fef9c3"
    else:
        verdict = "不建議進場"
        tone = "目前技術面條件不足，追進去的風險大於優勢。"
        color = "#dc2626"  # red
        bg = "#fee2e2"

    return {
        "verdict": verdict,
        "tone": tone,
        "score": score,
        "color": color,
        "bg": bg,
        "last": last,
        "ma5": ma5_last,
        "ma20": ma20_last,
        "ma60": ma60_last,
        "rsi": rsi_last,
        "roc5": roc5,
        "roc10": roc10,
        "roc20": roc20,
        "dist_ma20": dist_ma20,
        "dist_ma60": dist_ma60,
        "vol_ratio": vol_ratio,
        "bb_pos": bb_pos,
        "rows": len(close),
        "reasons_good": reasons_good,
        "reasons_bad": reasons_bad,
    }


def choose_and_fetch(code, market_mode, period, stock_map):
    """依市場選擇抓資料；自動模式下若 .TW 失敗會自動試 .TWO，並盡量帶出股票名稱。"""
    # 無論抓價是否成功，股票名稱都優先從證交所/櫃買清單來
    display_name = stock_map.get(code, {}).get("name", code)

    candidates = []

    if market_mode == "上市 .TW":
        candidates = [(f"{code}.TW", display_name, "上市")]
    elif market_mode == "上櫃 .TWO":
        candidates = [(f"{code}.TWO", display_name, "上櫃")]
    else:
        if code in stock_map:
            info = stock_map[code]
            primary = info["ticker"]
            display_name = info["name"]
            if primary.endswith(".TW"):
                candidates = [
                    (primary, display_name, "上市"),
                    (f"{code}.TWO", display_name, "上櫃"),
                ]
            else:
                candidates = [
                    (primary, display_name, "上櫃"),
                    (f"{code}.TW", display_name, "上市"),
                ]
        else:
            candidates = [
                (f"{code}.TW", display_name, "上市"),
                (f"{code}.TWO", display_name, "上櫃"),
            ]

    attempts = []
    for ticker, name, market in candidates:
        df = fetch_price_once(ticker, period)
        attempts.append((ticker, 0 if df is None else len(df)))
        if df is not None and not df.empty:
            return ticker, name, market, df, attempts

    return candidates[-1][0], display_name, "抓取失敗", pd.DataFrame(), attempts


with st.form("input_form"):
    code = st.text_input("股票代號", placeholder="例如：2330、2317、6664、0050").strip()
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

        ticker, name, market, df, attempts = choose_and_fetch(code, market_mode, period, stock_map)
        result = analyze(df)

    if show_debug:
        st.write("實際成功/最後抓取代號：", ticker)
        st.write("嘗試紀錄：", attempts)
        st.write("抓到資料列數：", 0 if df is None else len(df))
        if df is not None and not df.empty:
            st.write("欄位：", [str(c) for c in df.columns])
            st.dataframe(df.tail(), use_container_width=True)

    if result is None:
        st.error("資料不足或抓取失敗。若是上櫃股票，請手動切換為「上櫃 .TWO」後再試。")
        st.stop()

    # 只顯示「代號 股票名稱」，不顯示 market / ticker / fallback
    st.subheader(f"{code} {name}")

    st.markdown(
        f"""
        <div style="
            padding: 1rem 1.1rem;
            border-radius: 0.9rem;
            background: {result['bg']};
            border: 1px solid {result['color']}33;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.95rem; color: #374151; margin-bottom: 0.25rem;">技術面快評</div>
            <div style="font-size: 2rem; font-weight: 800; color: {result['color']}; line-height: 1.2;">
                {result['verdict']}
            </div>
            <div style="font-size: 1rem; color: #374151; margin-top: 0.4rem;">
                分數 {result['score']:.1f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        raw = pd.DataFrame(
            [
                ["收盤價", result["last"]],
                ["MA5", result["ma5"]],
                ["MA20", result["ma20"]],
                ["MA60", result["ma60"]],
                ["5日漲跌幅%", result["roc5"]],
                ["10日漲跌幅%", result["roc10"]],
                ["20日漲跌幅%", result["roc20"]],
                ["距MA20%", result["dist_ma20"]],
                ["距MA60%", result["dist_ma60"]],
                ["近5日/20日均量", result["vol_ratio"]],
                ["布林區間位置", result["bb_pos"]],
                ["有效資料筆數", result["rows"]],
            ],
            columns=["指標", "數值"],
        )
        st.dataframe(raw, use_container_width=True, hide_index=True)

    st.warning("提醒：本工具只是依固定技術規則產生快評，不含基本面、新聞、籌碼、法說、財報與重大事件分析。請不要把它當作保證獲利的買賣建議。")
else:
    st.info("輸入股票代號後按「分析」。例如 2330、2317、6664、0050。")
