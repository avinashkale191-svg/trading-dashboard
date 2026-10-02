import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf
import io
import requests as http_requests

try:
    from nselib import derivatives, capital_market
    NSELIB_AVAILABLE = True
except Exception:
    NSELIB_AVAILABLE = False


# Upstox integration
try:
    from upstox_data import get_spot_quote, get_option_chain, compute_max_pain
    UPSTOX_AVAILABLE = True
except Exception:
    UPSTOX_AVAILABLE = False
st.set_page_config(page_title="Trading Dashboard", page_icon="📈", layout="wide")

# =========================================================================
# AUTO-REFRESH
# =========================================================================
if 'refresh_counter' not in st.session_state:
    st.session_state['refresh_counter'] = 0

def trigger_refresh():
    st.session_state['refresh_counter'] += 1

st.title("📈 Global Trading & Smart Money Dashboard")

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.header("⚙️ Settings")

st.sidebar.markdown("### 🔄 Data Refresh")
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 30s)", value=False)
if st.sidebar.button("🔄 Manual Refresh Now", use_container_width=True):
    trigger_refresh()
    st.rerun()

if 'last_update_str' not in st.session_state:
    st.session_state['last_update_str'] = datetime.now().strftime('%H:%M:%S')

st.sidebar.caption(f"Last update: **{st.session_state['last_update_str']}**")

if auto_refresh:
    st.markdown(
        """
        <script>
        setTimeout(function() { window.location.reload(); }, 15000);
        </script>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

# --- ASSET DROPDOWN ---
st.sidebar.subheader("📊 Select Asset")

asset_options = {
    "--- INDIAN INDICES ---": None,
    "NIFTY 50": "NSE_INDEX|Nifty 50",
    "BANK NIFTY": "NSE_INDEX|Nifty Bank",
    "SENSEX": "BSE_INDEX|SENSEX",
    "INDIA VIX": "NSE_INDEX|India VIX",
    "--- INDIAN STOCKS ---": None,
    "RELIANCE": "NSE_EQ|INE002A01018",
    "TCS": "NSE_EQ|INE467B01029",
    "INFY": "NSE_EQ|INE009A01021",
    "HDFCBANK": "NSE_EQ|INE040A01034",
    "SBIN": "NSE_EQ|INE062A01020",
    "ITC": "NSE_EQ|INE154A01025",
    "TATAMOTORS": "NSE_EQ|INE155A01022",
}
valid_options = {k: v for k, v in asset_options.items() if v is not None}
selected_label = st.sidebar.selectbox("Select Asset", list(valid_options.keys()))
ticker_symbol = valid_options[selected_label]

manual_ticker = st.sidebar.text_input("Custom Ticker (overrides dropdown)", value="")
if manual_ticker.strip() != "":
    ticker_symbol = manual_ticker.strip()
    selected_label = ticker_symbol

st.sidebar.caption("Examples: AAPL, ^GSPC, GC=F, BTC-USD, RELIANCE.NS")

# --- TIMEFRAME ---
st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Timeframe")
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=4)

interval_limits = {
    "1m":  ["1d", "5d"],
    "5m":  ["5d", "1mo"],
    "15m": ["5d", "1mo", "3mo"],
    "30m": ["5d", "1mo", "3mo"],
    "1h":  ["5d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
}
available_periods = interval_limits.get(interval, ["1mo", "3mo"])
default_period_idx = min(1, len(available_periods) - 1)
period = st.sidebar.selectbox("Period", available_periods, index=default_period_idx)
st.sidebar.caption(f"✅ Valid: {', '.join(available_periods)}")

show_last_n = st.sidebar.slider("Bars to show (zoom)", 30, 500, 150, step=10)

# --- INDICATORS (AUTO-ADJUST BY TIMEFRAME) ---
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Indicators")
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
sma_fast = st.sidebar.slider("SMA Fast", 5, 50, 20)
sma_slow = st.sidebar.slider("SMA Slow", 20, 200, 50)

# Auto-adjust S/R pivot by timeframe
auto_sr_pivot = {
    "1m": 20, "5m": 15, "15m": 12, "30m": 10, "1h": 8, "1d": 5
}.get(interval, 10)

pivot_len = st.sidebar.slider(
    "S/R Pivot Length",
    3, 30,
    value=auto_sr_pivot,
    help="Auto-set based on timeframe. Larger = fewer, cleaner levels."
)
st.sidebar.caption(f"📌 Auto for {interval}: {auto_sr_pivot}")

# Auto-adjust SMC pivot by timeframe
auto_smc_pivot = {
    "1m": 10, "5m": 8, "15m": 6, "30m": 5, "1h": 5, "1d": 3
}.get(interval, 5)

smc_pivot = st.sidebar.slider(
    "SMC Pivot Length",
    3, 30,
    value=auto_smc_pivot,
    help="Auto-set based on timeframe."
)
st.sidebar.caption(f"📌 Auto for {interval}: {auto_smc_pivot}")

auto_trend_pivot = {
    "1m": 5, "5m": 5, "15m": 6, "30m": 6, "1h": 8, "1d": 10
}.get(interval, 8)

trend_pivot = st.sidebar.slider(
    "Trend Line Pivot Length",
    3, 30,
    value=auto_trend_pivot,
    help="Auto-set based on timeframe."
)
st.sidebar.caption(f"📌 Auto for {interval}: {auto_trend_pivot}")

# --- OVERLAYS ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Overlays")
show_sr = st.sidebar.checkbox("Support / Resistance", value=True)
show_smc = st.sidebar.checkbox("SMC Liquidity (BSL/SSL)", value=False)
show_bos = st.sidebar.checkbox("BOS / CHoCH", value=True)
show_sma = st.sidebar.checkbox("Moving Averages", value=True)
show_trendlines = st.sidebar.checkbox("Auto Trend Lines", value=True)
extend_trendlines = st.sidebar.checkbox("Extend Trend Lines", value=True)
show_rr_boxes = st.sidebar.checkbox("R:R Boxes", value=False)

# =========================================================================
# DATA FETCH
# =========================================================================
def fetch_data(symbol, period, interval, refresh_key):
    """Fetch OHLC candles. Upstox-only for Indian symbols.
    Symbol from dropdown is ALREADY the Upstox instrument key.
    """
    try:
        from upstox_data import fetch_candles
    except Exception as _imp:
        print(f"Could not import fetch_candles: {_imp}")
        return None

    up_symbol = str(symbol).strip()

    if "|" not in up_symbol:
        fallback_map = {
            "NIFTY": "NSE_INDEX|Nifty 50",
            "NIFTY 50": "NSE_INDEX|Nifty 50",
            "NSEI": "NSE_INDEX|Nifty 50",
            "BANKNIFTY": "NSE_INDEX|Nifty Bank",
            "BANK NIFTY": "NSE_INDEX|Nifty Bank",
            "NSEBANK": "NSE_INDEX|Nifty Bank",
            "SENSEX": "BSE_INDEX|SENSEX",
            "BSESN": "BSE_INDEX|SENSEX",
            "INDIA VIX": "NSE_INDEX|India VIX",
            "INDIAVIX": "NSE_INDEX|India VIX",
            "RELIANCE": "NSE_EQ|INE002A01018",
            "TCS": "NSE_EQ|INE467B01029",
            "INFY": "NSE_EQ|INE009A01021",
            "HDFCBANK": "NSE_EQ|INE040A01034",
            "SBIN": "NSE_EQ|INE062A01020",
            "ITC": "NSE_EQ|INE154A01025",
            "TATAMOTORS": "NSE_EQ|INE155A01022",
        }
        up_symbol = fallback_map.get(up_symbol.upper(), up_symbol)

    up_interval = interval if interval in ["1m", "5m", "15m", "30m", "1h", "1d"] else "1m"

    try:
        df = fetch_candles(symbol=up_symbol, interval=up_interval, days=10)
        if df is not None and not df.empty:
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            if not df.empty:
                return df
        print(f"Upstox returned no data for {up_symbol}")
        return None
    except Exception as e:
        print(f"Upstox fetch failed for {up_symbol}: {e}")
        return None

# =========================================================================
# INDICATORS
# =========================================================================
def calc_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_order_flow(df):
    bar_range = df['High'] - df['Low']
    bar_body = (df['Close'] - df['Open']).abs()
    body_ratio = np.where(bar_range > 0, bar_body / bar_range, 0)
    raw = np.where(df['Close'] > df['Open'], df['Volume'] * body_ratio,
          np.where(df['Close'] < df['Open'], -df['Volume'] * body_ratio, 0))
    return pd.Series(raw, index=df.index).rolling(3).mean()

def find_sr(df, pivot_len):
    highs = df['High'].values
    lows = df['Low'].values
    res, sup = [], []
    for i in range(pivot_len, len(df) - pivot_len):
        if highs[i] == highs[i-pivot_len:i+pivot_len+1].max():
            res.append({'price': float(highs[i]), 'bar': i})
        if lows[i] == lows[i-pivot_len:i+pivot_len+1].min():
            sup.append({'price': float(lows[i]), 'bar': i})
    return res, sup

def find_smc(df, pivot_len=5, tol=0.1):
    highs = df['High'].values
    lows = df['Low'].values
    bsl, ssl = [], []
    for i in range(pivot_len, len(df) - pivot_len):
        if highs[i] == highs[i-pivot_len:i+pivot_len+1].max():
            bsl.append({'price': float(highs[i]), 'bar': i, 'eq': False, 'swept': False})
        if lows[i] == lows[i-pivot_len:i+pivot_len+1].min():
            ssl.append({'price': float(lows[i]), 'bar': i, 'eq': False, 'swept': False})
    for i, lvl in enumerate(bsl):
        for j in range(i+1, len(bsl)):
            if abs(lvl['price'] - bsl[j]['price']) / lvl['price'] * 100 <= tol:
                bsl[j]['eq'] = True
    for i, lvl in enumerate(ssl):
        for j in range(i+1, len(ssl)):
            if abs(lvl['price'] - ssl[j]['price']) / ssl[j]['price'] * 100 <= tol:
                ssl[j]['eq'] = True
    for lvl in bsl:
        for k in range(lvl['bar']+1, len(df)):
            if df['High'].iloc[k] > lvl['price'] and df['Close'].iloc[k] < lvl['price']:
                lvl['swept'] = True
                lvl['sweep_bar'] = k
                break
    for lvl in ssl:
        for k in range(lvl['bar']+1, len(df)):
            if df['Low'].iloc[k] < lvl['price'] and df['Close'].iloc[k] > lvl['price']:
                lvl['swept'] = True
                lvl['sweep_bar'] = k
                break
    return bsl, ssl

def find_bos(df, pivot_len=5):
    events = []
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    last_high = None
    last_low = None
    for i in range(pivot_len, len(df) - pivot_len):
        if highs[i] == highs[i-pivot_len:i+pivot_len+1].max():
            last_high = {'price': highs[i], 'bar': i}
        if lows[i] == lows[i-pivot_len:i+pivot_len+1].min():
            last_low = {'price': lows[i], 'bar': i}
        if last_high and closes[i] > last_high['price']:
            events.append({'type': 'BOS_UP', 'bar': i, 'price': float(last_high['price'])})
            last_high = None
        if last_low and closes[i] < last_low['price']:
            events.append({'type': 'BOS_DOWN', 'bar': i, 'price': float(last_low['price'])})
            last_low = None
    return events[-10:]

def find_trend_lines(df, pivot_len=8):
    """Auto trend lines. Works on any timeframe."""
    highs = df['High'].values
    lows = df['Low'].values
    dates = df['Date'].values

    swing_highs = []
    swing_lows = []

    if len(df) < pivot_len * 3:
        return []

    for i in range(pivot_len, len(df) - pivot_len):
        if highs[i] == highs[i-pivot_len:i+pivot_len+1].max():
            swing_highs.append({'idx': i, 'price': float(highs[i]), 'date': dates[i]})
        if lows[i] == lows[i-pivot_len:i+pivot_len+1].min():
            swing_lows.append({'idx': i, 'price': float(lows[i]), 'date': dates[i]})

    trend_lines = []

    if len(swing_highs) >= 2:
        for offset in range(0, min(3, len(swing_highs) - 1)):
            h1 = swing_highs[-(2 + offset)]
            h2 = swing_highs[-(1 + offset)]
            if abs(h2['idx'] - h1['idx']) >= 3:
                trend_lines.append({
                    'type': 'resistance',
                    'idx1': h1['idx'], 'y1': h1['price'], 'date1': h1['date'],
                    'idx2': h2['idx'], 'y2': h2['price'], 'date2': h2['date'],
                })
                break

    if len(swing_lows) >= 2:
        for offset in range(0, min(3, len(swing_lows) - 1)):
            l1 = swing_lows[-(2 + offset)]
            l2 = swing_lows[-(1 + offset)]
            if abs(l2['idx'] - l1['idx']) >= 3:
                trend_lines.append({
                    'type': 'support',
                    'idx1': l1['idx'], 'y1': l1['price'], 'date1': l1['date'],
                    'idx2': l2['idx'], 'y2': l2['price'], 'date2': l2['date'],
                })
                break

    return trend_lines

def calc_trend(df, sma_f, sma_s):
    if len(df) < 55:
        return "INSUFFICIENT DATA", "gray", 0
    latest = df.iloc[-1]
    score = 0
    if latest['Close'] > sma_f and sma_f > sma_s:
        score += 2
    elif latest['Close'] < sma_f and sma_f < sma_s:
        score -= 2
    if not pd.isna(latest['RSI']):
        if latest['RSI'] > 60: score += 1
        elif latest['RSI'] < 40: score -= 1
    if not pd.isna(latest['OrderFlow']):
        if latest['OrderFlow'] > 0: score += 1
        else: score -= 1
    if len(df) > 10:
        rc = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100
        if rc > 1: score += 1
        elif rc < -1: score -= 1
    if score >= 3: return "BULLISH", "#00ff88", score
    elif score <= -3: return "BEARISH", "#ff4444", score
    elif score >= 1: return "MILD BULLISH", "#88ff88", score
    elif score <= -1: return "MILD BEARISH", "#ff8888", score
    else: return "SIDEWAYS", "#ffaa00", score

# =========================================================================
# SIGNAL GENERATION (v14.1 style)
# =========================================================================

def generate_signals(df, bias_score, res_levels, sup_levels, bsl, ssl, bos_events, vol_avg_period=20, vol_mult=1.5, bias_filter=2, bos_threshold=4):
    """Generates BUY/SELL signals at each bar based on v14.1 logic."""
    signals = []

    if len(df) < 50:
        return signals

    df['VolAvg'] = df['Volume'].rolling(vol_avg_period).mean()
    df['VolConfirmed'] = df['Volume'] >= (df['VolAvg'] * vol_mult)

    recent_sweeps_sell = set()
    recent_sweeps_buy = set()

    for lvl in bsl:
        if lvl.get('swept') and 'sweep_bar' in lvl:
            recent_sweeps_sell.add(lvl['sweep_bar'])
    for lvl in ssl:
        if lvl.get('swept') and 'sweep_bar' in lvl:
            recent_sweeps_buy.add(lvl['sweep_bar'])

    bos_by_bar = {}
    for evt in bos_events:
        bos_by_bar.setdefault(evt['bar'], []).append(evt['type'])

    for i in range(50, len(df)):
        row = df.iloc[i]

        local_bias = 0
        if i >= 50:
            sma_f = df['SMA_Fast'].iloc[i] if not pd.isna(df['SMA_Fast'].iloc[i]) else row['Close']
            sma_s = df['SMA_Slow'].iloc[i] if not pd.isna(df['SMA_Slow'].iloc[i]) else row['Close']
            if row['Close'] > sma_f and sma_f > sma_s:
                local_bias += 2
            elif row['Close'] < sma_f and sma_f < sma_s:
                local_bias -= 2
            if not pd.isna(row['RSI']):
                if row['RSI'] > 60: local_bias += 1
                elif row['RSI'] < 40: local_bias -= 1
            if i >= 20:
                if row['Close'] > df['Close'].iloc[i-20]:
                    local_bias += 1
                else:
                    local_bias -= 1

        vol_ok = bool(row['VolConfirmed']) if not pd.isna(row['VolConfirmed']) else False

        sweep_buy = any((i - sb) <= 3 and (i - sb) >= 0 for sb in recent_sweeps_buy)
        sweep_sell = any((i - sb) <= 3 and (i - sb) >= 0 for sb in recent_sweeps_sell)

        bos_here = bos_by_bar.get(i, [])
        bos_up = 'BOS_UP' in bos_here
        bos_down = 'BOS_DOWN' in bos_here

        if (sweep_buy and local_bias >= bias_filter and vol_ok) or (bos_up and local_bias >= bos_threshold and vol_ok):
            signals.append({
                'bar': i,
                'date': row['Date'],
                'price': float(row['Low']) * 0.999,
                'type': 'BUY',
                'entry': float(row['Close']),
                'sl': float(row['Low']) * 0.995,
                't1': float(row['Close']) * 1.005,
                't2': float(row['Close']) * 1.010,
            })
        elif (sweep_sell and local_bias <= -bias_filter and vol_ok) or (bos_down and local_bias <= -bos_threshold and vol_ok):
            signals.append({
                'bar': i,
                'date': row['Date'],
                'price': float(row['High']) * 1.001,
                'type': 'SELL',
                'entry': float(row['Close']),
                'sl': float(row['High']) * 1.005,
                't1': float(row['Close']) * 0.995,
                't2': float(row['Close']) * 0.990,
            })

    return signals[-10:]

# =========================================================================
# FETCH WITH AUTO-CORRECTION
# =========================================================================
df = fetch_data(ticker_symbol, period, interval, st.session_state['refresh_counter'])

if df is None or len(df) < 20:
    retry_periods = {
        "1m":  ["5d", "1d"],
        "5m":  ["5d", "1mo"],
        "15m": ["5d", "1mo", "3mo"],
        "30m": ["5d", "1mo", "3mo"],
        "1h":  ["1mo", "5d", "3mo", "6mo"],
        "1d":  ["3mo", "6mo", "1y", "1mo"],
    }

    for alt_period in retry_periods.get(interval, ["1mo", "3mo"]):
        if alt_period == period:
            continue
        df = fetch_data(ticker_symbol, alt_period, interval, st.session_state['refresh_counter'])
        if df is not None and len(df) >= 20:
            st.warning(f"⚠️ Auto-adjusted period from `{period}` to `{alt_period}` for {interval} interval.")
            period = alt_period
            break

if df is None or len(df) < 20:
    fallback_intervals = ["1d", "1h", "15m", "5m"]
    for alt_interval in fallback_intervals:
        if alt_interval == interval:
            continue
        alt_period = {"1d": "6mo", "1h": "1mo", "15m": "5d", "5m": "5d"}.get(alt_interval, "1mo")
        df = fetch_data(ticker_symbol, alt_period, alt_interval, st.session_state['refresh_counter'])
        if df is not None and len(df) >= 20:
            st.warning(f"⚠️ Auto-switched from `{interval}` to `{alt_interval}` interval for this asset.")
            interval = alt_interval
            period = alt_period
            break

if df is None or len(df) < 20:
    st.error(f"❌ Could not fetch data for **{ticker_symbol}** at any timeframe.")
    st.info("""
    **Possible reasons:**
    - Market is closed (Indian market: 9:15 AM - 3:30 PM IST)
    - Symbol is invalid or delisted

    **Try:**
    - A different asset from the dropdown
    - `NIFTY 50` with `1h` interval
    - `RELIANCE` with `1d` interval
    """)
    st.stop()

# =========================================================================
# PROCESS INDICATORS
# =========================================================================
st.session_state['last_update_str'] = datetime.now().strftime('%H:%M:%S')

df['RSI'] = calc_rsi(df['Close'], rsi_period)
df['OrderFlow'] = calc_order_flow(df)
df['SMA_Fast'] = df['Close'].rolling(sma_fast).mean()
df['SMA_Slow'] = df['Close'].rolling(sma_slow).mean()

res_levels, sup_levels = find_sr(df, pivot_len)
bsl, ssl = find_smc(df, smc_pivot)
bos_events = find_bos(df, smc_pivot)
trend_lines = find_trend_lines(df, trend_pivot)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
pct_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0

sma_f = df['SMA_Fast'].iloc[-1]
sma_s = df['SMA_Slow'].iloc[-1]
trend_label, trend_color, trend_score = calc_trend(df, sma_f, sma_s)

# =========================================================================
# HTF DATA FETCH (1H + 1D)
# =========================================================================
@st.cache_data(ttl=60)
def fetch_htf_bias(symbol, tf):
    """Fetch HTF data from Upstox and compute simple bias."""
    try:
        from upstox_data import fetch_candles as _fc
        _days = 5 if tf == "1h" else 90
        htf_df = _fc(symbol=symbol, interval=tf, days=_days)
        if htf_df is None or len(htf_df) < 55:
            return "FLAT", 0, "WAIT"

        sma_f_h = htf_df['Close'].rolling(20).mean().iloc[-1]
        sma_s_h = htf_df['Close'].rolling(50).mean().iloc[-1]
        close_now_h = htf_df['Close'].iloc[-1]
        rsi_now_h = calc_rsi(htf_df['Close'], 14).iloc[-1]

        score = 0
        if close_now_h > sma_f_h and sma_f_h > sma_s_h:
            score += 2
        elif close_now_h < sma_f_h and sma_f_h < sma_s_h:
            score -= 2
        if not pd.isna(rsi_now_h):
            if rsi_now_h > 60:
                score += 1
            elif rsi_now_h < 40:
                score -= 1
        if len(htf_df) > 20:
            if close_now_h > htf_df['Close'].iloc[-20]:
                score += 1
            else:
                score -= 1

        if score >= 2:
            return "BULL", score, "BUY"
        elif score <= -2:
            return "BEAR", score, "SELL"
        else:
            return "FLAT", score, "WAIT"
    except Exception:
        return "FLAT", 0, "WAIT"

htf_1h_label, htf_1h_score, htf_1h_signal = fetch_htf_bias(ticker_symbol, "1h")
htf_1d_label, htf_1d_score, htf_1d_signal = fetch_htf_bias(ticker_symbol, "1d")

mtf_1m_label, mtf_1m_score, mtf_1m_signal = fetch_htf_bias(ticker_symbol, "1m")
mtf_5m_label, mtf_5m_score, mtf_5m_signal = fetch_htf_bias(ticker_symbol, "5m")
mtf_15m_label, mtf_15m_score, mtf_15m_signal = fetch_htf_bias(ticker_symbol, "15m")

# =========================================================================
# TOP METRICS
# =========================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{selected_label}", f"{float(latest['Close']):.2f}", f"{pct_change:+.2f}%")
c2.metric("RSI", f"{float(latest['RSI']):.1f}" if not pd.isna(latest['RSI']) else "—")
c3.metric("Order Flow", f"{float(latest['OrderFlow']):,.0f}" if not pd.isna(latest['OrderFlow']) else "—")
c4.metric("Nearest BSL", f"{bsl[-1]['price']:.2f}" if bsl else "—")
c5.metric("Nearest SSL", f"{ssl[-1]['price']:.2f}" if ssl else "—")

# =========================================================================
# HTF PANEL DISPLAY (1H + 1D)
# =========================================================================
htf_1h_color = "#00ff88" if htf_1h_label == "BULL" else "#ff4444" if htf_1h_label == "BEAR" else "#ffaa00"
htf_1d_color = "#00ff88" if htf_1d_label == "BULL" else "#ff4444" if htf_1d_label == "BEAR" else "#ffaa00"

st.markdown(
    f"""<div style="background:#131722;border:2px solid #2a2e39;border-radius:8px;padding:12px;margin:10px 0;">
    <div style="color:#aaa;font-size:12px;font-weight:bold;margin-bottom:8px;">📊 HIGHER TIMEFRAME BIAS</div>
    <table style="width:100%;border-collapse:collapse;color:#d1d4dc;font-size:13px;">
    <tr style="background:#2a2e39;">
        <th style="padding:6px;text-align:left;">TF</th>
        <th style="padding:6px;text-align:center;">BIAS</th>
        <th style="padding:6px;text-align:center;">SCORE</th>
        <th style="padding:6px;text-align:center;">SIGNAL</th>
    </tr>
    <tr>
        <td style="padding:6px;font-weight:bold;">1H</td>
        <td style="padding:6px;text-align:center;background:{htf_1h_color}33;color:{htf_1h_color};font-weight:bold;">{htf_1h_label}</td>
        <td style="padding:6px;text-align:center;">{htf_1h_score:+d}</td>
        <td style="padding:6px;text-align:center;background:{htf_1h_color}33;color:{htf_1h_color};font-weight:bold;">{htf_1h_signal}</td>
    </tr>
    <tr>
        <td style="padding:6px;font-weight:bold;">1D</td>
        <td style="padding:6px;text-align:center;background:{htf_1d_color}33;color:{htf_1d_color};font-weight:bold;">{htf_1d_label}</td>
        <td style="padding:6px;text-align:center;">{htf_1d_score:+d}</td>
        <td style="padding:6px;text-align:center;background:{htf_1d_color}33;color:{htf_1d_color};font-weight:bold;">{htf_1d_signal}</td>
    </tr>
    </table></div>""",
    unsafe_allow_html=True
)

# =========================================================================
# MTF PANEL DISPLAY (1m + 5m + 15m)
# =========================================================================
mtf_1m_color = "#00ff88" if mtf_1m_label == "BULL" else "#ff4444" if mtf_1m_label == "BEAR" else "#ffaa00"
mtf_5m_color = "#00ff88" if mtf_5m_label == "BULL" else "#ff4444" if mtf_5m_label == "BEAR" else "#ffaa00"
mtf_15m_color =
