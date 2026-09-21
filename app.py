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

st.set_page_config(page_title="Trading Dashboard", page_icon="📈", layout="wide")

# =========================================================================
# AUTO-REFRESH MECHANISM
# =========================================================================
# Increment a counter in session state every time refresh is triggered
if 'refresh_counter' not in st.session_state:
    st.session_state['refresh_counter'] = 0

def trigger_refresh():
    st.session_state['refresh_counter'] += 1

st.title("📈 Global Trading & Smart Money Dashboard")

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.header("⚙️ Settings")

# --- REFRESH CONTROLS ---
st.sidebar.markdown("### 🔄 Data Refresh")
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 30s)", value=False)
if st.sidebar.button("🔄 Manual Refresh Now", use_container_width=True):
    trigger_refresh()
    st.rerun()

# Show last update time
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = datetime.now()
    st.session_state['last_update_str'] = st.session_state['last_update'].strftime('%H:%M:%S')

st.sidebar.caption(f"Last update: **{st.session_state['last_update_str']}**")

# Auto-refresh mechanism
if auto_refresh:
    import time
    # Use a placeholder to trigger re-run after 30 seconds
    st.markdown(
        """
        <script>
        setTimeout(function() {
            window.location.reload();
        }, 30000);
        </script>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")

# --- ASSET DROPDOWN ---
st.sidebar.subheader("📊 Select Asset")

asset_options = {
    "--- INDIAN INDICES ---": None,
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "INDIA VIX": "^INDIAVIX",
    "--- INDIAN STOCKS ---": None,
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "--- US STOCKS ---": None,
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Meta": "META",
    "--- GLOBAL INDICES ---": None,
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Nikkei 225": "^N225",
    "FTSE 100": "^FTSE",
    "DAX": "^GDAXI",
    "Hang Seng": "^HSI",
    "--- COMMODITIES ---": None,
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "Natural Gas": "NG=F",
    "Copper": "HG=F",
    "--- FOREX ---": None,
    "USD/INR": "USDINR=X",
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "--- CRYPTO ---": None,
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
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
    "5m":  ["1d", "5d", "1mo"],
    "15m": ["1d", "5d", "1mo", "3mo"],
    "30m": ["1d", "5d", "1mo", "3mo"],
    "1h":  ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
}
available_periods = interval_limits.get(interval, ["1mo", "3mo", "6mo"])
period = st.sidebar.selectbox("Period", available_periods, index=min(2, len(available_periods)-1))
st.sidebar.caption(f"⚠️ Max history for {interval}: {available_periods[-1]}")

show_last_n = st.sidebar.slider("Bars to show (zoom)", 30, 500, 150, step=10)

# --- Indicators ---
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Indicators")
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
sma_fast = st.sidebar.slider("SMA Fast", 5, 50, 20)
sma_slow = st.sidebar.slider("SMA Slow", 20, 200, 50)
pivot_len = st.sidebar.slider("S/R Pivot Length", 3, 30, 10)
smc_pivot = st.sidebar.slider("SMC Pivot Length", 3, 30, 5)

# --- Overlays ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Overlays")
show_sr = st.sidebar.checkbox("Support / Resistance", value=True)
show_smc = st.sidebar.checkbox("SMC Liquidity (BSL/SSL)", value=True)
show_bos = st.sidebar.checkbox("BOS / CHoCH", value=True)
show_sma = st.sidebar.checkbox("Moving Averages", value=True)

# =========================================================================
# DATA FETCH (No cache — manual refresh controls it)
# =========================================================================
def fetch_data(symbol, period, interval, refresh_key):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        if 'Datetime' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        required = ['Date', 'Open', 'High', 'Low', 'Close']
        for col in required:
            if col not in df.columns:
                return None
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        return df
    except Exception as e:
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
# FETCH AND PROCESS
# =========================================================================
df = fetch_data(ticker_symbol, period, interval, st.session_state['refresh_counter'])
if df is None or len(df) < 20:
    st.error(f"⚠️ Could not fetch data for **{ticker_symbol}** at {interval}/{period}.")
    st.info("Try a different interval/period. For NIFTY 50 use `1d` or `1h`.")
    st.stop()

# Update timestamp
st.session_state['last_update_str'] = datetime.now().strftime('%H:%M:%S')

df['RSI'] = calc_rsi(df['Close'], rsi_period)
df['OrderFlow'] = calc_order_flow(df)
df['SMA_Fast'] = df['Close'].rolling(sma_fast).mean()
df['SMA_Slow'] = df['Close'].rolling(sma_slow).mean()

res_levels, sup_levels = find_sr(df, pivot_len)
bsl, ssl = find_smc(df, smc_pivot)
bos_events = find_bos(df, smc_pivot)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
pct_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] != 0 else 0

sma_f = df['SMA_Fast'].iloc[-1]
sma_s = df['SMA_Slow'].iloc[-1]
trend_label, trend_color, trend_score = calc_trend(df, sma_f, sma_s)

# =========================================================================
# TOP METRICS
# =========================================================================
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"{selected_label}", f"{float(latest['Close']):.2f}", f"{pct_change:+.2f}%")
c2.metric("RSI", f"{float(latest['RSI']):.1f}" if not pd.isna(latest['RSI']) else "—")
c3.metric("Order Flow", f"{float(latest['OrderFlow']):,.0f}" if not pd.isna(latest['OrderFlow']) else "—")
c4.metric("Nearest BSL", f"{bsl[-1]['price']:.2f}" if bsl else "—")
c5.metric("Nearest SSL", f"{ssl[-1]['price']:.2f}" if ssl else "—")

st.markdown(
    f"""<div style="background:{trend_color}22;border-left:6px solid {trend_color};
    padding:12px 20px;border-radius:8px;margin:12px 0;">
    <span style="color:{trend_color};font-size:20px;font-weight:bold;">
    📊 MARKET TREND: {trend_label}
    </span>
    <span style="color:#aaa;font-size:14px;margin-left:20px;">
    Score: {trend_score:+d} · SMA20: {sma_f:.2f} · SMA50: {sma_s:.2f}
    </span></div>""",
    unsafe_allow_html=True
)

df_display = df.tail(show_last_n).copy()
bars_offset = len(df) - len(df_display)
x_start = df_display['Date'].iloc[0]
x_end = df_display['Date'].iloc[-1]

# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3 = st.tabs(["📊 Chart", "💧 SMC Liquidity", "🏦 Smart Money"])

# -------------------------------------------------------------------------
# TAB 1: CHART
# -------------------------------------------------------------------------
with tab1:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.20, 0.15],
                        vertical_spacing=0.02,
                        subplot_titles=("", "RSI", "Order Flow"))
    
    fig.add_trace(go.Candlestick(
        x=df_display['Date'], open=df_display['Open'], high=df_display['High'],
        low=df_display['Low'], close=df_display['Close'], name="Price",
        increasing_line_color='#00ff88', decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff88', decreasing_fillcolor='#ff4444',
        line=dict(width=1)
    ), row=1, col=1)
    
    if show_sma:
        fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['SMA_Fast'],
                                 name=f"SMA{sma_fast}", line=dict(color='#00aaff', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['SMA_Slow'],
                                 name=f"SMA{sma_slow}", line=dict(color='#ff8800', width=1.5)), row=1, col=1)
    
    if show_sr:
        for lvl in res_levels[-5:]:
            fig.add_shape(type='line', xref='x', yref='y',
                          x0=x_start, x1=x_end, y0=lvl['price'], y1=lvl['price'],
                          line=dict(color='#ff4444', width=1.5, dash='dash'))
            fig.add_annotation(x=x_end, y=lvl['price'], xref='x', yref='y',
                               text=f"R {lvl['price']:.2f}", showarrow=False, xanchor='left',
                               font=dict(color='#ff4444', size=10), bgcolor='#000000')
        for lvl in sup_levels[-5:]:
            fig.add_shape(type='line', xref='x', yref='y',
                          x0=x_start, x1=x_end, y0=lvl['price'], y1=lvl['price'],
                          line=dict(color='#00ff88', width=1.5, dash='dash'))
            fig.add_annotation(x=x_end, y=lvl['price'], xref='x', yref='y',
                               text=f"S {lvl['price']:.2f}", showarrow=False, xanchor='left',
                               font=dict(color='#00ff88', size=10), bgcolor='#000000')
    
    if show_smc:
        for lvl in bsl[-5:]:
            color = '#ffaa00' if lvl['eq'] else '#ff6666'
            dash = 'dot' if lvl['swept'] else 'dash'
            fig.add_shape(type='line', xref='x', yref='y',
                          x0=x_start, x1=x_end, y0=lvl['price'], y1=lvl['price'],
                          line=dict(color=color, width=1, dash=dash))
        for lvl in ssl[-5:]:
            color = '#ffaa00' if lvl['eq'] else '#66ff66'
            dash = 'dot' if lvl['swept'] else 'dash'
            fig.add_shape(type='line', xref='x', yref='y',
                          x0=x_start, x1=x_end, y0=lvl['price'], y1=lvl['price'],
                          line=dict(color=color, width=1, dash=dash))
    
    if show_bos:
        for evt in bos_events:
            if evt['bar'] >= bars_offset:
                idx = evt['bar'] - bars_offset
                if idx < len(df_display):
                    color = '#00ff88' if evt['type'] == 'BOS_UP' else '#ff4444'
                    symbol = 'triangle-up' if evt['type'] == 'BOS_UP' else 'triangle-down'
                    fig.add_trace(go.Scatter(x=[df_display['Date'].iloc[idx]], y=[evt['price']],
                                             mode='markers', marker=dict(color=color, size=10, symbol=symbol),
                                             showlegend=False), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df_display['Date'], y=df_display['RSI'],
                             name="RSI", line=dict(color='#ffaa00', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='#ff4444', width=1, dash='dot'), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='#00ff88', width=1, dash='dot'), row=2, col=1)
    fig.add_hline(y=50, line=dict(color='#666', width=1), row=2, col=1)
    
    of_colors = ['#ff4444' if v < 0 else '#00ff88' for v in df_display['OrderFlow'].fillna(0)]
    fig.add_trace(go.Bar(x=df_display['Date'], y=df_display['OrderFlow'],
                         name="Order Flow", marker_color=of_colors), row=3, col=1)
    
    fig.update_layout(height=800, template="plotly_dark",
                      paper_bgcolor='#131722', plot_bgcolor='#131722',
                      font=dict(color='#d1d4dc', size=11),
                      xaxis_rangeslider_visible=False, hovermode='x unified',
                      margin=dict(l=10, r=80, t=30, b=30),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor='#2a2e39', showgrid=True, zeroline=False)
    fig.update_yaxes(gridcolor='#2a2e39', showgrid=True, zeroline=False, side='right')
    
    st.plotly_chart(fig, use_container_width=True, config={
        'scrollZoom': True, 'displayModeBar': True, 'displaylogo': False
    })

# -------------------------------------------------------------------------
# TAB 2: SMC
# -------------------------------------------------------------------------
with tab2:
    st.subheader("💧 SMC Liquidity Zones")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 🔴 Buy-Side Liquidity (Above)")
        if bsl:
            bsl_df = pd.DataFrame(bsl[-10:])[['price', 'eq', 'swept']]
            bsl_df['distance_%'] = ((bsl_df['price'] - float(latest['Close'])) / float(latest['Close']) * 100).round(2)
            st.dataframe(bsl_df, use_container_width=True, hide_index=True)
        else:
            st.info("No BSL levels")
    with col_b:
        st.markdown("### 🟢 Sell-Side Liquidity (Below)")
        if ssl:
            ssl_df = pd.DataFrame(ssl[-10:])[['price', 'eq', 'swept']]
            ssl_df['distance_%'] = ((float(latest['Close']) - ssl_df['price']) / float(latest['Close']) * 100).round(2)
            st.dataframe(ssl_df, use_container_width=True, hide_index=True)
        else:
            st.info("No SSL levels")

# -------------------------------------------------------------------------
# TAB 3: SMART MONEY
# -------------------------------------------------------------------------
with tab3:
    st.subheader("🏦 Smart Money Positioning")
    if not NSELIB_AVAILABLE:
        st.error("nselib not installed")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### FII / DII Activity")
            if st.button("🔄 Fetch FII/DII", key="fii"):
                try:
                    fii_df = capital_market.fii_dii_trading_activity()
                    if fii_df is not None and not fii_df.empty:
                        st.dataframe(fii_df, use_container_width=True, height=300)
                    else:
                        st.warning("No data returned")
                except Exception as e:
                    st.error(f"Error: {e}")
        with col_b:
            st.markdown("### Participant-wise OI")
            date_input = st.date_input("Date", value=datetime.now())
            if st.button("🔄 Fetch Participant OI", key="poi"):
                date_str = date_input.strftime('%d-%m-%Y')
                try:
                    poi_df = derivatives.participant_wise_open_interest(trade_date=date_str)
                    if poi_df is not None and not poi_df.empty:
                        st.dataframe(poi_df, use_container_width=True, height=300)
                    else:
                        st.warning(f"No data for {date_str}. Try earlier date.")
                except Exception as e:
                    st.error(f"Error: {e}")

# =========================================================================
# FOOTER
# =========================================================================
st.markdown("---")
st.caption(f"Last update: {st.session_state['last_update_str']} | Data by Yahoo Finance")
