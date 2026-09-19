import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import yfinance as yf

st.set_page_config(page_title="Trading Dashboard", page_icon="📈", layout="wide")
st.title("📈 Institutional Flow & SMC Dashboard")
st.caption("Live NSE data | RSI | Option Chain | SMC Liquidity | Order Flow")

st.sidebar.header("Settings")
symbol = st.sidebar.text_input("NSE Symbol", value="RELIANCE").upper()
period = st.sidebar.selectbox("Data Period", ["1mo", "3mo", "6mo", "1y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "15m"], index=0)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
pivot_len = st.sidebar.slider("SMC Pivot Length", 5, 30, 10)
liq_tolerance = st.sidebar.slider("EQH/EQL Tolerance %", 0.01, 0.50, 0.08, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("Option Chain (Manual)")
call_oi_strike = st.sidebar.number_input("Call OI Wall", value=0.0, step=50.0)
put_oi_strike = st.sidebar.number_input("Put OI Wall", value=0.0, step=50.0)
max_pain = st.sidebar.number_input("Max Pain", value=0.0, step=50.0)
total_call_oi = st.sidebar.number_input("Total Call OI (Lakh)", value=0.0, step=1.0)
total_put_oi = st.sidebar.number_input("Total Put OI (Lakh)", value=0.0, step=1.0)

@st.cache_data(ttl=300)
def fetch_data(symbol, period, interval):
    ticker = f"{symbol}.NS"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.reset_index()
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_order_flow(df):
    bar_range = df['High'] - df['Low']
    bar_body = (df['Close'] - df['Open']).abs()
    body_ratio = np.where(bar_range > 0, bar_body / bar_range, 0)
    raw_delta = np.where(df['Close'] > df['Open'], df['Volume'] * body_ratio,
                np.where(df['Close'] < df['Open'], -df['Volume'] * body_ratio, 0))
    return pd.Series(raw_delta, index=df.index).rolling(3).mean()

def find_liquidity_levels(df, pivot_len=10, tolerance=0.08):
    highs = df['High'].values
    lows = df['Low'].values
    bsl, ssl = [], []
    for i in range(pivot_len, len(df) - pivot_len):
        if highs[i] == highs[i-pivot_len:i+pivot_len+1].max():
            bsl.append({'price': float(highs[i]), 'type': 'BSL', 'eq': False})
        if lows[i] == lows[i-pivot_len:i+pivot_len+1].min():
            ssl.append({'price': float(lows[i]), 'type': 'SSL', 'eq': False})
    for i, lvl in enumerate(bsl):
        for j in range(i+1, len(bsl)):
            if abs(lvl['price'] - bsl[j]['price']) / lvl['price'] * 100 <= tolerance:
                bsl[j]['eq'] = True
    for i, lvl in enumerate(ssl):
        for j in range(i+1, len(ssl)):
            if abs(lvl['price'] - ssl[j]['price']) / ssl[j]['price'] * 100 <= tolerance:
                ssl[j]['eq'] = True
    return bsl, ssl

df = fetch_data(symbol, period, interval)
if df is None or df.empty:
    st.error(f"Could not fetch data for {symbol}. Try another symbol.")
    st.stop()

df['RSI'] = calculate_rsi(df['Close'], rsi_period)
df['OrderFlow'] = calculate_order_flow(df)
df['SMA20'] = df['Close'].rolling(20).mean()
bsl, ssl = find_liquidity_levels(df, pivot_len, liq_tolerance)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100 if prev['Close'] != 0 else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(f"{symbol}", f"₹{float(latest['Close']):.2f}", f"{pct_change:+.2f}%")
col2.metric("RSI", f"{float(latest['RSI']):.2f}" if not pd.isna(latest['RSI']) else "N/A")
col3.metric("Order Flow", f"{float(latest['OrderFlow']):.0f}" if not pd.isna(latest['OrderFlow']) else "N/A")
col4.metric("Nearest BSL", f"₹{bsl[-1]['price']:.2f}" if bsl else "N/A")
col5.metric("Nearest SSL", f"₹{ssl[-1]['price']:.2f}" if ssl else "N/A")

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Chart", "💧 SMC Liquidity", "📉 Option Chain", "📋 Data Table"])

with tab1:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03, subplot_titles=("Price", "RSI", "Order Flow"))
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price",
                                 increasing_line_color='yellow', decreasing_line_color='red'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], name="SMA20",
                             line=dict(color='cyan', width=1)), row=1, col=1)
    if call_oi_strike > 0:
        fig.add_hline(y=call_oi_strike, line=dict(color='red', width=2), row=1, col=1)
    if put_oi_strike > 0:
        fig.add_hline(y=put_oi_strike, line=dict(color='green', width=2), row=1, col=1)
    if max_pain > 0:
        fig.add_hline(y=max_pain, line=dict(color='orange', width=2, dash='dot'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI",
                             line=dict(color='yellow', width=2)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash', width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash', width=1), row=2, col=1)
    of_colors = ['red' if v < 0 else 'yellow' for v in df['OrderFlow'].fillna(0)]
    fig.add_trace(go.Bar(x=df['Date'], y=df['OrderFlow'], name="Order Flow",
                         marker_color=of_colors), row=3, col=1)
    fig.update_layout(height=800, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("SMC Liquidity Zones")
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name="Price",
                                  increasing_line_color='yellow', decreasing_line_color='red'))
    for lvl in bsl[-8:]:
        color = 'orange' if lvl.get('eq') else 'red'
        fig2.add_hline(y=lvl['price'], line=dict(color=color, width=1, dash='dash'))
    for lvl in ssl[-8:]:
        color = 'orange' if lvl.get('eq') else 'green'
        fig2.add_hline(y=lvl['price'], line=dict(color=color, width=1, dash='dash'))
    fig2.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🔴 Buy-Side Liquidity (Above)**")
        if bsl:
            st.dataframe(pd.DataFrame(bsl[-8:])[['price', 'type']], use_container_width=True, hide_index=True)
        else:
            st.info("No BSL levels")
    with col_b:
        st.markdown("**🟢 Sell-Side Liquidity (Below)**")
        if ssl:
            st.dataframe(pd.DataFrame(ssl[-8:])[['price', 'type']], use_container_width=True, hide_index=True)
        else:
            st.info("No SSL levels")

with tab3:
    st.subheader("Option Chain Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric("Call OI Wall", f"₹{call_oi_strike:.2f}" if call_oi_strike > 0 else "Not set")
    col2.metric("Put OI Wall", f"₹{put_oi_strike:.2f}" if put_oi_strike > 0 else "Not set")
    col3.metric("Max Pain", f"₹{max_pain:.2f}" if max_pain > 0 else "Not set")
    if total_call_oi > 0 and total_put_oi > 0:
        pcr = total_put_oi / total_call_oi
        st.metric("PCR", f"{pcr:.3f}")
        if pcr > 1.0:
            st.success(f"PCR {pcr:.3f} → BULLISH bias")
        elif pcr < 0.8:
            st.error(f"PCR {pcr:.3f} → BEARISH bias")
        else:
            st.info(f"PCR {pcr:.3f} → NEUTRAL")
    else:
        st.info("Enter Total Call OI and Put OI in sidebar")

with tab4:
    st.subheader("Data Table")
    display_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'OrderFlow']].tail(50).round(2)
    st.dataframe(display_df, use_container_width=True, height=600)
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"{symbol}_data.csv", "text/csv")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
