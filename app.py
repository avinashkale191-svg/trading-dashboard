import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from nselib import derivatives, capital_market

st.set_page_config(page_title="Global Trading Dashboard", page_icon="🌍", layout="wide")
st.title("🌍 Global Institutional Flow & SMC Dashboard")
st.caption("S/R | SMC Liquidity | Order Blocks | BOS | RSI | Order Flow | Option Chain | Smart Money")

st.sidebar.header("Settings")

# --- ASSET DROPDOWN ---
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

st.sidebar.markdown("**Or type any ticker manually:**")
manual_ticker = st.sidebar.text_input("Custom Ticker (overrides dropdown)", value="")
if manual_ticker.strip() != "":
    ticker_symbol = manual_ticker.strip()
    selected_label = ticker_symbol

st.sidebar.caption("Examples: AAPL, ^GSPC, GC=F, BTC-USD, RELIANCE.NS")

# --- TIMEFRAME (FIXED) ---
interval = st.sidebar.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)

interval_limits = {
    "5m": ["1d", "5d", "1mo"],
    "15m": ["1d", "5d", "1mo", "3mo"],
    "30m": ["1d", "5d", "1mo", "3mo"],
    "1h": ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"],
}
available_periods = interval_limits.get(interval, ["1mo", "3mo", "6mo", "1y"])
default_idx = min(2, len(available_periods) - 1)
period = st.sidebar.selectbox("Data Period", available_periods, index=default_idx)

st.sidebar.caption(f"⚠️ {interval} max history: {available_periods[-1]}")

rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
pivot_len = st.sidebar.slider("S/R Pivot Length", 3, 30, 10)
smc_pivot = st.sidebar.slider("SMC Pivot Length (Liquidity)", 3, 30, 5)
liq_tolerance = st.sidebar.slider("EQH/EQL Tolerance %", 0.01, 0.50, 0.08, step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("Chart Overlays")
show_sr = st.sidebar.checkbox("Show Support/Resistance", value=True)
show_bsl_ssl = st.sidebar.checkbox("Show SMC Liquidity (BSL/SSL)", value=True)
show_ob = st.sidebar.checkbox("Show Order Blocks", value=True)
show_bos = st.sidebar.checkbox("Show BOS / CHoCH", value=True)
show_sweeps = st.sidebar.checkbox("Show Liquidity Sweeps", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Option Chain (Manual)")
call_oi_strike = st.sidebar.number_input("Call OI Wall", value=0.0, step=50.0)
put_oi_strike = st.sidebar.number_input("Put OI Wall", value=0.0, step=50.0)
max_pain = st.sidebar.number_input("Max Pain", value=0.0, step=50.0)
total_call_oi = st.sidebar.number_input("Total Call OI", value=0.0, step=1.0)
total_put_oi = st.sidebar.number_input("Total Put OI", value=0.0, step=1.0)

# =========================================================================
# DATA FETCH
# =========================================================================
@st.cache_data(ttl=300)
def fetch_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.reset_index()
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

# =========================================================================
# INDICATORS
# =========================================================================
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

def find_pivot_levels(df, pivot_len):
    highs = df['High'].values
    lows = df['Low'].values
    res = []
    sup = []
    for i in range(pivot_len, len(df) - pivot_len):
        if highs[i] == highs[i-pivot_len:i+pivot_len+1].max():
            res.append({'price': float(highs[i]), 'bar': i})
        if lows[i] == lows[i-pivot_len:i+pivot_len+1].min():
            sup.append({'price': float(lows[i]), 'bar': i})
    return res, sup

def find_smc_liquidity(df, pivot_len=5, tolerance=0.08):
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
            if abs(lvl['price'] - bsl[j]['price']) / lvl['price'] * 100 <= tolerance:
                bsl[j]['eq'] = True
    for i, lvl in enumerate(ssl):
        for j in range(i+1, len(ssl)):
            if abs(lvl['price'] - ssl[j]['price']) / ssl[j]['price'] * 100 <= tolerance:
                ssl[j]['eq'] = True
    
    for lvl in bsl:
        bar = lvl['bar']
        for k in range(bar + 1, len(df)):
            if df['High'].iloc[k] > lvl['price'] and df['Close'].iloc[k] < lvl['price']:
                lvl['swept'] = True
                lvl['sweep_bar'] = k
                break
    
    for lvl in ssl:
        bar = lvl['bar']
        for k in range(bar + 1, len(df)):
            if df['Low'].iloc[k] < lvl['price'] and df['Close'].iloc[k] > lvl['price']:
                lvl['swept'] = True
                lvl['sweep_bar'] = k
                break
    
    return bsl, ssl

def find_order_blocks(df, lookback=50):
    obs = []
    start = max(1, len(df) - lookback)
    for i in range(start, len(df) - 2):
        cur = df.iloc[i]
        nxt = df.iloc[i + 1]
        nxt2 = df.iloc[i + 2]
        
        if cur['Close'] < cur['Open']:
            move = nxt2['Close'] - nxt['Open']
            if move > (cur['Open'] - cur['Close']) * 1.5:
                obs.append({
                    'type': 'bullish',
                    'top': float(cur['High']),
                    'bottom': float(cur['Low']),
                    'bar': i
                })
        
        if cur['Close'] > cur['Open']:
            move = nxt['Open'] - nxt2['Close']
            if move > (cur['Close'] - cur['Open']) * 1.5:
                obs.append({
                    'type': 'bearish',
                    'top': float(cur['High']),
                    'bottom': float(cur['Low']),
                    'bar': i
                })
    return obs[-10:]

def find_bos_choch(df, pivot_len=5):
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
            events.append({'type': 'BOS_UP', 'bar': i, 'price': last_high['price']})
            last_high = None
        
        if last_low and closes[i] < last_low['price']:
            events.append({'type': 'BOS_DOWN', 'bar': i, 'price': last_low['price']})
            last_low = None
    
    return events[-15:]

# =========================================================================
# OPTION CHAIN ENGINE
# =========================================================================
@st.cache_data(ttl=300)
def fetch_option_chain(index_name):
    try:
        from nsefin import NSEClient
        nse = NSEClient()
        oc = nse.get_option_chain(index_name)
        return oc
    except Exception as e:
        return None

def analyze_option_chain(oc, spot_price):
    if oc is None or oc.empty:
        return None
    
    results = {}
    
    try:
        oc.columns = [c.strip().lower().replace(' ', '_') for c in oc.columns]
        
        strike_col = None
        ce_vol_col = None
        pe_vol_col = None
        ce_oi_col = None
        pe_oi_col = None
        
        for col in oc.columns:
            if 'strike' in col:
                strike_col = col
            if 'ce' in col and 'volume' in col:
                ce_vol_col = col
            if 'pe' in col and 'volume' in col:
                pe_vol_col = col
            if 'ce' in col and ('oi' in col or 'open_interest' in col):
                ce_oi_col = col
            if 'pe' in col and ('oi' in col or 'open_interest' in col):
                pe_oi_col = col
        
        if not strike_col or not ce_vol_col or not pe_vol_col:
            return None
        
        oc[strike_col] = pd.to_numeric(oc[strike_col], errors='coerce')
        oc[ce_vol_col] = pd.to_numeric(oc[ce_vol_col], errors='coerce').fillna(0)
        oc[pe_vol_col] = pd.to_numeric(oc[pe_vol_col], errors='coerce').fillna(0)
        
        oc = oc.dropna(subset=[strike_col])
        
        total_ce_vol = oc[ce_vol_col].sum()
        total_pe_vol = oc[pe_vol_col].sum()
        
        total_ce_oi = pd.to_numeric(oc[ce_oi_col], errors='coerce').fillna(0).sum() if ce_oi_col else 0
        total_pe_oi = pd.to_numeric(oc[pe_oi_col], errors='coerce').fillna(0).sum() if pe_oi_col else 0
        
        pcr_volume = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
        pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        max_ce_vol_row = oc.loc[oc[ce_vol_col].idxmax()]
        max_pe_vol_row = oc.loc[oc[pe_vol_col].idxmax()]
        
        call_wall_strike = max_ce_vol_row[strike_col]
        put_wall_strike = max_pe_vol_row[strike_col]
        call_wall_vol = max_ce_vol_row[ce_vol_col]
        put_wall_vol = max_pe_vol_row[pe_vol_col]
        
        oc['vol_ratio'] = oc[ce_vol_col] / (oc[pe_vol_col] + 1)
        oc['vol_shift'] = oc['vol_ratio'] - 1
        
        top_call_strikes = oc.nlargest(3, ce_vol_col)[[strike_col, ce_vol_col, pe_vol_col]]
        top_put_strikes = oc.nlargest(3, pe_vol_col)[[strike_col, ce_vol_col, pe_vol_col]]
        
        oc['total_value'] = oc[ce_vol_col] + oc[pe_vol_col]
        max_pain_strike = oc.loc[oc['total_value'].idxmin(), strike_col]
        
        if pcr_volume > 1.2:
            trend = "BULLISH"
            trend_reason = f"PCR {pcr_volume:.2f} > 1.2 → More Put volume than Call → Bullish"
        elif pcr_volume < 0.7:
            trend = "BEARISH"
            trend_reason = f"PCR {pcr_volume:.2f} < 0.7 → More Call volume than Put → Bearish"
        else:
            trend = "NEUTRAL"
            trend_reason = f"PCR {pcr_volume:.2f} → Balanced volume → Range-bound"
        
        results = {
            'strike_col': strike_col,
            'ce_vol_col': ce_vol_col,
            'pe_vol_col': pe_vol_col,
            'total_ce_vol': total_ce_vol,
            'total_pe_vol': total_pe_vol,
            'pcr_volume': pcr_volume,
            'pcr_oi': pcr_oi,
            'call_wall_strike': call_wall_strike,
            'put_wall_strike': put_wall_strike,
            'call_wall_vol': call_wall_vol,
            'put_wall_vol': put_wall_vol,
            'max_pain_strike': max_pain_strike,
            'trend': trend,
            'trend_reason': trend_reason,
            'oc_df': oc,
            'top_call_strikes': top_call_strikes,
            'top_put_strikes': top_put_strikes,
        }
        
        return results
    except Exception as e:
        return None

# =========================================================================
# SMART MONEY ENGINE (NEW)
# =========================================================================
@st.cache_data(ttl=300)
def fetch_participant_oi(trade_date_str):
    """Fetch participant-wise open interest for a given date."""
    try:
        df = derivatives.participant_wise_open_interest(trade_date=trade_date_str)
        return df
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def fetch_fii_dii_activity():
    """Fetch latest FII/DII trading activity."""
    try:
        df = capital_market.fii_dii_trading_activity()
        return df
    except Exception as e:
        return None

# =========================================================================
# FETCH AND CALCULATE
# =========================================================================
df = fetch_data(ticker_symbol, period, interval)
if df is None or df.empty:
    st.error(f"Could not fetch data for '{ticker_symbol}'. Try another asset or shorter period.")
    st.stop()

df['RSI'] = calculate_rsi(df['Close'], rsi_period)
df['OrderFlow'] = calculate_order_flow(df)
df['SMA20'] = df['Close'].rolling(20).mean()

res_levels, sup_levels = find_pivot_levels(df, pivot_len)
bsl, ssl = find_smc_liquidity(df, smc_pivot, liq_tolerance)
order_blocks = find_order_blocks(df, lookback=min(100, len(df)-3))
bos_events = find_bos_choch(df, smc_pivot)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
pct_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100 if prev['Close'] != 0 else 0

# =========================================================================
# METRICS
# =========================================================================
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(f"{selected_label}", f"{float(latest['Close']):.2f}", f"{pct_change:+.2f}%")
col2.metric("RSI", f"{float(latest['RSI']):.2f}" if not pd.isna(latest['RSI']) else "N/A")
col3.metric("Order Flow", f"{float(latest['OrderFlow']):.0f}" if not pd.isna(latest['OrderFlow']) else "N/A")
col4.metric("Nearest BSL", f"{bsl[-1]['price']:.2f}" if bsl else "N/A")
col5.metric("Nearest SSL", f"{ssl[-1]['price']:.2f}" if ssl else "N/A")

st.markdown("---")

# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Main Chart", "💧 SMC Liquidity", "📦 Order Blocks + BOS",
    "📉 Option Chain", "📊 Volume Shift", "🏦 Smart Money", "📋 Data Table"
])

# -------------------------------------------------------------------------
# TAB 1: MAIN CHART
# -------------------------------------------------------------------------
with tab1:
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03, subplot_titles=("Price + S/R", "RSI", "Order Flow"))
    
    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name="Price",
                                 increasing_line_color='yellow', decreasing_line_color='red'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], name="SMA20",
                             line=dict(color='cyan', width=1)), row=1, col=1)
    
    if show_sr:
        for lvl in res_levels[-5:]:
            fig.add_hline(y=lvl['price'], line=dict(color='red', width=1, dash='dash'), row=1, col=1)
        for lvl in sup_levels[-5:]:
            fig.add_hline(y=lvl['price'], line=dict(color='green', width=1, dash='dash'), row=1, col=1)
    
    if call_oi_strike > 0:
        fig.add_hline(y=call_oi_strike, line=dict(color='darkred', width=2), row=1, col=1)
    if put_oi_strike > 0:
        fig.add_hline(y=put_oi_strike, line=dict(color='darkgreen', width=2), row=1, col=1)
    if max_pain > 0:
        fig.add_hline(y=max_pain, line=dict(color='orange', width=2, dash='dot'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI",
                             line=dict(color='yellow', width=2)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color='red', dash='dash', width=1), row=2, col=1)
    fig.add_hline(y=30, line=dict(color='green', dash='dash', width=1), row=2, col=1)
    
    of_colors = ['red' if v < 0 else 'yellow' for v in df['OrderFlow'].fillna(0)]
    fig.add_trace(go.Bar(x=df['Date'], y=df['OrderFlow'], name="Order Flow",
                         marker_color=of_colors), row=3, col=1)
    
    fig.update_layout(height=850, template="plotly_dark", xaxis_rangeslider_visible=False,
                      margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------------
# TAB 2: SMC LIQUIDITY
# -------------------------------------------------------------------------
with tab2:
    st.subheader("SMC Liquidity Zones")
    st.caption("BSL = Buy-side liquidity (above swings) | SSL = Sell-side liquidity (below swings) | X = Swept")
    
    fig2 = go.Figure()
    fig2.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name="Price",
                                  increasing_line_color='yellow', decreasing_line_color='red'))
    
    if show_bsl_ssl:
        for lvl in bsl[-10:]:
            color = 'orange' if lvl['eq'] else ('gray' if lvl['swept'] else 'red')
            dash = 'dot' if lvl['swept'] else 'dash'
            fig2.add_hline(y=lvl['price'], line=dict(color=color, width=1, dash=dash))
        
        for lvl in ssl[-10:]:
            color = 'orange' if lvl['eq'] else ('gray' if lvl['swept'] else 'green')
            dash = 'dot' if lvl['swept'] else 'dash'
            fig2.add_hline(y=lvl['price'], line=dict(color=color, width=1, dash=dash))
    
    if show_sweeps:
        for lvl in bsl:
            if lvl['swept'] and 'sweep_bar' in lvl:
                fig2.add_trace(go.Scatter(
                    x=[df['Date'].iloc[lvl['sweep_bar']]],
                    y=[df['High'].iloc[lvl['sweep_bar']]],
                    mode='markers', marker=dict(color='red', size=15, symbol='x'),
                    name='BSL Sweep', showlegend=False
                ))
        for lvl in ssl:
            if lvl['swept'] and 'sweep_bar' in lvl:
                fig2.add_trace(go.Scatter(
                    x=[df['Date'].iloc[lvl['sweep_bar']]],
                    y=[df['Low'].iloc[lvl['sweep_bar']]],
                    mode='markers', marker=dict(color='green', size=15, symbol='x'),
                    name='SSL Sweep', showlegend=False
                ))
    
    fig2.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🔴 Buy-Side Liquidity (Above)**")
        if bsl:
            bsl_df = pd.DataFrame(bsl[-10:])[['price', 'eq', 'swept']]
            bsl_df['distance_%'] = ((bsl_df['price'] - float(latest['Close'])) / float(latest['Close']) * 100).round(2)
            st.dataframe(bsl_df, use_container_width=True, hide_index=True)
        else:
            st.info("No BSL levels")
    with col_b:
        st.markdown("**🟢 Sell-Side Liquidity (Below)**")
        if ssl:
            ssl_df = pd.DataFrame(ssl[-10:])[['price', 'eq', 'swept']]
            ssl_df['distance_%'] = ((float(latest['Close']) - ssl_df['price']) / float(latest['Close']) * 100).round(2)
            st.dataframe(ssl_df, use_container_width=True, hide_index=True)
        else:
            st.info("No SSL levels")

# -------------------------------------------------------------------------
# TAB 3: ORDER BLOCKS + BOS
# -------------------------------------------------------------------------
with tab3:
    st.subheader("Order Blocks & Break of Structure")
    
    fig3 = go.Figure()
    fig3.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                  low=df['Low'], close=df['Close'], name="Price",
                                  increasing_line_color='yellow', decreasing_line_color='red'))
    
    if show_ob:
        for ob in order_blocks:
            color = 'rgba(0,255,0,0.15)' if ob['type'] == 'bullish' else 'rgba(255,0,0,0.15)'
            fig3.add_shape(
                type='rect',
                x0=df['Date'].iloc[ob['bar']],
                x1=df['Date'].iloc[-1],
                y0=ob['bottom'],
                y1=ob['top'],
                fillcolor=color,
                line=dict(color='yellow' if ob['type'] == 'bullish' else 'red', width=1),
                layer='below'
            )
    
    if show_bos:
        for evt in bos_events:
            if evt['type'] == 'BOS_UP':
                fig3.add_trace(go.Scatter(
                    x=[df['Date'].iloc[evt['bar']]],
                    y=[evt['price']],
                    mode='markers+text',
                    marker=dict(color='lime', size=12, symbol='triangle-up'),
                    text=['BOS'], textposition='top center',
                    textfont=dict(color='lime', size=10),
                    showlegend=False
                ))
            else:
                fig3.add_trace(go.Scatter(
                    x=[df['Date'].iloc[evt['bar']]],
                    y=[evt['price']],
                    mode='markers+text',
                    marker=dict(color='red', size=12, symbol='triangle-down'),
                    text=['BOS'], textposition='bottom center',
                    textfont=dict(color='red', size=10),
                    showlegend=False
                ))
    
    fig3.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("**Recent BOS Events**")
    if bos_events:
        bos_df = pd.DataFrame(bos_events)[['type', 'price']]
        bos_df.columns = ['Type', 'Level']
        st.dataframe(bos_df.tail(10), use_container_width=True, hide_index=True)
    else:
        st.info("No BOS events detected in this range.")
    
    st.markdown("**Recent Order Blocks**")
    if order_blocks:
        ob_df = pd.DataFrame(order_blocks)[['type', 'top', 'bottom']]
        ob_df.columns = ['Type', 'Top', 'Bottom']
        st.dataframe(ob_df.tail(10), use_container_width=True, hide_index=True)
    else:
        st.info("No order blocks detected.")

# -------------------------------------------------------------------------
# TAB 4: OPTION CHAIN
# -------------------------------------------------------------------------
with tab4:
    st.subheader("📉 Live Option Chain Analysis")
    st.caption("Fetches real-time NSE option chain data. Select NIFTY or BANKNIFTY below.")
    
    oc_index = st.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], key="oc_index")
    
    if st.button("🔄 Fetch Option Chain", key="fetch_oc"):
        with st.spinner("Fetching option chain from NSE..."):
            oc_df = fetch_option_chain(oc_index)
            if oc_df is not None and not oc_df.empty:
                analysis = analyze_option_chain(oc_df, float(latest['Close']))
                if analysis:
                    st.session_state['oc_analysis'] = analysis
                    st.session_state['oc_index'] = oc_index
                    st.success("Option chain loaded!")
                else:
                    st.error("Could not parse option chain data.")
            else:
                st.error("Could not fetch option chain. NSE might be rate-limiting. Try again in a minute.")
    
    if 'oc_analysis' in st.session_state:
        analysis = st.session_state['oc_analysis']
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("PCR (Volume)", f"{analysis['pcr_volume']:.3f}")
        col2.metric("PCR (OI)", f"{analysis['pcr_oi']:.3f}")
        col3.metric("Call Wall (Resistance)", f"{analysis['call_wall_strike']:.0f}")
        col4.metric("Put Wall (Support)", f"{analysis['put_wall_strike']:.0f}")
        
        st.markdown("---")
        
        trend = analysis['trend']
        if trend == "BULLISH":
            st.success(f"**Trend: {trend}** — {analysis['trend_reason']}")
        elif trend == "BEARISH":
            st.error(f"**Trend: {trend}** — {analysis['trend_reason']}")
        else:
            st.info(f"**Trend: {trend}** — {analysis['trend_reason']}")
        
        st.markdown("---")
        st.markdown("### Volume Walls (Highest Volume Strikes)")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔴 Top Call Volume Strikes (Resistance)**")
            top_calls = analysis['top_call_strikes'].copy()
            top_calls.columns = ['Strike', 'Call Volume', 'Put Volume']
            st.dataframe(top_calls, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown("**🟢 Top Put Volume Strikes (Support)**")
            top_puts = analysis['top_put_strikes'].copy()
            top_puts.columns = ['Strike', 'Call Volume', 'Put Volume']
            st.dataframe(top_puts, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### Full Option Chain Data")
        st.dataframe(analysis['oc_df'], use_container_width=True, height=400)
    else:
        st.info("Click 'Fetch Option Chain' to load live data.")

# -------------------------------------------------------------------------
# TAB 5: VOLUME SHIFT ANALYSIS
# -------------------------------------------------------------------------
with tab5:
    st.subheader("📊 Volume Shift Analysis")
    st.caption("Shows where Call vs Put volume is shifting — this reveals institutional money movement.")
    
    if 'oc_analysis' not in st.session_state:
        st.warning("Please fetch the option chain first in the '📉 Option Chain' tab.")
    else:
        analysis = st.session_state['oc_analysis']
        oc = analysis['oc_df']
        strike_col = analysis['strike_col']
        ce_vol_col = analysis['ce_vol_col']
        pe_vol_col = analysis['pe_vol_col']
        
        fig_shift = go.Figure()
        fig_shift.add_trace(go.Bar(
            x=oc[strike_col], y=oc[ce_vol_col],
            name='Call Volume', marker_color='red', opacity=0.7
        ))
        fig_shift.add_trace(go.Bar(
            x=oc[strike_col], y=oc[pe_vol_col],
            name='Put Volume', marker_color='green', opacity=0.7
        ))
        
        fig_shift.update_layout(
            barmode='group', height=500, template='plotly_dark',
            title='Call vs Put Volume by Strike',
            xaxis_title='Strike Price', yaxis_title='Volume',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        st.plotly_chart(fig_shift, use_container_width=True)
        
        oc_clean = oc.dropna(subset=[strike_col])
        oc_clean = oc_clean.sort_values(strike_col)
        
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(
            x=oc_clean[strike_col], y=oc_clean['vol_ratio'],
            mode='lines+markers', name='Call/Put Volume Ratio',
            line=dict(color='yellow', width=2),
            marker=dict(size=8, color='yellow')
        ))
        fig_ratio.add_hline(y=1.0, line=dict(color='gray', dash='dash'), annotation_text='Balanced (1.0)')
        fig_ratio.add_hline(y=1.2, line=dict(color='red', dash='dot'), annotation_text='Call Heavy (>1.2)')
        fig_ratio.add_hline(y=0.8, line=dict(color='green', dash='dot'), annotation_text='Put Heavy (<0.8)')
        
        fig_ratio.update_layout(
            height=400, template='plotly_dark',
            title='Volume Shift Ratio (Call Volume / Put Volume)',
            xaxis_title='Strike Price', yaxis_title='Ratio'
        )
        st.plotly_chart(fig_ratio, use_container_width=True)
        
        st.markdown("### Where Volume is Shifting")
        
        shift_df = oc_clean[[strike_col, ce_vol_col, pe_vol_col, 'vol_ratio', 'vol_shift']].copy()
        shift_df.columns = ['Strike', 'Call Vol', 'Put Vol', 'Ratio', 'Shift']
        shift_df['Shift_Label'] = shift_df['Shift'].apply(
            lambda x: '🔴 Call Heavy' if x > 0.5 else ('🟢 Put Heavy' if x < -0.3 else '⚪ Balanced')
        )
        shift_df = shift_df.sort_values('Ratio', ascending=False)
        
        st.dataframe(shift_df, use_container_width=True, hide_index=True)
        
        st.markdown("### Summary")
        bullish_strikes = shift_df[shift_df['Ratio'] > 1.2]
        bearish_strikes = shift_df[shift_df['Ratio'] < 0.8]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**🔴 Strikes with Call Heaviness (Bearish Resistance)**")
            if not bullish_strikes.empty:
                st.dataframe(bullish_strikes[['Strike', 'Call Vol', 'Put Vol', 'Ratio']].head(5),
                           use_container_width=True, hide_index=True)
            else:
                st.info("No strong call-heavy strikes detected.")
        with col_b:
            st.markdown("**🟢 Strikes with Put Heaviness (Bullish Support)**")
            if not bearish_strikes.empty:
                st.dataframe(bearish_strikes[['Strike', 'Call Vol', 'Put Vol', 'Ratio']].head(5),
                           use_container_width=True, hide_index=True)
            else:
                st.info("No strong put-heavy strikes detected.")

# -------------------------------------------------------------------------
# TAB 6: SMART MONEY (NEW)
# -------------------------------------------------------------------------
with tab6:
    st.subheader("🏦 Smart Money Positioning")
    st.caption("Tracks FII/DII flows and participant-wise Open Interest to reveal where big money is moving.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### FII / DII Cash Market Flows")
        with st.spinner("Fetching FII/DII activity..."):
            fii_dii_df = fetch_fii_dii_activity()
            if fii_dii_df is not None and not fii_dii_df.empty:
                st.dataframe(fii_dii_df, use_container_width=True, height=300)
            else:
                st.warning("FII/DII data not available. NSE might be rate-limiting. Try again later.")

    with col_b:
        st.markdown("### Participant-wise Open Interest")
        today = datetime.now()
        # Try to fetch today's data, or fall back to yesterday if not yet published
        for days_back in range(0, 3):
            check_date = today - timedelta(days=days_back)
            date_str = check_date.strftime('%d-%m-%Y')
            with st.spinner(f"Fetching participant OI for {date_str}..."):
                participant_df = fetch_participant_oi(date_str)
                if participant_df is not None and not participant_df.empty:
                    st.success(f"Data loaded for {date_str}")
                    st.dataframe(participant_df, use_container_width=True, height=300)
                    break
                else:
                    st.warning(f"No data for {date_str}. Trying previous day...")
            if participant_df is not None:
                break

    # --- Who's Winning Signal ---
    if fii_dii_df is not None and participant_df is not None:
        st.markdown("---")
        st.markdown("### 🏆 Who is Winning?")
        
        try:
            # Extract net FII position from FII/DII cash flow
            # The dataframe structure varies; we try to find the FII row
            fii_cash_net = 0
            for _, row in fii_dii_df.iterrows():
                row_str = str(row.values)
                if 'FII' in row_str or 'FPI' in row_str:
                    # Try to find a numeric column for net value
                    for col in fii_dii_df.columns:
                        try:
                            val = float(row[col])
                            if abs(val) > 100:  # Net value should be in crores
                                fii_cash_net = val
                                break
                        except:
                            pass
                    break
            
            # Extract net FII position from participant OI (Index Futures)
            fii_futures_net = 0
            if 'FII' in participant_df.iloc[:, 0].values or 'FII' in participant_df.iloc[:, 0].astype(str).values:
                fii_row = participant_df[participant_df.iloc[:, 0].astype(str).str.contains('FII', na=False)]
                if not fii_row.empty:
                    # Find the Index Futures column
                    for col in participant_df.columns:
                        if 'future' in col.lower() or 'index' in col.lower():
                            try:
                                fii_futures_net = float(fii_row.iloc[0][col])
                            except:
                                pass
                            break
            
            # Determine who's winning
            if fii_cash_net < 0 and fii_futures_net < 0:
                st.error("🔴 **FIIs are BEARISH** — Selling in cash market AND short in futures.")
            elif fii_cash_net > 0 and fii_futures_net > 0:
                st.success("🟢 **FIIs are BULLISH** — Buying in cash market AND long in futures.")
            elif fii_cash_net < 0 and fii_futures_net > 0:
                st.warning("🟡 **Mixed Signals** — FIIs selling in cash but hedging with long futures.")
            elif fii_cash_net > 0 and fii_futures_net < 0:
                st.warning("🟡 **Mixed Signals** — FIIs buying in cash but hedging with short futures.")
            else:
                st.info("⚪ **Neutral** — No strong directional bias from FIIs.")
                
        except Exception as e:
            st.info("Could not compute the Who's Winning signal. Check the raw data above.")
    else:
        st.info("Fetch both FII/DII and Participant OI data to see the Who's Winning signal.")

# -------------------------------------------------------------------------
# TAB 7: DATA TABLE
# -------------------------------------------------------------------------
with tab7:
    st.subheader("Raw Data")
    display_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'OrderFlow']].tail(100).round(2)
    st.dataframe(display_df, use_container_width=True, height=600)
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, f"{selected_label}_data.csv", "text/csv")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
