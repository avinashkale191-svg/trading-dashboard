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
st.title("📈 Global Trading & Smart Money Dashboard")

# =========================================================================
# SIDEBAR
# =========================================================================
st.sidebar.header("Settings")

asset_options = {
    "--- INDIAN INDICES ---": None,
    "NIFTY 50": "^NSEI",
    "SENSEX": "^BSESN",
    "BANK NIFTY": "^NSEBANK",
    "--- INDIAN STOCKS ---": None,
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "SBIN": "SBIN.NS",
    "--- US STOCKS ---": None,
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "--- GLOBAL INDICES ---": None,
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Nikkei 225": "^N225",
    "FTSE 100": "^FTSE",
    "--- COMMODITIES ---": None,
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Crude Oil": "CL=F",
    "--- FOREX ---": None,
    "USD/INR": "USDINR=X",
    "EUR/USD": "EURUSD=X",
    "--- CRYPTO ---": None,
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
}

valid_options = {k: v for k, v in asset_options.items() if v is not None}
selected_label = st.sidebar.selectbox("Select Asset", list(valid_options.keys()))
ticker_symbol = valid_options[selected_label]

manual_ticker = st.sidebar.text_input("Custom Ticker", value="")
if manual_ticker.strip() != "":
    ticker_symbol = manual_ticker.strip()
    selected_label = ticker_symbol

# --- TIMEFRAME ---
st.sidebar.markdown("---")
st.sidebar.subheader("Timeframe")
interval = st.sidebar.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=5)

interval_limits = {
    "1m":  ["1d", "5d"],
    "5m":  ["1d", "5d", "1mo"],
    "15m": ["1d", "5d", "1mo"],
    "30m": ["1d", "5d", "1mo"],
    "1h":  ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    "1d":  ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
}
available_periods = interval_limits.get(interval, ["1mo", "3mo"])
period = st.sidebar.selectbox("Period", available_periods, index=min(2, len(available_periods)-1))

show_last_n = st.sidebar.slider("Bars to show (zoom)", 30, 500, 150, step=10)

# --- Indicators ---
st.sidebar.markdown("---")
st.sidebar.subheader("Indicators")
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
sma_fast = st.sidebar.slider("SMA Fast", 5, 50, 20)
sma_slow = st.sidebar.slider("SMA Slow", 20, 200, 50)
pivot_len = st.sidebar.slider("S/R Pivot Length", 3, 30, 10)
smc_pivot = st.sidebar.slider("SMC Pivot Length", 3, 30, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Overlays")
show_sr = st.sidebar.checkbox("Support / Resistance", value=True)
show_smc = st.sidebar.checkbox("SMC Liquidity", value=True)
show_bos = st.sidebar.checkbox("BOS", value=True)
show_sma = st.sidebar.checkbox("Moving Averages", value=True)

# =========================================================================
# DATA FETCH
# =========================================================================
@st.cache_data(ttl=180)
def fetch_data(symbol, period, interval):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        if 'Datetime' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'Datetime': 'Date'})
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in df.columns:
                return None
        df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        return df
    except Exception:
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
# OPTION CHAIN FUNCTIONS
# =========================================================================
@st.cache_data(ttl=180)
def fetch_option_chain_live(index_name):
    """Read option chain from GitHub-hosted JSON (auto-updated by GitHub Actions every 10 min)."""
    try:
        url = f"https://raw.githubusercontent.com/avinashkale191-svg/trading-dashboard/main/option_chain_{index_name.lower()}.json"
        r = http_requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            rows = data.get('strikes', [])
            if rows:
                df = pd.DataFrame(rows)
                st.session_state['oc_meta'] = {
                    'fetched_at': data.get('fetched_at', 'unknown'),
                    'underlying': data.get('underlying', 0),
                    'expiry': data.get('expiry', 'unknown'),
                }
                return df
        return None
    except Exception as e:
        return None
        def parse_option_chain(df):
    """Normalize option chain data to standard columns."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    
    # Find relevant columns
    strike_col = next((c for c in df.columns if 'strike' in c), None)
    ce_vol_col = next((c for c in df.columns if 'ce' in c and 'vol' in c), None)
    pe_vol_col = next((c for c in df.columns if 'pe' in c and 'vol' in c), None)
    ce_oi_col = next((c for c in df.columns if 'ce' in c and 'oi' in c), None)
    pe_oi_col = next((c for c in df.columns if 'pe' in c and 'oi' in c), None)
    
    if not strike_col:
        return None
    
    df = df.rename(columns={
        strike_col: 'Strike',
        **({ce_vol_col: 'Call_Volume'} if ce_vol_col else {}),
        **({pe_vol_col: 'Put_Volume'} if pe_vol_col else {}),
        **({ce_oi_col: 'Call_OI'} if ce_oi_col else {}),
        **({pe_oi_col: 'Put_OI'} if pe_oi_col else {}),
    })
    
    for col in ['Strike', 'Call_Volume', 'Put_Volume', 'Call_OI', 'Put_OI']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df[df['Strike'] > 0].sort_values('Strike').reset_index(drop=True)
    
    # Ensure Call_Volume and Put_Volume exist
    if 'Call_Volume' not in df.columns:
        df['Call_Volume'] = 0
    if 'Put_Volume' not in df.columns:
        df['Put_Volume'] = 0
    
    return df

def analyze_option_chain(df):
    """Compute PCR, walls, max pain, and volume shift."""
    if df is None or df.empty:
        return None
    
    total_ce_vol = df['Call_Volume'].sum()
    total_pe_vol = df['Put_Volume'].sum()
    pcr_vol = total_pe_vol / total_ce_vol if total_ce_vol > 0 else 0
    
    total_ce_oi = df['Call_OI'].sum() if 'Call_OI' in df.columns else 0
    total_pe_oi = df['Put_OI'].sum() if 'Put_OI' in df.columns else 0
    pcr_oi = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
    
    call_wall = df.loc[df['Call_Volume'].idxmax()] if total_ce_vol > 0 else None
    put_wall = df.loc[df['Put_Volume'].idxmax()] if total_pe_vol > 0 else None
    
    # Max pain: strike with minimum total value
    df['Total_Vol'] = df['Call_Volume'] + df['Put_Volume']
    max_pain_strike = df.loc[df['Total_Vol'].idxmin(), 'Strike'] if len(df) > 0 else 0
    
    # Volume ratio per strike
    df['Vol_Ratio'] = df['Call_Volume'] / (df['Put_Volume'] + 1)
    df['Shift'] = df['Vol_Ratio'] - 1
    df['Shift_Label'] = df['Shift'].apply(
        lambda x: 'CALL HEAVY' if x > 0.5 else ('PUT HEAVY' if x < -0.3 else 'BALANCED')
    )
    
    # Trend
    if pcr_vol > 1.2:
        trend = "BULLISH"
        reason = f"PCR {pcr_vol:.2f} > 1.2 → Puts dominate → Bulls writing puts"
    elif pcr_vol < 0.7:
        trend = "BEARISH"
        reason = f"PCR {pcr_vol:.2f} < 0.7 → Calls dominate → Bears writing calls"
    else:
        trend = "NEUTRAL"
        reason = f"PCR {pcr_vol:.2f} → Balanced activity"
    
    return {
        'total_ce_vol': total_ce_vol,
        'total_pe_vol': total_pe_vol,
        'total_ce_oi': total_ce_oi,
        'total_pe_oi': total_pe_oi,
        'pcr_vol': pcr_vol,
        'pcr_oi': pcr_oi,
        'call_wall_strike': call_wall['Strike'] if call_wall is not None else 0,
        'call_wall_vol': call_wall['Call_Volume'] if call_wall is not None else 0,
        'put_wall_strike': put_wall['Strike'] if put_wall is not None else 0,
        'put_wall_vol': put_wall['Put_Volume'] if put_wall is not None else 0,
        'max_pain': max_pain_strike,
        'trend': trend,
        'reason': reason,
        'df': df,
    }

# =========================================================================
# PROCESS
# =========================================================================
df = fetch_data(ticker_symbol, period, interval)
if df is None or len(df) < 20:
    st.error(f"⚠️ Could not fetch data for **{ticker_symbol}** at {interval}/{period}")
    st.info("Try a different interval/period. For NIFTY 50 use `1d` or `1h`.")
    st.stop()

df['RSI'] = calc_rsi(df['Close'], rsi_period)
df['OrderFlow'] = calc_order_flow(df)
df['SMA_Fast'] = df['Close'].rolling(sma_fast).mean()
df['SMA_Slow'] = df['Close'].rolling(sma_slow).mean()

res_levels, sup_levels = find_sr(df, pivot_len)
bsl, ssl = find_smc(df, smc_pivot)
bos_events = find_bos(df, smc_pivot)

latest = df.iloc[-1]
prev = df.iloc[-2]
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
    Score: {trend_score:+d}</span></div>""",
    unsafe_allow_html=True
)

df_display = df.tail(show_last_n).copy()
bars_offset = len(df) - len(df_display)
x_start = df_display['Date'].iloc[0]
x_end = df_display['Date'].iloc[-1]

# =========================================================================
# TABS
# =========================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Chart", "📉 Option Chain", "💧 SMC", "🏦 Smart Money", "📋 Data"
])

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
# TAB 2: OPTION CHAIN
# -------------------------------------------------------------------------
with tab2:
    st.subheader("📉 Option Chain — Call/Put Volume Analysis")
    
    if 'oc_data' not in st.session_state:
        st.session_state['oc_data'] = None
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🚀 Option 1: Auto-Fetch (NIFTY/BANKNIFTY)")
        oc_index = st.selectbox("Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"], key="oc_idx")
        if st.button("🔄 Fetch Live Option Chain", key="fetch_oc"):
            with st.spinner("Fetching from NSE..."):
                raw = fetch_option_chain_live(oc_index)
                parsed = parse_option_chain(raw)
                if parsed is not None and not parsed.empty:
                    st.session_state['oc_data'] = parsed
                    st.success(f"✅ Loaded {len(parsed)} strikes")
                else:
                    st.error("❌ Auto-fetch failed. NSE is likely blocking. Use Manual Paste below.")
    
    with col2:
        st.markdown("### ✍️ Option 2: Manual Paste (Always Works)")
        st.caption("Go to nseindia.com → Option Chain → Copy the table → Paste here")
        pasted = st.text_area("Paste CSV/TSV from NSE option chain", height=150, key="manual_paste")
        if st.button("📋 Load Pasted Data", key="load_paste"):
            if pasted.strip():
                try:
                    manual_df = pd.read_csv(io.StringIO(pasted), sep=None, engine='python')
                    parsed = parse_option_chain(manual_df)
                    if parsed is not None and not parsed.empty:
                        st.session_state['oc_data'] = parsed
                        st.success(f"✅ Loaded {len(parsed)} strikes")
                    else:
                        st.error("Could not parse. Ensure columns include Strike, Call Volume, Put Volume.")
                except Exception as e:
                    st.error(f"Parse error: {e}")
            else:
                st.warning("Paste some data first")
    
    # Display analysis
    oc_df = st.session_state.get('oc_data')
    if oc_df is not None and not oc_df.empty:
        st.markdown("---")
        analysis = analyze_option_chain(oc_df)
        
        # Summary metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Call Volume", f"{analysis['total_ce_vol']:,.0f}")
        m2.metric("Total Put Volume", f"{analysis['total_pe_vol']:,.0f}")
        m3.metric("PCR (Volume)", f"{analysis['pcr_vol']:.3f}")
        m4.metric("Call Wall", f"{analysis['call_wall_strike']:.0f}")
        m5.metric("Put Wall", f"{analysis['put_wall_strike']:.0f}")
        
        m6, m7, m8 = st.columns(3)
        m6.metric("PCR (OI)", f"{analysis['pcr_oi']:.3f}" if analysis['pcr_oi'] > 0 else "—")
        m7.metric("Max Pain", f"{analysis['max_pain']:.0f}")
        m8.metric("Trend", analysis['trend'])
        
        st.info(f"**{analysis['trend']}** — {analysis['reason']}")
        
        # Volume bar chart
        st.markdown("### Volume by Strike")
        fig_oc = go.Figure()
        fig_oc.add_trace(go.Bar(x=oc_df['Strike'], y=oc_df['Call_Volume'],
                                name='Call Volume', marker_color='#ff4444', opacity=0.8))
        fig_oc.add_trace(go.Bar(x=oc_df['Strike'], y=oc_df['Put_Volume'],
                                name='Put Volume', marker_color='#00ff88', opacity=0.8))
        fig_oc.update_layout(barmode='group', height=400, template='plotly_dark',
                             paper_bgcolor='#131722', plot_bgcolor='#131722',
                             xaxis_title='Strike', yaxis_title='Volume',
                             legend=dict(orientation='h', y=1.02))
        st.plotly_chart(fig_oc, use_container_width=True)
        
        # Volume ratio line
        st.markdown("### Call vs Put Volume Ratio (per strike)")
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=oc_df['Strike'], y=oc_df['Vol_Ratio'],
                                       mode='lines+markers', name='CE/PE Ratio',
                                       line=dict(color='#ffaa00', width=2)))
        fig_ratio.add_hline(y=1.0, line=dict(color='gray', dash='dash'), annotation_text='Balanced')
        fig_ratio.add_hline(y=1.2, line=dict(color='#ff4444', dash='dot'), annotation_text='Call Heavy')
        fig_ratio.add_hline(y=0.8, line=dict(color='#00ff88', dash='dot'), annotation_text='Put Heavy')
        fig_ratio.update_layout(height=350, template='plotly_dark',
                                paper_bgcolor='#131722', plot_bgcolor='#131722',
                                xaxis_title='Strike', yaxis_title='Ratio')
        st.plotly_chart(fig_ratio, use_container_width=True)
        
        # Full table with numbers
        st.markdown("### Full Option Chain Table")
        
        display_cols = ['Strike']
        if 'Call_OI' in oc_df.columns: display_cols.append('Call_OI')
        display_cols.append('Call_Volume')
        if 'Put_OI' in oc_df.columns: display_cols.append('Put_OI')
        display_cols.append('Put_Volume')
        display_cols.extend(['Vol_Ratio', 'Shift_Label'])
        
        table_df = oc_df[display_cols].copy().round(2)
        
        # Highlight max call and put volume strikes
        st.dataframe(
            table_df.style.apply(
                lambda row: ['background-color: #ff444433' if row['Strike'] == analysis['call_wall_strike']
                             else ('background-color: #00ff8833' if row['Strike'] == analysis['put_wall_strike']
                                   else '') for _ in row],
                axis=1
            ),
            use_container_width=True, height=500
        )
        
        st.caption(f"🔴 Call Wall (Resistance): **{analysis['call_wall_strike']:.0f}** ({analysis['call_wall_vol']:,.0f} calls) · 🟢 Put Wall (Support): **{analysis['put_wall_strike']:.0f}** ({analysis['put_wall_vol']:,.0f} puts)")
        
        # Clear button
        if st.button("🗑️ Clear Option Chain"):
            st.session_state['oc_data'] = None
            st.rerun()
    else:
        st.info("👆 Use Auto-Fetch or Manual Paste to load option chain data")

# -------------------------------------------------------------------------
# TAB 3: SMC
# -------------------------------------------------------------------------
with tab3:
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
# TAB 4: SMART MONEY
# -------------------------------------------------------------------------
with tab4:
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

# -------------------------------------------------------------------------
# TAB 5: DATA
# -------------------------------------------------------------------------
with tab5:
    st.subheader("📋 Raw Data")
    display_df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'OrderFlow']].tail(200).round(2)
    st.dataframe(display_df, use_container_width=True, height=600)
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download CSV", csv, f"{selected_label}_data.csv", "text/csv")

st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
