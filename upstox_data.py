"""
upstox_data.py — Fetch live BankNifty + option chain from Upstox Analytics API.
Read-only. No order placement.
"""

import os
import requests
import pandas as pd
from datetime import datetime
import streamlit as st


# =========================================================================
# TOKEN LOADING (works in Codespace + Streamlit Cloud)
# =========================================================================
def get_upstox_token():
    """Read token from Streamlit secrets OR env variable."""
    try:
        return st.secrets["UPSTOX_TOKEN"]
    except Exception:
        pass
    
    token = os.getenv("UPSTOX_TOKEN")
    if token:
        return token
    
    # Fallback: read from local file (Codespace testing)
    try:
        with open("token.txt", "r") as f:
            return f.read().strip()
    except Exception:
        pass
    
    return None


UPSTOX_TOKEN = get_upstox_token()
BASE_URL = "https://api.upstox.com/v2"


# =========================================================================
# HEADERS
# =========================================================================
def get_headers():
    return {
        "Authorization": f"Bearer {UPSTOX_TOKEN}",
        "Accept": "application/json"
    }


# =========================================================================
# 1. LIVE SPOT QUOTE
# =========================================================================
def get_spot_quote(symbol="NSE_INDEX|Nifty Bank"):
    """Fetch live spot price for BankNifty / Nifty."""
    if not UPSTOX_TOKEN:
        return None
    
    url = f"{BASE_URL}/market-quote/quotes"
    params = {"symbol": symbol}
    
    try:
        r = requests.get(url, headers=get_headers(), params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            key = list(data.get("data", {}).keys())
            if key:
                quote = data["data"][key[0]]
                return {
                    "symbol": symbol,
                    "ltp": quote.get("last_price"),
                    "open": quote.get("ohlc", {}).get("open"),
                    "high": quote.get("ohlc", {}).get("high"),
                    "low": quote.get("ohlc", {}).get("low"),
                    "close": quote.get("ohlc", {}).get("close"),
                    "volume": quote.get("volume"),
                    "change": quote.get("net_change"),
                    "pct_change": quote.get("net_change", 0) / quote.get("ohlc", {}).get("close", 1) * 100 if quote.get("ohlc", {}).get("close") else 0,
                }
    except Exception as e:
        print(f"Error fetching spot: {e}")
    
    return None


# =========================================================================
# 2. OPTION CHAIN
# =========================================================================
def get_option_chain(underlying="NSE_INDEX|Nifty Bank", expiry=None):
    """
    Fetch live option chain.
    Returns: DataFrame with strikes, CE OI, PE OI, IV, LTP, PCR, Max Pain.
    """
    if not UPSTOX_TOKEN:
        return None
    
    # First get available expiries
    if expiry is None:
        expiry = get_nearest_expiry(underlying)
        if expiry is None:
            return None
    
    url = f"{BASE_URL}/option/chain"
    params = {
        "instrument_key": underlying,
        "expiry_date": expiry
    }
    
    try:
        r = requests.get(url, headers=get_headers(), params=params, timeout=15)
        if r.status_code != 200:
            print(f"Option chain error: {r.status_code} — {r.text[:200]}")
            return None
        
        data = r.json().get("data", [])
        if not data:
            return None
        
        rows = []
        for item in data:
            strike = item.get("strike_price")
            ce = item.get("call_options", {}).get("market_data", {})
            pe = item.get("put_options", {}).get("market_data", {})
            ce_greeks = item.get("call_options", {}).get("option_greeks", {})
            pe_greeks = item.get("put_options", {}).get("option_greeks", {})
            
            rows.append({
                "strike": strike,
                "ce_ltp": ce.get("ltp"),
                "ce_oi": ce.get("oi"),
                "ce_volume": ce.get("volume"),
                "ce_iv": ce_greeks.get("iv"),
                "ce_delta": ce_greeks.get("delta"),
                "pe_ltp": pe.get("ltp"),
                "pe_oi": pe.get("oi"),
                "pe_volume": pe.get("volume"),
                "pe_iv": pe_greeks.get("iv"),
                "pe_delta": pe_greeks.get("delta"),
            })
        
        df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
        
        # Compute PCR
        total_ce_oi = df["ce_oi"].sum()
        total_pe_oi = df["pe_oi"].sum()
        pcr = total_pe_oi / total_ce_oi if total_ce_oi > 0 else 0
        
        # Compute Max Pain
        max_pain = compute_max_pain(df)
        
        df.attrs["pcr"] = pcr
        df.attrs["max_pain"] = max_pain
        df.attrs["total_ce_oi"] = total_ce_oi
        df.attrs["total_pe_oi"] = total_pe_oi
        df.attrs["expiry"] = expiry
        
        return df
        
    except Exception as e:
        print(f"Error fetching option chain: {e}")
        return None


# =========================================================================
# 3. GET NEAREST EXPIRY
# =========================================================================
def get_nearest_expiry(underlying="NSE_INDEX|Nifty Bank"):
    """Get nearest upcoming expiry."""
    url = f"{BASE_URL}/option/contract"
    params = {"instrument_key": underlying}
    
    try:
        r = requests.get(url, headers=get_headers(), params=params, timeout=10)
        if r.status_code == 200:
            data = r.json().get("data", [])
            expiries = sorted(set(item.get("expiry") for item in data if item.get("expiry")))
            if expiries:
                today = datetime.now().date()
                for exp in expiries:
                    try:
                        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
                        if exp_date >= today:
                            return exp
                    except Exception:
                        continue
                return expiries[0]
    except Exception as e:
        print(f"Error fetching expiries: {e}")
    
    return None


# =========================================================================
# 4. MAX PAIN CALCULATOR
# =========================================================================
def compute_max_pain(df):
    """Compute Max Pain strike = strike where total OI loss is minimum."""
    if df.empty:
        return None
    
    strikes = df["strike"].tolist()
    min_loss = None
    max_pain_strike = None
    
    for test_strike in strikes:
        total_loss = 0
        for _, row in df.iterrows():
            strike = row["strike"]
            ce_oi = row["ce_oi"] or 0
            pe_oi = row["pe_oi"] or 0
            
            # CE writers lose if strike < test (ITM call)
            if strike < test_strike:
                total_loss += ce_oi * (test_strike - strike)
            
            # PE writers lose if strike > test (ITM put)
            if strike > test_strike:
                total_loss += pe_oi * (strike - test_strike)
        
        if min_loss is None or total_loss < min_loss:
            min_loss = total_loss
            max_pain_strike = test_strike
    
    return max_pain_strike


# =========================================================================
# CANDLES (OHLC) — for chart
# =========================================================================
def fetch_candles(symbol="NSE_INDEX|Nifty Bank", interval="1m", days=5):
    """Fetch OHLC candles from Upstox.
    Tries INTRADAY endpoint first (live today),
    falls back to HISTORICAL endpoint (older dates).
    """
    if not UPSTOX_TOKEN:
        return None

    interval_map = {
        "1m": "1minute", "5m": "5minute", "15m": "15minute",
        "30m": "30minute", "1h": "60minute", "1d": "day",
    }
    upstox_interval = interval_map.get(interval, "1minute")

    encoded_symbol = symbol.replace("|", "%7C")

    def _parse(json_data):
        candles = json_data.get("data", {}).get("candles", [])
        if not candles:
            return None
        df = pd.DataFrame(candles, columns=["Date", "Open", "High", "Low", "Close", "Volume", "OI"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
        return df

    # ---- Try INTRADAY endpoint first (today's live candles) ----
    try:
        intraday_url = f"{BASE_URL}/historical-candle/intraday/{encoded_symbol}/{upstox_interval}"
        r = requests.get(intraday_url, headers=get_headers(), timeout=15)
        if r.status_code == 200:
            df = _parse(r.json())
            if df is not None and not df.empty:
                return df
        else:
            print(f"Intraday endpoint {r.status_code} — falling back to historical")
    except Exception as e:
        print(f"Intraday exception: {e}")

    # ---- Fallback: HISTORICAL endpoint ----
    from_date_obj = datetime.now() - pd.Timedelta(days=days)
    from_date = from_date_obj.strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    hist_url = f"{BASE_URL}/historical-candle/{encoded_symbol}/{upstox_interval}/{to_date}/{from_date}"

    try:
        r = requests.get(hist_url, headers=get_headers(), timeout=15)
        if r.status_code != 200:
            print(f"Historical error: {r.status_code} - {r.text[:200]}")
            return None
        return _parse(r.json())
    except Exception as e:
        print(f"Historical exception: {e}")
        return None
# =========================================================================
# 5. TEST
# =========================================================================
if __name__ == "__main__":
    print("Testing Upstox API...")
    print(f"Token loaded: {'YES' if UPSTOX_TOKEN else 'NO'}")
    
    if not UPSTOX_TOKEN:
        print("❌ No token found. Save token in token.txt or Streamlit secrets.")
        exit(1)
    
    print("\n--- BankNifty Spot ---")
    spot = get_spot_quote()
    if spot:
        for k, v in spot.items():
            print(f"  {k}: {v}")
    else:
        print("  ❌ Failed")
    
    print("\n--- Option Chain ---")
    chain = get_option_chain()
    if chain is not None and not chain.empty:
        print(f"  Strikes: {len(chain)}")
        print(f"  PCR: {chain.attrs['pcr']:.2f}")
        print(f"  Max Pain: {chain.attrs['max_pain']}")
        print(f"  Expiry: {chain.attrs['expiry']}")
        print(f"\n  Top 5 strikes by CE OI:")
        print(chain.nlargest(5, "ce_oi")[["strike", "ce_oi", "pe_oi"]].to_string(index=False))
    else:
        print("  ❌ Failed")
