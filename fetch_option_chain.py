"""
Fetches NSE option chain data and saves as JSON.
Runs from GitHub Actions (Microsoft IPs) which work better than AWS.
"""
import json
import requests
import pandas as pd
from datetime import datetime
import time
import sys

def fetch_nse_option_chain(symbol="NIFTY"):
    """Fetch option chain from NSE with proper headers and cookies."""
    session = requests.Session()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                      '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Referer': 'https://www.nseindia.com/option-chain',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
    }
    
    session.headers.update(headers)
    
    # Step 1: Hit homepage to get cookies
    try:
        session.get('https://www.nseindia.com', timeout=15)
        time.sleep(1)
        session.get('https://www.nseindia.com/option-chain', timeout=15)
        time.sleep(1)
    except Exception as e:
        print(f"Cookie fetch failed: {e}")
    
    # Step 2: Fetch option chain API
    url = f'https://www.nseindia.com/api/option-chain-indices?symbol={symbol}'
    
    try:
        r = session.get(url, timeout=20)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"API returned status {r.status_code}")
            return None
    except Exception as e:
        print(f"API fetch failed: {e}")
        return None

def parse_chain(data):
    """Convert NSE option chain JSON to clean list of strikes."""
    if not data:
        return None
    
    records = data.get('records', {})
    expiry_dates = records.get('expiryDates', [])
    
    if not expiry_dates:
        return None
    
    # Use nearest expiry
    nearest_expiry = expiry_dates[0]
    
    rows = []
    for item in records.get('data', []):
        if item.get('expiryDate') != nearest_expiry:
            continue
        
        strike = item.get('strikePrice', 0)
        ce = item.get('CE', {})
        pe = item.get('PE', {})
        
        rows.append({
            'Strike': strike,
            'Call_OI': ce.get('openInterest', 0),
            'Call_Volume': ce.get('totalTradedVolume', 0),
            'Call_LTP': ce.get('lastPrice', 0),
            'Call_Change': ce.get('changeinOpenInterest', 0),
            'Put_OI': pe.get('openInterest', 0),
            'Put_Volume': pe.get('totalTradedVolume', 0),
            'Put_LTP': pe.get('lastPrice', 0),
            'Put_Change': pe.get('changeinOpenInterest', 0),
        })
    
    return {
        'expiry': nearest_expiry,
        'timestamp': records.get('timestamp', ''),
        'underlying_value': records.get('underlyingValue', 0),
        'strikes': rows,
    }

def main():
    for symbol in ["NIFTY", "BANKNIFTY"]:
        print(f"Fetching {symbol}...")
        data = fetch_nse_option_chain(symbol)
        parsed = parse_chain(data)
        
        if parsed:
            output = {
                'symbol': symbol,
                'fetched_at': datetime.now().isoformat(),
                'underlying': parsed['underlying_value'],
                'expiry': parsed['expiry'],
                'strikes': parsed['strikes'],
            }
            
            filename = f'option_chain_{symbol.lower()}.json'
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"Saved {symbol}: {len(parsed['strikes'])} strikes, spot={parsed['underlying_value']}")
        else:
            print(f"Failed to fetch {symbol}")
        
        time.sleep(2)

if __name__ == '__main__':
    main()
