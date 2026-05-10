import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import firebase_admin
from firebase_admin import credentials, messaging
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# FIREBASE SETUP
# ---------------------------------------------------------

try:
    cred = credentials.Certificate("service-account.json")
    firebase_admin.initialize_app(cred)
    firebase_ready = True
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Warning: Firebase not initialized (service-account.json missing). Notifications will be printed to console. Error: {e}")
    firebase_ready = False

class TokenRequest(BaseModel):
    token: str

registered_tokens = set()

def send_push_notification(title, body):
    if not registered_tokens:
        print("No tokens registered. Notification not sent:", title)
        return

    if firebase_ready:
        for token in registered_tokens:
            try:
                message = messaging.Message(
                    notification=messaging.Notification(title=title, body=body),
                    token=token,
                )
                response = messaging.send(message)
                print('Successfully sent push notification:', response)
            except Exception as e:
                print('Error sending push notification:', e)
    else:
        print(f"\n🔔 [MOCK NOTIFICATION] {title}: {body}")

# ---------------------------------------------------------
# INDICATOR FUNCTIONS (Fully Custom for maximum reliability)
# ---------------------------------------------------------

def calculate_psar(df, step=0.03, max_step=0.09):
    """Calculates Parabolic SAR."""
    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values
    
    psar = close.copy()
    psar_dir = 1  
    af = step
    ep = high[0]
    psar[0] = low[0]
    
    for i in range(1, len(df)):
        prev_psar = psar[i-1]
        psar[i] = prev_psar + af * (ep - prev_psar)
        
        if psar_dir == 1:
            if low[i] < psar[i]:
                psar_dir = -1
                psar[i] = ep
                ep = low[i]
                af = step
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + step, max_step)
        else:
            if high[i] > psar[i]:
                psar_dir = 1
                psar[i] = ep
                ep = high[i]
                af = step
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + step, max_step)
    return psar

def calculate_basic_smc(df):
    """Calculates Break of Structure (BOS) using Swing Highs/Lows."""
    df['Swing_High'] = df['High'] == df['High'].rolling(window=5, center=True).max()
    df['Swing_Low'] = df['Low'] == df['Low'].rolling(window=5, center=True).min()
    
    df['BOS_Bullish'] = False
    df['BOS_Bearish'] = False
    
    last_swing_high = None
    last_swing_low = None
    
    for i in range(len(df)):
        if df['Swing_High'].iloc[i]:
            last_swing_high = df['High'].iloc[i]
        if df['Swing_Low'].iloc[i]:
            last_swing_low = df['Low'].iloc[i]
            
        if last_swing_high is not None and df['Close'].iloc[i] > last_swing_high:
            df.loc[df.index[i], 'BOS_Bullish'] = True
            last_swing_high = None 
            
        if last_swing_low is not None and df['Close'].iloc[i] < last_swing_low:
            df.loc[df.index[i], 'BOS_Bearish'] = True
            last_swing_low = None 

    return df

# ---------------------------------------------------------
# SENTIMENT ANALYSIS
# ---------------------------------------------------------

def analyze_sentiment(symbol: str):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if not news:
            return {"score": 0.0, "label": "Neutral ⚪", "articles_analyzed": 0}
            
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in news:
            title = article.get('title', '')
            if title:
                vs = analyzer.polarity_scores(title)
                scores.append(vs['compound'])
                
        if not scores:
            return {"score": 0.0, "label": "Neutral ⚪", "articles_analyzed": 0}
            
        avg_score = sum(scores) / len(scores)
        
        if avg_score > 0.05:
            label = "Bullish 🟢"
        elif avg_score < -0.05:
            label = "Bearish 🔴"
        else:
            label = "Neutral ⚪"
            
        return {
            "score": round(avg_score, 2),
            "label": label,
            "articles_analyzed": len(scores)
        }
    except Exception as e:
        print(f"Sentiment error: {e}")
        return {"score": 0.0, "label": "Neutral ⚪", "articles_analyzed": 0}

# ---------------------------------------------------------
# FASTAPI ENDPOINTS
# ---------------------------------------------------------

@app.post("/api/register-token")
def register_token(req: TokenRequest):
    registered_tokens.add(req.token)
    print(f"Token registered successfully: {req.token}")
    return {"status": "success"}

@app.get("/api/data")
def get_chart_data(symbol: str = "GC=F"):
    try:
        data = yf.download(symbol, period="5d", interval="15m", progress=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if data.empty:
        raise HTTPException(status_code=404, detail="No data fetched.")
        
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['PSAR'] = calculate_psar(df, step=0.03, max_step=0.09)
    df = calculate_basic_smc(df)
    
    candles = []
    markers = []
    
    for index, row in df.iterrows():
        timestamp = int(index.timestamp())
        
        # Check for NaN and handle
        if pd.isna(row['Open']) or pd.isna(row['High']) or pd.isna(row['Low']) or pd.isna(row['Close']):
            continue
            
        candles.append({
            "time": timestamp,
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close'])
        })
        
        is_uptrend = row['Close'] > row['PSAR']
        is_downtrend = row['Close'] < row['PSAR']
        
        if is_uptrend and row['BOS_Bullish']:
            markers.append({
                "time": timestamp,
                "position": "belowBar",
                "color": "#26a69a",
                "shape": "arrowUp",
                "text": "BUY SIGNAL (BOS)"
            })
        elif is_downtrend and row['BOS_Bearish']:
            markers.append({
                "time": timestamp,
                "position": "aboveBar",
                "color": "#ef5350",
                "shape": "arrowDown",
                "text": "SELL SIGNAL (BOS)"
            })
            
    sentiment_data = analyze_sentiment(symbol)
    
    # Check the last candle for live signals to trigger push notification
    last_row = df.iloc[-1]
    is_uptrend_last = last_row['Close'] > last_row['PSAR']
    is_downtrend_last = last_row['Close'] < last_row['PSAR']
    
    if is_uptrend_last and last_row['BOS_Bullish']:
        send_push_notification("🟢 BUY SIGNAL", f"{symbol} is showing a bullish setup at {last_row['Close']:.2f}")
    elif is_downtrend_last and last_row['BOS_Bearish']:
        send_push_notification("🔴 SELL SIGNAL", f"{symbol} is showing a bearish setup at {last_row['Close']:.2f}")
            
    return {
        "candles": candles,
        "markers": markers,
        "sentiment": sentiment_data
    }

# ---------------------------------------------------------
# MASTER SYMBOL CATALOG
# ---------------------------------------------------------
# Format: display_pair → { ticker, category, decimals, tp_pct, sl_pct }
#   tp_pct / sl_pct: % move for TP and SL (positive = away from entry)
SYMBOL_CATALOG = {
    # ── COMMODITIES ─────────────────────────────────────────
    "XAUUSD":  {"ticker": "GC=F",       "category": "Commodities", "decimals": 2, "tp": 1.2, "sl": 0.6},
    "XAGUSD":  {"ticker": "SI=F",       "category": "Commodities", "decimals": 3, "tp": 1.5, "sl": 0.7},
    "XTIUSD":  {"ticker": "CL=F",       "category": "Commodities", "decimals": 2, "tp": 1.5, "sl": 0.7},
    "XNGUSD":  {"ticker": "NG=F",       "category": "Commodities", "decimals": 3, "tp": 1.8, "sl": 0.9},
    "XCUUSD":  {"ticker": "HG=F",       "category": "Commodities", "decimals": 4, "tp": 1.2, "sl": 0.6},

    # ── INDICES ─────────────────────────────────────────────
    "US30":    {"ticker": "YM=F",       "category": "Indices",     "decimals": 0, "tp": 0.8, "sl": 0.4},
    "US100":   {"ticker": "NQ=F",       "category": "Indices",     "decimals": 0, "tp": 0.8, "sl": 0.4},
    "US500":   {"ticker": "ES=F",       "category": "Indices",     "decimals": 1, "tp": 0.8, "sl": 0.4},
    "GER40":   {"ticker": "FDAX=F",     "category": "Indices",     "decimals": 0, "tp": 0.8, "sl": 0.4},
    "UK100":   {"ticker": "Z=F",        "category": "Indices",     "decimals": 0, "tp": 0.8, "sl": 0.4},

    # ── STOCKS ──────────────────────────────────────────────
    "NVDA":    {"ticker": "NVDA",       "category": "Stocks",      "decimals": 2, "tp": 1.5, "sl": 0.7},
    "TSLA":    {"ticker": "TSLA",       "category": "Stocks",      "decimals": 2, "tp": 1.8, "sl": 0.9},
    "AAPL":    {"ticker": "AAPL",       "category": "Stocks",      "decimals": 2, "tp": 1.2, "sl": 0.6},
    "GOOGL":   {"ticker": "GOOGL",      "category": "Stocks",      "decimals": 2, "tp": 1.2, "sl": 0.6},
    "AMD":     {"ticker": "AMD",        "category": "Stocks",      "decimals": 2, "tp": 1.5, "sl": 0.8},
    "INTC":    {"ticker": "INTC",       "category": "Stocks",      "decimals": 2, "tp": 1.2, "sl": 0.6},
    "MSFT":    {"ticker": "MSFT",       "category": "Stocks",      "decimals": 2, "tp": 1.2, "sl": 0.6},
    "AMZN":    {"ticker": "AMZN",       "category": "Stocks",      "decimals": 2, "tp": 1.2, "sl": 0.6},
    "META":    {"ticker": "META",       "category": "Stocks",      "decimals": 2, "tp": 1.5, "sl": 0.7},

    # ── CRYPTO ──────────────────────────────────────────────
    "BTCUSD":  {"ticker": "BTC-USD",    "category": "Crypto",      "decimals": 0, "tp": 2.0, "sl": 1.0},
    "ETHUSD":  {"ticker": "ETH-USD",    "category": "Crypto",      "decimals": 1, "tp": 2.5, "sl": 1.2},
    "SOLUSD":  {"ticker": "SOL-USD",    "category": "Crypto",      "decimals": 2, "tp": 3.0, "sl": 1.5},
    "BNBUSD":  {"ticker": "BNB-USD",    "category": "Crypto",      "decimals": 2, "tp": 2.0, "sl": 1.0},

    # ── MAJOR FOREX ─────────────────────────────────────────
    "EURUSD":  {"ticker": "EURUSD=X",   "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},
    "GBPUSD":  {"ticker": "GBPUSD=X",   "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},
    "USDJPY":  {"ticker": "JPY=X",      "category": "Forex",       "decimals": 3, "tp": 0.4, "sl": 0.2},
    "USDCHF":  {"ticker": "CHF=X",      "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},
    "USDCAD":  {"ticker": "CAD=X",      "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},
    "AUDUSD":  {"ticker": "AUDUSD=X",   "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},
    "NZDUSD":  {"ticker": "NZDUSD=X",   "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},

    # ── CROSS FOREX ─────────────────────────────────────────
    "EURGBP":  {"ticker": "EURGBP=X",   "category": "Forex",       "decimals": 5, "tp": 0.3, "sl": 0.15},
    "EURJPY":  {"ticker": "EURJPY=X",   "category": "Forex",       "decimals": 3, "tp": 0.4, "sl": 0.2},
    "GBPJPY":  {"ticker": "GBPJPY=X",   "category": "Forex",       "decimals": 3, "tp": 0.5, "sl": 0.25},
    "AUDJPY":  {"ticker": "AUDJPY=X",   "category": "Forex",       "decimals": 3, "tp": 0.4, "sl": 0.2},
    "CADJPY":  {"ticker": "CADJPY=X",   "category": "Forex",       "decimals": 3, "tp": 0.4, "sl": 0.2},
    "CHFJPY":  {"ticker": "CHFJPY=X",   "category": "Forex",       "decimals": 3, "tp": 0.4, "sl": 0.2},
    "EURCHF":  {"ticker": "EURCHF=X",   "category": "Forex",       "decimals": 5, "tp": 0.3, "sl": 0.15},
    "GBPCHF":  {"ticker": "GBPCHF=X",   "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},
    "EURAUD":  {"ticker": "EURAUD=X",   "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},
    "EURCAD":  {"ticker": "EURCAD=X",   "category": "Forex",       "decimals": 5, "tp": 0.4, "sl": 0.2},
    "GBPAUD":  {"ticker": "GBPAUD=X",   "category": "Forex",       "decimals": 5, "tp": 0.5, "sl": 0.25},
}

# Flat ticker map for backward compat (used by /api/data)
SIGNAL_SYMBOLS = {k: v["ticker"] for k, v in SYMBOL_CATALOG.items()}

def pips_away(price: float, pct: float) -> float:
    return round(price * (1 + pct / 100), 8)


def build_signal_from_df(pair: str, df: pd.DataFrame, sentiment: dict) -> dict | None:
    """Derive a structured signal from the last BOS event in the dataframe."""
    try:
        meta = SYMBOL_CATALOG.get(pair, {})
        decimals = meta.get("decimals", 5)
        tp_pct   = meta.get("tp", 1.5)
        sl_pct   = meta.get("sl", 0.7)
        category = meta.get("category", "Forex")

        df = df.copy()
        df['PSAR'] = calculate_psar(df)
        df = calculate_basic_smc(df)

        last_buy_idx  = df[df['BOS_Bullish']].index.max()
        last_sell_idx = df[df['BOS_Bearish']].index.max()

        if pd.isna(last_buy_idx) and pd.isna(last_sell_idx):
            return None

        if pd.isna(last_buy_idx):
            direction, signal_idx = "SELL", last_sell_idx
        elif pd.isna(last_sell_idx):
            direction, signal_idx = "BUY", last_buy_idx
        else:
            direction = "BUY" if last_buy_idx > last_sell_idx else "SELL"
            signal_idx = last_buy_idx if direction == "BUY" else last_sell_idx

        row = df.loc[signal_idx]
        current_price = float(df.iloc[-1]['Close'])
        entry = float(row['Close'])

        if direction == "BUY":
            tp = pips_away(entry,  tp_pct)
            sl = pips_away(entry, -sl_pct)
        else:
            tp = pips_away(entry, -tp_pct)
            sl = pips_away(entry,  sl_pct)

        candle_age    = len(df) - df.index.get_loc(signal_idx)
        recency_score = max(0, 100 - (candle_age * 3))

        sentiment_score = sentiment.get("score", 0.0)
        sentiment_bonus = int(sentiment_score * 30) if direction == "BUY" else int(-sentiment_score * 30)
        confidence = min(97, max(45, recency_score + sentiment_bonus))
        time_remaining = max(120, 900 - (candle_age * 60))

        fmt = f"{{:.{decimals}f}}"
        return {
            "id":            f"{pair}_{int(signal_idx.timestamp())}",
            "pair":          pair,
            "category":      category,
            "direction":     direction,
            "entry":         fmt.format(entry),
            "tp":            fmt.format(tp),
            "sl":            fmt.format(sl),
            "confidence":    confidence,
            "timeRemaining": time_remaining,
            "sentiment":     sentiment,
            "currentPrice":  fmt.format(current_price),
            "timestamp":     int(signal_idx.timestamp()),
        }
    except Exception as e:
        print(f"Signal build error for {pair}: {e}")
        return None


def _scan_one(pair: str, meta: dict) -> dict | None:
    """Worker function: download + analyse one symbol. Runs in thread pool."""
    ticker = meta["ticker"]
    try:
        data = yf.download(ticker, period="5d", interval="15m",
                           progress=False, auto_adjust=True)
        if data.empty:
            return None
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        sentiment = analyze_sentiment(ticker)
        return build_signal_from_df(pair, df, sentiment)
    except Exception as e:
        print(f"Scan error {pair} ({ticker}): {e}")
        return None


@app.get("/api/signals")
def get_signals(category: str = ""):
    """Parallel scan of all (or filtered) symbols. Returns active BOS signals."""
    catalog = {
        k: v for k, v in SYMBOL_CATALOG.items()
        if not category or v["category"].lower() == category.lower()
    }

    results = []
    # 10 parallel workers — keeps Railway within memory limits
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(_scan_one, pair, meta): pair
                   for pair, meta in catalog.items()}
        for future in as_completed(futures):
            sig = future.result()
            if sig:
                results.append(sig)

    results.sort(key=lambda s: s["confidence"], reverse=True)
    return {"signals": results, "count": len(results)}


@app.get("/api/symbols")
def get_symbols():
    """Return full symbol catalog grouped by category."""
    grouped: dict[str, list] = {}
    for pair, meta in SYMBOL_CATALOG.items():
        cat = meta["category"]
        grouped.setdefault(cat, []).append({
            "pair":     pair,
            "ticker":   meta["ticker"],
            "decimals": meta["decimals"],
        })
    return {"categories": grouped, "total": len(SYMBOL_CATALOG)}


@app.get("/api/health")
def health():
    return {"status": "ok", "symbols": len(SYMBOL_CATALOG), "firebase": firebase_ready}


if __name__ == "__main__":
    print(f"Starting Trading Bot API — {len(SYMBOL_CATALOG)} symbols loaded")
    uvicorn.run(app, host="0.0.0.0", port=8000)
