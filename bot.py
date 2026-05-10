import yfinance as yf
import pandas as pd
import numpy as np
import numpy as np
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
# SIGNAL SCANNER — /api/signals
# ---------------------------------------------------------

# Symbol map: display pair → yfinance ticker
SIGNAL_SYMBOLS = {
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPJPY": "GBPJPY=X",
    "US30":   "YM=F",
    "BTCUSD": "BTC-USD",
    "GBPUSD": "GBPUSD=X",
}

def pips_away(current_price: float, pct: float) -> float:
    return round(current_price * (1 + pct / 100), 5)

def build_signal_from_df(pair: str, df: pd.DataFrame, sentiment: dict) -> dict | None:
    """Derive a structured signal from the last BOS event in the dataframe."""
    try:
        df['PSAR'] = calculate_psar(df)
        df = calculate_basic_smc(df)

        # Find the most recent BOS signal
        last_buy_idx  = df[df['BOS_Bullish']].index.max()
        last_sell_idx = df[df['BOS_Bearish']].index.max()

        # Determine which is more recent
        direction = None
        signal_idx = None

        if pd.isna(last_buy_idx) and pd.isna(last_sell_idx):
            return None

        if pd.isna(last_buy_idx):
            direction, signal_idx = "SELL", last_sell_idx
        elif pd.isna(last_sell_idx):
            direction, signal_idx = "BUY", last_buy_idx
        else:
            if last_buy_idx > last_sell_idx:
                direction, signal_idx = "BUY", last_buy_idx
            else:
                direction, signal_idx = "SELL", last_sell_idx

        row = df.loc[signal_idx]
        current_price = float(df.iloc[-1]['Close'])
        entry = float(row['Close'])

        # TP = 1.5% move in signal direction, SL = 0.7% against
        if direction == "BUY":
            tp = pips_away(entry,  1.5)
            sl = pips_away(entry, -0.7)
        else:
            tp = pips_away(entry, -1.5)
            sl = pips_away(entry,  0.7)

        # Confidence: recency (how many candles ago) + sentiment alignment
        candle_age = len(df) - df.index.get_loc(signal_idx)
        recency_score = max(0, 100 - (candle_age * 3))   # -3 per candle

        sentiment_score = sentiment.get("score", 0.0)
        if direction == "BUY":
            sentiment_bonus = int(sentiment_score * 30)   # up to +30 if bullish
        else:
            sentiment_bonus = int(-sentiment_score * 30)  # up to +30 if bearish

        confidence = min(97, max(45, recency_score + sentiment_bonus))

        # Time remaining estimate: 15-min candle = 900s window; decays with age
        time_remaining = max(120, 900 - (candle_age * 60))

        # Format price to match pair type
        decimals = 2 if pair in ("XAUUSD", "US30", "BTCUSD") else 5
        fmt = f"{{:.{decimals}f}}"

        return {
            "id":            f"{pair}_{int(signal_idx.timestamp())}",
            "pair":          pair,
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


@app.get("/api/signals")
def get_signals():
    """Scan all supported pairs and return active BOS signals."""
    results = []
    for pair, ticker in SIGNAL_SYMBOLS.items():
        try:
            data = yf.download(ticker, period="5d", interval="15m", progress=False)
            if data.empty:
                continue

            df = data.copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            sentiment = analyze_sentiment(ticker)
            signal = build_signal_from_df(pair, df, sentiment)

            if signal:
                results.append(signal)
        except Exception as e:
            print(f"Scan error for {pair} ({ticker}): {e}")

    # Sort by confidence descending
    results.sort(key=lambda s: s["confidence"], reverse=True)
    return {"signals": results, "count": len(results)}


@app.get("/api/health")
def health():
    return {"status": "ok", "firebase": firebase_ready}


if __name__ == "__main__":
    print("Starting Trading Bot API on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
