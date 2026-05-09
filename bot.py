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

if __name__ == "__main__":
    print("Starting Trading Bot API on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
