import os
import requests
import json
from datetime import datetime, timezone
import yfinance as yf
import matplotlib.pyplot as plt
import io
from google import genai

# ==========================================
# CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL_MARKET = os.getenv("DISCORD_WEBHOOK_URL_MARKET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-2.5-flash'

# Tickers to track
INDICES = {
    "^GSPC": "S&P 500", 
    "^IXIC": "Nasdaq", 
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000 (Small Cap)",
    "^VIX": "VIX (Fear Index)"
}
STOCKS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "GOOGL": "Alphabet", 
    "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla"
}

def fetch_market_data():
    """Fetches daily percentage change for indices and stocks."""
    print("Fetching market data...")
    results = []
    
    all_tickers = list(INDICES.keys()) + list(STOCKS.keys())
    data = yf.download(all_tickers, period="5d", progress=False)['Close']
    
    for ticker in all_tickers:
        try:
            ticker_data = data[ticker].dropna()
            if len(ticker_data) >= 2:
                latest_price = ticker_data.iloc[-1]
                prev_price = ticker_data.iloc[-2]
                pct_change = ((latest_price - prev_price) / prev_price) * 100
                
                name = INDICES.get(ticker, STOCKS.get(ticker, ticker))
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "price": latest_price,
                    "change": pct_change,
                    "is_index": ticker in INDICES
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
    indices_data = [r for r in results if r["is_index"]]
    stocks_data = [r for r in results if not r["is_index"]]
    stocks_data = sorted(stocks_data, key=lambda x: x["change"], reverse=True)
    
    return indices_data + stocks_data

def generate_ai_insight(data):
    """Uses Gemini to generate a professional market recap based on the raw data."""
    if not GEMINI_API_KEY:
        return "⚠️ (ไม่ได้ใส่ API Key ของ Gemini จึงไม่มีบทวิเคราะห์ AI)"
        
    print("Generating AI Market Insight...")
    data_str = "\n".join([f"{item['name']}: {item['change']:+.2f}% (Price: {item['price']:.2f})" for item in data])
    
    prompt = f"""
You are a world-class US Stock Market Analyst.
I have the daily percentage changes for major US indices and the Magnificent 7 tech stocks:
{data_str}

Please write a brief, professional daily market recap in THAI (2-3 short paragraphs).
1. Summarize the overall market mood (Bullish, Bearish, Mixed) based on the indices (especially S&P 500, Nasdaq, Dow Jones, and VIX). Note: If VIX is up, fear is up.
2. Highlight the key movers among the Magnificent 7.
3. Provide a brief actionable insight for investors.
Use markdown, emojis, and make it very engaging. Do not use generic filler words.
"""
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error in AI market analysis: {e}")
        return "⚠️ (AI เกิดข้อผิดพลาดในการวิเคราะห์ชั่วคราว)"

def generate_chart(data):
    """Generates a premium-looking bar chart of the daily changes."""
    print("Generating chart...")
    
    chart_data = [r for r in data if not r["is_index"]]
    names = [r["ticker"] for r in chart_data]
    changes = [r["change"] for r in chart_data]
    
    colors = ['#00e676' if c >= 0 else '#ff1744' for c in changes] # Brighter neon colors
    
    # Premium Dark Mode setup
    plt.figure(figsize=(10, 5), facecolor='#1e1e2e')
    ax = plt.gca()
    ax.set_facecolor('#1e1e2e')
    
    bars = plt.bar(names, changes, color=colors, width=0.6, edgecolor='none')
    plt.axhline(0, color='#cdd6f4', linewidth=1.5, alpha=0.5)
    
    plt.title('Magnificent 7 - Daily Performance', color='#cdd6f4', fontsize=16, pad=20, fontweight='bold')
    
    # Remove borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    # Style ticks
    ax.tick_params(axis='x', colors='#cdd6f4', labelsize=11)
    ax.tick_params(axis='y', colors='#a6adc8', labelsize=10)
    plt.grid(axis='y', color='#313244', linestyle='--', alpha=0.7)
    
    # Add floating text labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (0.1 if yval >= 0 else -0.4),
                 f"{yval:+.2f}%", ha='center', va='bottom' if yval >= 0 else 'top', 
                 color='white', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    chart_filename = "market_chart.png"
    plt.savefig(chart_filename, dpi=120, bbox_inches='tight', facecolor='#1e1e2e', edgecolor='none')
    plt.close()
    
    return chart_filename

def format_text_summary(data, ai_insight):
    """Creates a comprehensive markdown formatted summary."""
    summary = "### 🧠 AI Market Insight (บทวิเคราะห์จาก AI)\n"
    summary += f"{ai_insight}\n\n"
    
    summary += "### 📊 สรุปตัวเลขดัชนีหลัก (Market Indices)\n"
    for item in data:
        if item["is_index"]:
            emoji = "🟩" if item["change"] >= 0 else "🟥"
            if item["ticker"] == "^VIX":
                emoji = "🚨" if item["change"] >= 0 else "😌" # VIX is inverted logic
            summary += f"{emoji} **{item['name']}**: {item['price']:,.2f} ({item['change']:+.2f}%)\n"
            
    summary += "\n### 🔥 หุ้นเทคยักษ์ใหญ่ (Magnificent 7)\n"
    for item in data:
        if not item["is_index"]:
            emoji = "🟩" if item["change"] >= 0 else "🟥"
            summary += f"{emoji} **{item['ticker']}**: {item['price']:,.2f} ({item['change']:+.2f}%)\n"
            
    return summary

def send_to_discord(summary, chart_filename):
    """Sends the text summary and chart image to Discord."""
    if not DISCORD_WEBHOOK_URL_MARKET:
        print("Webhook URL for market missing.")
        return
        
    print("Sending to Discord...")
    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    payload_json = {
        "content": "📈 **สรุปภาวะตลาดหุ้นสหรัฐฯ ประจำวัน (Daily Market Brief)** 🇺🇸",
        "embeds": [
            {
                "description": summary,
                "color": 16766720, # Gold color
                "image": {
                    "url": "attachment://" + chart_filename
                },
                "footer": {
                    "text": "🤖 AI Analysis & Market Data by yfinance"
                },
                "timestamp": now_iso
            }
        ]
    }
    
    with open(chart_filename, "rb") as f:
        response = requests.post(
            DISCORD_WEBHOOK_URL_MARKET,
            data={"payload_json": json.dumps(payload_json)},
            files={"file": (chart_filename, f, "image/png")}
        )
        
    if response.status_code in [200, 204]:
        print("Successfully sent market summary to Discord!")
    else:
        print(f"Error sending to Discord: {response.status_code} - {response.text}")

def main():
    market_data = fetch_market_data()
    if not market_data:
        print("No market data found.")
        return
        
    chart_file = generate_chart(market_data)
    ai_insight = generate_ai_insight(market_data)
    summary_text = format_text_summary(market_data, ai_insight)
    
    send_to_discord(summary_text, chart_file)

if __name__ == "__main__":
    main()
