import os
import requests
import json
from datetime import datetime, timezone
import yfinance as yf
import matplotlib.pyplot as plt
from google import genai

from news_utils import fetch_rss_feeds

# ==========================================
# CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL_MARKET = os.getenv("DISCORD_WEBHOOK_URL_MARKET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"

# Tickers to track
INDICES = {
    "^GSPC": "S&P 500",
    "^IXIC": "Nasdaq",
    "^DJI": "Dow Jones",
    "^RUT": "Russell 2000 (Small Cap)",
}
MACRO_SIGNALS = {
    "^VIX": "VIX (Fear Index)",
    "^TNX": "US 10Y Bond Yield (%)",
    "DX-Y.NYB": "US Dollar Index (DXY)",
}
STOCKS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "Nvidia",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla",
}

NEWS_CONTEXT_FEEDS = {
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "CNBC Markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "Yahoo Finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,NVDA,AAPL,MSFT,TSLA&region=US&lang=en-US",
}


def fetch_market_data():
    """Fetches daily percentage change for indices, macro indicators, and Mag 7 stocks with date alignment."""
    print("Fetching market data...")
    results = []

    # 1. Fetch ^GSPC first to establish the latest completed US equity session date
    all_equity_tickers = list(INDICES.keys()) + list(STOCKS.keys())
    equity_df = yf.download(all_equity_tickers, period="7d", progress=False)["Close"]

    ref_series = equity_df["^GSPC"].dropna() if "^GSPC" in equity_df else None
    ref_date = ref_series.index[-1] if ref_series is not None and len(ref_series) >= 1 else None

    for ticker in all_equity_tickers:
        try:
            series = equity_df[ticker].dropna()
            # Align to S&P 500 latest completed trading date so indices/stocks stay on the same day
            if ref_date is not None:
                series = series[series.index <= ref_date]
            if len(series) >= 2:
                latest_price = float(series.iloc[-1])
                prev_price = float(series.iloc[-2])
                pct_change = ((latest_price - prev_price) / prev_price) * 100
                date_str = series.index[-1].strftime("%Y-%m-%d")

                name = INDICES.get(ticker, STOCKS.get(ticker, ticker))
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "price": latest_price,
                    "change": pct_change,
                    "date": date_str,
                    "group": "index" if ticker in INDICES else "stock",
                })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # 2. Fetch Macro Signals individually to preserve exact date for each (VIX, 10Y Yield, DXY)
    for ticker, name in MACRO_SIGNALS.items():
        try:
            hist = yf.Ticker(ticker).history(period="7d")["Close"].dropna()
            if len(hist) >= 2:
                latest_price = float(hist.iloc[-1])
                prev_price = float(hist.iloc[-2])
                pct_change = ((latest_price - prev_price) / prev_price) * 100
                date_str = hist.index[-1].strftime("%Y-%m-%d")
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "price": latest_price,
                    "change": pct_change,
                    "date": date_str,
                    "group": "macro",
                })
        except Exception as e:
            print(f"Error processing macro {ticker}: {e}")

    indices_data = [r for r in results if r["group"] == "index"]
    macro_data = [r for r in results if r["group"] == "macro"]
    stocks_data = sorted([r for r in results if r["group"] == "stock"], key=lambda x: x["change"], reverse=True)

    return indices_data + macro_data + stocks_data


def fetch_top_headlines_context() -> str:
    """Fetches top recent headlines to give Gemini real-world news catalysts for market movement."""
    try:
        articles = fetch_rss_feeds(NEWS_CONTEXT_FEEDS, max_per_feed=4, timeout=8)
        if not articles:
            return ""
        lines = []
        for idx, art in enumerate(articles[:10], start=1):
            lines.append(f"{idx}. [{art['source']}] {art['title']}")
        return "\n".join(lines)
    except Exception as e:
        print(f"Notice: Could not fetch headlines context ({e})")
        return ""


def generate_ai_insight(data: list, headlines_context: str = "") -> str:
    """Uses Gemini to generate a professional market direction recap based on numbers + live news."""
    if not GEMINI_API_KEY:
        return "⚠️ (ไม่ได้ใส่ API Key ของ Gemini จึงไม่มีบทวิเคราะห์ AI)"

    print("Generating AI Market Insight (Numbers + News Catalysts)...")
    data_str = "\n".join(
        [f"{item['name']}: {item['change']:+.2f}% (Price: {item['price']:.2f} | Date: {item['date']})" for item in data]
    )

    news_block = (
        f"\nLatest Top Market Headlines (for real-world catalysts):\n{headlines_context}\n"
        if headlines_context
        else "\n(No live headlines fetched; focus on price & macro structure.)\n"
    )

    prompt = f"""
You are a senior US Equity & Macro Market Strategist.
I have the latest daily performance for major US equity indices, Macro Regime indicators (VIX, 10Y Treasury Yield, US Dollar DXY), and the Magnificent 7 tech stocks:
{data_str}
{news_block}

Please write a sharp, insightful daily market recap & direction outlook in THAI (3 short, structured paragraphs):
1. **ภาพรวมและทิศทางตลาด (Market Regime & Direction):** สรุปอารมณ์ตลาดว่าเป็น Risk-On, Risk-Off หรือ Mixed โดยเชื่อมโยง S&P 500, Nasdaq, VIX และ **US 10Y Bond Yield (^TNX) / DXY** (เช่น ยีลด์พุ่งกดดันหุ้น Tech หรือยีลด์ลงหนุนตลาด)
2. **สาเหตุที่ตลาดขับเคลื่อนจริง (Catalysts from News):** อ้างอิงข่าวจริงจากหัวข้อข่าวที่แนบมา เพื่ออธิบายว่าทำไมตลาดและหุ้น Mag 7 ตัวเด่นๆ ถึงปรับตัวขึ้นหรือลงในรอบล่าสุด
3. **มุมมองกลยุทธ์ถัดไป (Actionable Outlook):** แนะนำสิ่งที่นักลงทุนควรจับตาต่อในคืนถัดไปหรือแนวรับ-โมเมนตัมที่ต้องระวัง

Use markdown bolding and emojis cleanly. Avoid generic filler words.
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


def generate_chart(data: list) -> str:
    """Generates a premium-looking dark mode bar chart of the daily Mag 7 changes."""
    print("Generating chart...")

    chart_data = [r for r in data if r["group"] == "stock"]
    names = [r["ticker"] for r in chart_data]
    changes = [r["change"] for r in chart_data]

    colors = ["#00e676" if c >= 0 else "#ff1744" for c in changes]

    plt.figure(figsize=(10, 5), facecolor="#1e1e2e")
    ax = plt.gca()
    ax.set_facecolor("#1e1e2e")

    bars = plt.bar(names, changes, color=colors, width=0.6, edgecolor="none")
    plt.axhline(0, color="#cdd6f4", linewidth=1.5, alpha=0.5)

    ref_date = chart_data[0]["date"] if chart_data else ""
    plt.title(
        f"Magnificent 7 - Daily Performance ({ref_date})",
        color="#cdd6f4",
        fontsize=15,
        pad=20,
        fontweight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="x", colors="#cdd6f4", labelsize=11)
    ax.tick_params(axis="y", colors="#a6adc8", labelsize=10)
    plt.grid(axis="y", color="#313244", linestyle="--", alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        offset = 0.1 if yval >= 0 else -0.4
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + offset,
            f"{yval:+.2f}%",
            ha="center",
            va="bottom" if yval >= 0 else "top",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    chart_filename = "market_chart.png"
    plt.savefig(chart_filename, dpi=120, bbox_inches="tight", facecolor="#1e1e2e", edgecolor="none")
    plt.close()

    return chart_filename


def format_text_summary(data: list, ai_insight: str) -> str:
    """Creates a comprehensive markdown formatted summary."""
    summary = "### 🧠 AI Market Direction & Catalyst\n"
    summary += f"{ai_insight}\n\n"

    indices = [r for r in data if r["group"] == "index"]
    macros = [r for r in data if r["group"] == "macro"]
    stocks = [r for r in data if r["group"] == "stock"]

    eq_date = indices[0]["date"] if indices else ""
    summary += f"### 📊 ดัชนีหลักตลาดสหรัฐฯ (ปิดวันที่ {eq_date})\n"
    for item in indices:
        emoji = "🟩" if item["change"] >= 0 else "🟥"
        summary += f"{emoji} **{item['name']}**: {item['price']:,.2f} ({item['change']:+.2f}%)\n"

    if macros:
        summary += "\n### 🧭 ตัวชี้วัด Macro & ความเสี่ยง (Regime Signals)\n"
        for item in macros:
            if item["ticker"] == "^VIX":
                emoji = "🚨" if item["change"] >= 0 else "😌"
            elif item["ticker"] == "^TNX":
                emoji = "⚠️" if item["change"] > 0 else "🟢"
            else:
                emoji = "💵"
            summary += f"{emoji} **{item['name']}**: {item['price']:,.2f} ({item['change']:+.2f}%) `[{item['date']}]`\n"

    summary += "\n### 🔥 หุ้นเทคยักษ์ใหญ่ (Magnificent 7)\n"
    for item in stocks:
        emoji = "🟩" if item["change"] >= 0 else "🟥"
        summary += f"{emoji} **{item['ticker']}**: {item['price']:,.2f} ({item['change']:+.2f}%)\n"

    return summary[:4000]


def send_to_discord(summary: str, chart_filename: str):
    """Sends the text summary and chart image to Discord."""
    if not DISCORD_WEBHOOK_URL_MARKET:
        print("Webhook URL for market missing.")
        return

    print("Sending to Discord...")
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    payload_json = {
        "content": "📈 **สรุปภาวะตลาดหุ้นสหรัฐฯ & ทิศทาง Macro ประจำวัน (Daily Market Brief)** 🇺🇸",
        "embeds": [
            {
                "description": summary,
                "color": 16766720,  # Gold color
                "image": {
                    "url": "attachment://" + chart_filename
                },
                "footer": {
                    "text": "🤖 AI Direction Analysis (Macro + News Catalysts) • yfinance"
                },
                "timestamp": now_iso,
            }
        ],
    }

    with open(chart_filename, "rb") as f:
        response = requests.post(
            DISCORD_WEBHOOK_URL_MARKET,
            data={"payload_json": json.dumps(payload_json)},
            files={"file": (chart_filename, f, "image/png")},
            timeout=20,
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
    headlines = fetch_top_headlines_context()
    ai_insight = generate_ai_insight(market_data, headlines)
    summary_text = format_text_summary(market_data, ai_insight)

    send_to_discord(summary_text, chart_file)


if __name__ == "__main__":
    main()
