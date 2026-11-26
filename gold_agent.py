import os
import json
from datetime import datetime, timezone
import yfinance as yf
from google import genai
from google.genai import types
from pydantic import BaseModel

from news_utils import (
    load_sent_cache,
    save_sent_cache,
    format_article_links,
    fetch_rss_feeds,
    send_discord_embeds,
)

# ==========================================
# CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL_GOLD = os.getenv("DISCORD_WEBHOOK_URL_GOLD")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CACHE_FILE = os.path.join(os.path.dirname(__file__), "sent_gold_cache.json")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = "gemini-2.5-flash"

GOLD_MACRO_TICKERS = {
    "GC=F": "Gold Futures ($/oz)",
    "DX-Y.NYB": "US Dollar Index (DXY)",
    "^TNX": "US 10Y Bond Yield (%)",
    "SI=F": "Silver Futures ($/oz)",
}

# Verified Live Gold & Macro RSS Feeds
GOLD_RSS_FEEDS = {
    "Yahoo Gold & Macro": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F,DX-Y.NYB,^TNX,GLD&region=US&lang=en-US",
    "Investing.com Commodities": "https://www.investing.com/rss/news_11.rss",
    "FXStreet Macro": "https://www.fxstreet.com/rss/news",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_monetary.xml",
}


class GoldNewsItem(BaseModel):
    title: str
    analysis: str
    sentiment: str
    original_index: int


class GoldReport(BaseModel):
    macro_summary: str
    market_bias: str
    curated_news: list[GoldNewsItem]


def fetch_gold_macro_data() -> list:
    """Fetches latest prices & daily % changes for Gold, DXY, 10Y Bond Yield, and Silver."""
    print("Fetching Gold & Macro numbers from yfinance...")
    results = []
    for ticker, name in GOLD_MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="7d")["Close"].dropna()
            if len(hist) >= 2:
                latest = float(hist.iloc[-1])
                prev = float(hist.iloc[-2])
                pct = ((latest - prev) / prev) * 100
                date_str = hist.index[-1].strftime("%Y-%m-%d")
                results.append({
                    "ticker": ticker,
                    "name": name,
                    "price": latest,
                    "change": pct,
                    "date": date_str,
                })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    return results


def analyze_gold_market(macro_data: list, news_list: list) -> dict:
    """Uses Gemini to synthesize Gold drivers (DXY, Bond Yield, Fed, Geopolitics) + top news."""
    macro_str = "\n".join(
        [f"- {m['name']}: {m['price']:,.2f} ({m['change']:+.2f}% | Date: {m['date']})" for m in macro_data]
    ) or "No live price data."

    news_text = ""
    for i, news in enumerate(news_list):
        news_text += (
            f"[{i + 1}] Source: {news['source']} | Published: {news['published']} | "
            f"Title: {news['title']} | Summary: {news['summary']}\n"
        )

    prompt = f"""
You are a Senior Global Commodities & Gold (XAU/USD) Macro Strategist.
Analyze the current global Gold situation using BOTH the live market numbers and the news headlines below.

Live Market Numbers:
{macro_str}

Recent Gold / Dollar / Fed / Macro Headlines:
{news_text}

Your tasks:
1. For 'macro_summary': Write a crisp, insightful overview in THAI (2-3 short paragraphs) explaining:
   - What exact forces (US Dollar DXY, 10Y Treasury Yield, Fed interest rate expectations, Geopolitics / Safe-Haven demand) are driving Gold prices UP or DOWN right now.
   - How Gold's current movement reflects on the US Stock Market (e.g., Risk-Off Safe-Haven flight vs. Inflation/Yield pressure).
2. For 'market_bias': Return one of: "BULLISH", "BEARISH", or "NEUTRAL" for Gold right now.
3. For 'curated_news': Select 4 to 6 MOST RELEVANT news items specifically impacting Gold, US Dollar, Bond Yields, or Fed Policy. For each item:
   - Translate 'title' to THAI with a gold/macro emoji at the start.
   - Write 'analysis' in THAI with double newlines (\\n\\n) between sections and Markdown bold (**text**):
     - 📖 เนื้อหาข่าว (2-3 sentences explaining the event)
     - 🎯 เอฟเฟกต์ต่อราคาทองคำ (Why this pushes Gold up or presses it down)
     - 💡 มุมมองนักลงทุน (Takeaway for Gold & US equities)
   - Set 'sentiment' ("POSITIVE", "NEGATIVE", or "NEUTRAL" specifically from Gold's perspective).
"""
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GoldReport,
                temperature=0.4,
            ),
        )
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"Error in Gold AI analysis: {e}")
        return {}


def build_gold_embeds(macro_data: list, report: dict, raw_news: list) -> list:
    """Constructs the Gold Macro Overview embed followed by curated news embeds."""
    colors = {
        "POSITIVE": 5763719,   # Green
        "BULLISH": 16766720,   # Gold
        "NEGATIVE": 15548997,  # Red
        "BEARISH": 15548997,   # Red
        "NEUTRAL": 3447003,    # Blue
    }
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    embeds = []

    # 1. Macro Overview Card
    macro_lines = []
    for m in macro_data:
        if m["ticker"] in ("GC=F", "SI=F"):
            emoji = "🟡" if m["change"] >= 0 else "🔻"
        elif m["ticker"] == "^TNX":
            emoji = "⚠️" if m["change"] > 0 else "🟢"
        else:
            emoji = "💵"
        macro_lines.append(f"{emoji} **{m['name']}**: {m['price']:,.2f} ({m['change']:+.2f}%) `[{m['date']}]`")

    macro_summary = (report.get("macro_summary") or "ไม่มีบทสรุปภาพรวม").replace("\\n", "\n")
    bias = (report.get("market_bias") or "NEUTRAL").upper()
    overview_desc = (
        "### 📊 ตัวเลขขับเคลื่อนราคาทองคำ (Live Drivers)\n"
        + "\n".join(macro_lines)
        + "\n\n### 🧠 วิเคราะห์เอฟเฟกต์ทองคำโลก & สะท้อนตลาดหุ้น\n"
        + macro_summary
    )

    embeds.append({
        "title": f"🪙 ภาพรวมสถานการณ์ทองคำโลก (Gold Bias: {bias})",
        "description": overview_desc[:4000],
        "color": colors.get(bias, 16766720),
        "footer": {
            "text": "🤖 วิเคราะห์โดย Gemini AI • AlphaZone Gold & Macro Intelligence"
        },
        "timestamp": now_iso,
    })

    # 2. Curated News Cards
    for item in report.get("curated_news", []):
        orig_idx = int(item.get("original_index", 1)) - 1
        if 0 <= orig_idx < len(raw_news):
            original_link = raw_news[orig_idx]["link"]
            source_name = raw_news[orig_idx]["source"]
        else:
            original_link = "https://finance.yahoo.com/quote/GC=F/"
            source_name = "Commodities News"

        sentiment = item.get("sentiment", "NEUTRAL").upper()
        analysis_text = item.get("analysis", "ไม่มีเนื้อหา").replace("\\n", "\n")
        analysis_text = analysis_text.replace("🎯", "\n🎯").replace("💡", "\n💡").replace("\n\n\n", "\n\n")
        link_line = format_article_links(source_name, original_link)

        embeds.append({
            "title": item.get("title", "ข่าวทองคำ")[:250],
            "description": (analysis_text + f"\n\n{link_line}")[:4000],
            "color": colors.get(sentiment, 3447003),
            "footer": {
                "text": "🤖 วิเคราะห์ผลกระทบต่อทองคำโดย Gemini AI"
            },
            "timestamp": now_iso,
        })

    return embeds


def main():
    if not DISCORD_WEBHOOK_URL_GOLD or not GEMINI_API_KEY:
        print("Skipping Gold Agent: DISCORD_WEBHOOK_URL_GOLD or GEMINI_API_KEY not configured yet.")
        return

    print("1. Loading cache & fetching Gold / Macro news...")
    sent_cache = load_sent_cache(CACHE_FILE)
    print(f"   Sent gold cache has {len(sent_cache)} items.")

    macro_data = fetch_gold_macro_data()
    raw_news = fetch_rss_feeds(GOLD_RSS_FEEDS, sent_cache=sent_cache, max_per_feed=15)
    print(f"   Found {len(raw_news)} fresh gold/macro articles.")

    if not raw_news:
        print("No new unseen gold/macro articles since last run.")
        return

    print("2. Analyzing Gold market drivers & news with Gemini...")
    report = analyze_gold_market(macro_data, raw_news)
    if not report or not report.get("curated_news"):
        print("Failed to generate Gold report.")
        return

    print("3. Sending Gold report to Discord (#gold-news)...")
    embeds = build_gold_embeds(macro_data, report, raw_news)
    header = "🪙 **อัปเดตสถานการณ์ทองคำโลก & ปัจจัยขับเคลื่อน (Gold-News Brief)** ✨"
    if send_discord_embeds(DISCORD_WEBHOOK_URL_GOLD, header, embeds, chunk_size=2):
        for item in report.get("curated_news", []):
            orig_idx = int(item.get("original_index", 1)) - 1
            if 0 <= orig_idx < len(raw_news):
                sent_cache.add(raw_news[orig_idx]["link"])
        save_sent_cache(CACHE_FILE, sent_cache)
        print("4. Updated sent gold cache successfully.")


if __name__ == "__main__":
    main()
