import os
import feedparser
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel
from datetime import datetime, timezone
import json
import re

# ==========================================
# CONFIGURATION
# ==========================================
# Use environment variables for security (especially for GitHub Actions)
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not DISCORD_WEBHOOK_URL or not GEMINI_API_KEY:
    raise ValueError("Missing API Keys! Please set DISCORD_WEBHOOK_URL and GEMINI_API_KEY as environment variables.")

# Configure Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-2.5-flash'

class CuratedNewsItem(BaseModel):
    title: str
    analysis: str
    sentiment: str
    original_index: int

# RSS Feeds to monitor
RSS_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssurl",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"
}

def fetch_rss_headlines():
    """Fetches top headlines from defined RSS feeds."""
    all_news = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            # Take top 25 from each source to give AI enough context
            for entry in feed.entries[:25]:
                # Clean up summary HTML tags if any
                raw_summary = entry.get("summary", "")
                clean_summary = re.sub(r'<[^>]+>', '', raw_summary)[:400]
                
                all_news.append({
                    "source": source,
                    "title": entry.title,
                    "link": entry.link,
                    "published": entry.get("published", ""),
                    "summary": clean_summary
                })
        except Exception as e:
            print(f"Error fetching from {source}: {e}")
    return all_news

def curate_and_summarize_news(news_list):
    """Uses Gemini to pick top news and summarize them in Thai."""
    news_text = ""
    for i, news in enumerate(news_list):
        news_text += f"[{i+1}] Source: {news['source']} | Title: {news['title']} | Summary: {news['summary']}\n"

    prompt = f"""
You are a world-class US Stock Market Analyst.
Below is a list of recent financial news headlines and short summaries.
Your task:
1. Select 8 to 12 MOST IMPORTANT news items that have the biggest impact on the US macro economy, investment trends, or major specific tech stocks (like AAPL, TSLA, NVDA). If there are major breaking news, you can select up to 15 items.
2. For each selected news, TRANSLATE the title into THAI language.
3. Provide a DETAILED and COMPREHENSIVE analysis translated into THAI language. Do not make it too short.
4. Format the 'analysis' field to be engaging and easy to read using bullet points. Use standard newline characters (\\n) for line breaks. Include:
   - 📖 เนื้อหาข่าว (Detailed summary of what happened - at least 3-4 sentences)
   - 🎯 ประเด็นสำคัญ (Key takeaway)
   - 💡 ผลกระทบต่อนักลงทุน (Detailed impact on markets and investors)
5. Determine the sentiment of the news (POSITIVE, NEGATIVE, or NEUTRAL) for the 'sentiment' field.
6. Provide a suitable emoji at the beginning of the 'title' field.

Here is the news list:
{news_text}
"""
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[CuratedNewsItem],
                temperature=0.4
            )
        )
        result_text = response.text.strip()
        curated_news = json.loads(result_text)
        return curated_news
    except Exception as e:
        print(f"Error in AI curation: {e}")
        return []

def send_to_discord(curated_news, original_news_list):
    """Sends the formatted news to Discord Webhook using Embeds in chunks of 3."""
    import time
    if not curated_news:
        print("No news to send.")
        return

    # Color mapping based on sentiment
    colors = {
        "POSITIVE": 5763719,  # Green
        "NEGATIVE": 15548997, # Red
        "NEUTRAL": 3447003    # Blue
    }

    embeds = []
    # Get current time for footer in strict ISO8601 for Discord
    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    for item in curated_news:
        orig_idx = int(item.get("original_index", 1)) - 1
        if 0 <= orig_idx < len(original_news_list):
            original_link = original_news_list[orig_idx]["link"]
            source_name = original_news_list[orig_idx]["source"]
        else:
            original_link = "https://finance.yahoo.com/"
            source_name = "Unknown"

        sentiment = item.get("sentiment", "NEUTRAL").upper()
        embed_color = colors.get(sentiment, 3447003)

        # Fix literal \n outputted by LLM
        analysis_text = item.get("analysis", "ไม่มีเนื้อหา").replace("\\n", "\n")

        embed = {
            "title": item.get("title", "ไม่มีหัวข้อ")[:250],
            "description": (analysis_text + f"\n\n📰 **[อ่านข่าวต้นฉบับ ({source_name})]({original_link})**")[:4000],
            "color": embed_color,
            "footer": {
                "text": "🤖 วิเคราะห์โดย Gemini AI"
            },
            "timestamp": now_iso
        }
        embeds.append(embed)

    # Discord allows max 10 embeds per message, and 6000 chars total.
    # To prevent 500 Internal Server Errors from payload size or rate limits, we use chunk_size = 2.
    chunk_size = 2
    for i in range(0, len(embeds), chunk_size):
        chunk = embeds[i:i + chunk_size]
        payload = {
            "content": "✨ **อัปเดตข่าวหุ้นสหรัฐฯ น่าจับตามองประจำวัน!** 🇺🇸" if i == 0 else "",
            "embeds": chunk
        }

        try:
            response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
            response.raise_for_status()
            print(f"Successfully sent chunk {i//chunk_size + 1} to Discord!")
            time.sleep(2) # Add delay to avoid Discord rate limits
        except Exception as e:
            print(f"Error sending chunk to Discord: {e}")

def main():
    print("1. Fetching news...")
    raw_news = fetch_rss_headlines()
    print(f"   Found {len(raw_news)} articles.")
    
    if not raw_news:
        print("Failed to fetch news.")
        return

    print("2. Sending to AI for curation and summarization...")
    curated_news = curate_and_summarize_news(raw_news)
    
    if curated_news:
        print("3. AI Processing complete. Sending to Discord...")
        send_to_discord(curated_news, raw_news)
    else:
        print("Failed to curate news. Check API key and quota.")

if __name__ == "__main__":
    main()
