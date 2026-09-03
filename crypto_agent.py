import os
import feedparser
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel
from datetime import datetime, timezone
import json
import re
import time

# ==========================================
# CONFIGURATION
# ==========================================
# We will use a new Environment Variable for Crypto Webhook
DISCORD_WEBHOOK_URL_CRYPTO = os.getenv("DISCORD_WEBHOOK_URL_CRYPTO")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# For local testing, you can temporarily hardcode if needed, but DO NOT commit to GitHub
if not DISCORD_WEBHOOK_URL_CRYPTO or not GEMINI_API_KEY:
    print("Warning: Missing API Keys. Make sure DISCORD_WEBHOOK_URL_CRYPTO and GEMINI_API_KEY are set.")

# Configure Gemini Client
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-2.5-flash'

class CuratedNewsItem(BaseModel):
    title: str
    analysis: str
    sentiment: str
    original_index: int

# Crypto RSS Feeds to monitor
RSS_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss"
}

def fetch_rss_headlines():
    """Fetches top headlines from defined Crypto RSS feeds."""
    all_news = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:25]:
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
    """Uses Gemini to pick top crypto news and summarize them in Thai."""
    news_text = ""
    for i, news in enumerate(news_list):
        news_text += f"[{i+1}] Source: {news['source']} | Title: {news['title']} | Summary: {news['summary']}\n"

    prompt = f"""
You are a world-class Cryptocurrency and Blockchain Analyst.
Below is a list of recent crypto news headlines and short summaries.
Your task:
1. Select 6 to 10 MOST IMPORTANT news items that have the biggest impact on the Crypto market (Bitcoin, Ethereum, major Altcoins, regulation, ETFs, tech upgrades).
2. For each selected news, TRANSLATE the title into THAI language.
3. Provide a DETAILED analysis translated into THAI language.
4. Format the 'analysis' field to be engaging and easy to read using bullet points (\\n). Include:
   - 📖 เนื้อหาข่าว (Detailed summary of what happened)
   - 🎯 ประเด็นสำคัญ (Key takeaway)
   - 💡 ผลกระทบต่อตลาดคริปโต (Impact on the crypto market/prices)
5. Determine the sentiment of the news (POSITIVE, NEGATIVE, or NEUTRAL) for the 'sentiment' field.
6. Provide a suitable crypto-related emoji at the beginning of the 'title' field.

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
        curated_news = json.loads(response.text.strip())
        return curated_news
    except Exception as e:
        print(f"Error in AI curation: {e}")
        return []

def send_to_discord(curated_news, original_news_list):
    """Sends the formatted news to Discord Webhook using Embeds in chunks of 2."""
    if not curated_news:
        print("No news to send.")
        return
        
    if not DISCORD_WEBHOOK_URL_CRYPTO:
        print("Discord Webhook URL for Crypto is missing. Cannot send.")
        return

    colors = {
        "POSITIVE": 5763719,  # Green
        "NEGATIVE": 15548997, # Red
        "NEUTRAL": 3447003    # Blue
    }

    embeds = []
    now_iso = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')
    
    for item in curated_news:
        orig_idx = int(item.get("original_index", 1)) - 1
        if 0 <= orig_idx < len(original_news_list):
            original_link = original_news_list[orig_idx]["link"]
            source_name = original_news_list[orig_idx]["source"]
        else:
            original_link = "https://cointelegraph.com/"
            source_name = "Unknown"

        sentiment = item.get("sentiment", "NEUTRAL").upper()
        embed_color = colors.get(sentiment, 3447003)

        embed = {
            "title": item.get("title", "ไม่มีหัวข้อ")[:250],
            "description": (item.get("analysis", "ไม่มีเนื้อหา") + f"\n\n📰 **[อ่านข่าวต้นฉบับ ({source_name})]({original_link})**")[:4000],
            "color": embed_color,
            "footer": {
                "text": "🤖 วิเคราะห์โดย Gemini AI (Crypto Agent)"
            },
            "timestamp": now_iso
        }
        embeds.append(embed)

    chunk_size = 2
    for i in range(0, len(embeds), chunk_size):
        chunk = embeds[i:i + chunk_size]
        payload = {
            "content": "🪙 **อัปเดตข่าวคริปโต (Crypto News) น่าจับตามองประจำวัน!** 🚀" if i == 0 else "",
            "embeds": chunk
        }

        try:
            response = requests.post(DISCORD_WEBHOOK_URL_CRYPTO, json=payload)
            response.raise_for_status()
            print(f"Successfully sent chunk {i//chunk_size + 1} to Discord!")
            time.sleep(2)
        except Exception as e:
            print(f"Error sending chunk to Discord: {e}")

def main():
    print("1. Fetching Crypto news...")
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
        print("Failed to curate news.")

if __name__ == "__main__":
    main()
