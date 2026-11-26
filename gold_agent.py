import os
import json
from datetime import datetime, timezone
import yfinance as yf
from google.genai import types
from pydantic import BaseModel

from news_utils import (
    load_sent_cache,
    save_sent_cache,
    format_article_links,
    fetch_rss_feeds,
    send_discord_embeds,
    create_gemini_client,
    generate_gemini_with_fallback,
)

# ==========================================
# CONFIGURATION
# ==========================================
DISCORD_WEBHOOK_URL_GOLD = os.getenv("DISCORD_WEBHOOK_URL_GOLD")
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()

CACHE_FILE = os.path.join(os.path.dirname(__file__), "sent_gold_cache.json")

if GEMINI_API_KEY:
    client = create_gemini_client(GEMINI_API_KEY)

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


def fetch_gold_macro_data() -> dict:
    """Fetches prices, multi-horizon returns, and technical indicators (EMA, RSI, Support/Resistance) for Gold."""
    print("Fetching Gold & Macro numbers + technicals from yfinance...")
    macro_list = []
    prices_map = {}

    for ticker, name in GOLD_MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(period="7d")["Close"].dropna()
            if len(hist) >= 2:
                latest = float(hist.iloc[-1])
                prev = float(hist.iloc[-2])
                pct = ((latest - prev) / prev) * 100
                date_str = hist.index[-1].strftime("%Y-%m-%d")
                prices_map[ticker] = latest
                macro_list.append({
                    "ticker": ticker,
                    "name": name,
                    "price": latest,
                    "change": pct,
                    "date": date_str,
                })
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")

    # Gold/Silver ratio
    gs_ratio = None
    if "GC=F" in prices_map and "SI=F" in prices_map and prices_map["SI=F"] > 0:
        gs_ratio = prices_map["GC=F"] / prices_map["SI=F"]

    # Deep technical calculation for GC=F (1y history)
    technicals = {}
    try:
        gc = yf.Ticker("GC=F").history(period="1y").dropna()
        if len(gc) >= 50:
            closes = gc["Close"]
            latest_close = float(closes.iloc[-1])
            pct_5d = ((latest_close - float(closes.iloc[-6])) / float(closes.iloc[-6])) * 100 if len(closes) >= 6 else 0.0
            pct_1m = ((latest_close - float(closes.iloc[-22])) / float(closes.iloc[-22])) * 100 if len(closes) >= 22 else 0.0

            ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
            ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])

            # Wilder's RSI(14)
            delta = closes.diff()
            gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
            last_loss = float(loss.iloc[-1])
            rsi14 = 100.0 if last_loss == 0 else float(100 - (100 / (1 + (float(gain.iloc[-1]) / last_loss))))

            # Classic Pivot & S1/R1 from previous completed bar + 20-Day High/Low
            ref_bar = gc.iloc[-2] if len(gc) >= 2 else gc.iloc[-1]
            p_high = float(ref_bar["High"])
            p_low = float(ref_bar["Low"])
            p_close = float(ref_bar["Close"])
            pivot = (p_high + p_low + p_close) / 3.0
            r1 = 2 * pivot - p_low
            s1 = 2 * pivot - p_high

            high_20d = float(gc["High"].tail(20).max())
            low_20d = float(gc["Low"].tail(20).min())

            technicals = {
                "price": latest_close,
                "pct_5d": pct_5d,
                "pct_1m": pct_1m,
                "ema50": ema50,
                "ema200": ema200,
                "rsi14": rsi14,
                "pivot": pivot,
                "s1": s1,
                "r1": r1,
                "low_20d": low_20d,
                "high_20d": high_20d,
                "gs_ratio": gs_ratio,
                "date": gc.index[-1].strftime("%Y-%m-%d"),
            }
    except Exception as e:
        print(f"Error computing Gold technicals: {e}")

    return {"macro": macro_list, "technicals": technicals}


def analyze_gold_market(market_info: dict, news_list: list) -> dict:
    """Uses Gemini to produce an institutional-grade Gold technical + macro + news analysis."""
    macro_data = market_info.get("macro", [])
    tech = market_info.get("technicals", {})

    macro_str = "\n".join(
        [f"- {m['name']}: {m['price']:,.2f} ({m['change']:+.2f}% | Date: {m['date']})" for m in macro_data]
    ) or "No live price data."

    if tech:
        tech_str = (
            f"- Gold Futures (GC=F) Latest: ${tech['price']:,.2f} (5D: {tech['pct_5d']:+.2f}%, 1M: {tech['pct_1m']:+.2f}%)\n"
            f"- EMA 50: ${tech['ema50']:,.2f} | EMA 200: ${tech['ema200']:,.2f}\n"
            f"- RSI(14): {tech['rsi14']:.1f}\n"
            f"- Classic Pivot: ${tech['pivot']:,.2f} | Support S1: ${tech['s1']:,.2f} | Resistance R1: ${tech['r1']:,.2f}\n"
            f"- 20-Day Range: Low ${tech['low_20d']:,.2f} – High ${tech['high_20d']:,.2f}\n"
            + (f"- Gold/Silver Ratio: {tech['gs_ratio']:.2f}\n" if tech.get("gs_ratio") else "")
        )
    else:
        tech_str = "Technicals unavailable."

    news_text = ""
    for i, news in enumerate(news_list):
        news_text += (
            f"[{i + 1}] Source: {news['source']} | Published: {news['published']} | "
            f"Title: {news['title']} | Summary: {news['summary']}\n"
        )

    prompt = f"""
You are an Institutional Global Commodities & Gold (XAU/USD) Macro Strategist.
Analyze the current global Gold market deeply using BOTH live quantitative data/technicals and recent news catalysts.

Live Macro Numbers:
{macro_str}

Gold Technical Levels (GC=F):
{tech_str}

Recent Gold / Dollar / Fed / Macro Headlines:
{news_text}

Your tasks:
1. For 'macro_summary': Write a DEEP, structured institutional briefing in THAI using these 4 clear bullet sections (separated by double newlines \\n\\n, using Markdown bold **text**):
   - 🧭 **โครงสร้างกราฟและเทคนิคัลทองคำ:** วิเคราะห์สถานะเทียบ EMA 50/200, ค่า RSI(14) (Overbought/Neutral/Oversold), แนวโน้ม 5 วันและ 1 เดือน พร้อมระบุ **แนวรับสำคัญ (S1 / 20D Low)** และ **แนวต้านสำคัญ (R1 / 20D High)** เป็นตัวเลขชัดเจน
   - ⚖️ **กลไกเอฟเฟกต์ขับเคลื่อน (Macro Catalysts):** อธิบายเจาะลึกว่า **DXY**, **US 10Y Bond Yield (^TNX)**, **ท่าที Fed**, และ **ความเสี่ยงภูมิรัฐศาสตร์** ตัวไหนกำลังดันหรือกดราคาทองอยู่ในวันนี้ (เปรียบเทียบแรงบวก vs แรงกดดัน)
   - 📈 **สัญญาณสะท้อนถึงพอร์ตหุ้นสหรัฐฯ:** อธิบายว่าความเคลื่อนไหวของทอง + Gold/Silver Ratio + บอนด์ยีลด์ กำลังส่งสัญญาณ Risk-On หรือ Risk-Off ต่อตลาดหุ้น Tech / S&P 500 อย่างไร
   - 🎯 **กลยุทธ์และจุดจับตา:** ระดับราคาที่เป็นจุดวัดใจหากเบรกเอาต์ขึ้นต่อหรือจุดตั้งรับถ้าย่อตัว
2. For 'market_bias': Return one of: "BULLISH", "BEARISH", or "NEUTRAL" for Gold right now.
3. For 'curated_news': Select 4 to 6 MOST IMPACTFUL news items specifically affecting Gold, US Dollar, Bond Yields, or Fed Policy. For each item:
   - Translate 'title' to THAI with a relevant gold/macro emoji at the start.
   - Write a DETAILED 'analysis' in THAI with double newlines (\\n\\n) between sections and Markdown bold (**text**):
     - 📖 **เนื้อหาข่าว:** สรุปรายละเอียดเหตุการณ์ ตัวเลขเศรษฐกิจ บุคคล/สถาบันสำคัญที่ปรากฏในข่าว (3-4 ประโยค)
     - 🎯 **กลไกเอฟเฟกต์ต่อราคาทองคำ:** วิเคราะห์ Cause & Effect ชัดเจนว่าข่าวนี้กระทบ DXY / อัตราดอกเบี้ยแท้จริง (Real Yield) / Safe-Haven อย่างไร และมีผล **ระยะสั้น (รายวัน) vs ระยะกลาง (เทรนด์หลัก)** อย่างไร
     - 💡 **มุมมองนักลงทุน:** คำแนะนำและข้อควรระวังสำหรับทั้งผู้ถือทองคำและนักลงทุนหุ้นสหรัฐฯ
   - Set 'sentiment' ("POSITIVE", "NEGATIVE", or "NEUTRAL" specifically from Gold's perspective).
"""
    result_text, used_model = generate_gemini_with_fallback(
        client,
        prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GoldReport,
            temperature=0.4,
        ),
    )
    print(f"   Gold analysis generated using model: {used_model}")
    return json.loads(result_text)


def build_gold_embeds(market_info: dict, report: dict, raw_news: list) -> list:
    """Constructs the deep Gold Macro + Technical Command Card followed by curated news cards."""
    colors = {
        "POSITIVE": 5763719,   # Green
        "BULLISH": 16766720,   # Gold
        "NEGATIVE": 15548997,  # Red
        "BEARISH": 15548997,   # Red
        "NEUTRAL": 3447003,    # Blue
    }
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    embeds = []

    macro_data = market_info.get("macro", [])
    tech = market_info.get("technicals", {})

    # 1. Macro & Technical Command Card
    macro_lines = []
    for m in macro_data:
        if m["ticker"] in ("GC=F", "SI=F"):
            emoji = "🟡" if m["change"] >= 0 else "🔻"
        elif m["ticker"] == "^TNX":
            emoji = "⚠️" if m["change"] > 0 else "🟢"
        else:
            emoji = "💵"
        macro_lines.append(f"{emoji} **{m['name']}**: {m['price']:,.2f} ({m['change']:+.2f}%) `[{m['date']}]`")

    tech_block = ""
    if tech:
        trend_tag = "ขาขึ้นแข็งแกร่ง (เหนือ EMA50 & EMA200)" if tech["price"] > tech["ema50"] and tech["ema50"] > tech["ema200"] else (
            "แกว่งตัว/พักฐาน" if tech["price"] > tech["ema200"] else "อ่อนแรง (ต่ำกว่า EMA200)"
        )
        rsi_val = tech["rsi14"]
        rsi_tag = "Overbought 🔥" if rsi_val >= 70 else ("Oversold ❄️" if rsi_val <= 30 else "สมดุล/มีแรงวิ่ง ✅")
        gs_str = f" | Gold/Silver Ratio: **{tech['gs_ratio']:.1f}x**" if tech.get("gs_ratio") else ""
        tech_block = (
            "\n\n### 📐 โครงสร้างเทคนิคัลทองคำ (`GC=F`)\n"
            f"• **แนวโน้ม:** {trend_tag} (5D: **{tech['pct_5d']:+.2f}%** | 1M: **{tech['pct_1m']:+.2f}%**)\n"
            f"• **EMA50:** ${tech['ema50']:,.2f} | **EMA200:** ${tech['ema200']:,.2f}\n"
            f"• **RSI(14):** **{rsi_val:.1f}** ({rsi_tag}){gs_str}\n"
            f"• **แนวรับ (Support):** S1 **${tech['s1']:,.2f}** | โลว์ 20 วัน **${tech['low_20d']:,.2f}**\n"
            f"• **แนวต้าน (Resistance):** R1 **${tech['r1']:,.2f}** | ไฮ 20 วัน **${tech['high_20d']:,.2f}**"
        )

    macro_summary = (report.get("macro_summary") or "ไม่มีบทสรุปภาพรวม").replace("\\n", "\n")
    bias = (report.get("market_bias") or "NEUTRAL").upper()
    overview_desc = (
        "### 📊 ตัวเลขขับเคลื่อนราคาทองคำ (Live Drivers)\n"
        + "\n".join(macro_lines)
        + tech_block
        + "\n\n### 🧠 บทวิเคราะห์เจาะลึกและเอฟเฟกต์ต่อตลาด\n"
        + macro_summary
    )

    embeds.append({
        "title": f"🪙 Gold Macro & Technical Deep Brief (Bias: {bias})",
        "description": overview_desc[:4000],
        "color": colors.get(bias, 16766720),
        "footer": {
            "text": "🤖 Deep Technical + Macro AI Analysis • AlphaZone Gold Intelligence"
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

    print("1. Loading cache & fetching Gold / Macro + Technical data...")
    sent_cache = load_sent_cache(CACHE_FILE)
    print(f"   Sent gold cache has {len(sent_cache)} items.")

    market_info = fetch_gold_macro_data()
    raw_news = fetch_rss_feeds(GOLD_RSS_FEEDS, sent_cache=sent_cache, max_per_feed=15)
    print(f"   Found {len(raw_news)} fresh gold/macro articles.")

    if not raw_news:
        print("No new unseen gold/macro articles since last run.")
        return

    print("2. Analyzing Gold technicals, macro drivers & news with Gemini...")
    report = analyze_gold_market(market_info, raw_news)
    if not report or not report.get("curated_news"):
        print("Failed to generate Gold report.")
        return

    print("3. Sending deep Gold report to Discord (#gold-news)...")
    embeds = build_gold_embeds(market_info, report, raw_news)
    header = "🪙 **อัปเดตสถานการณ์ทองคำโลกเชิงลึก (Technical + Macro & News Brief)** ✨"
    if send_discord_embeds(DISCORD_WEBHOOK_URL_GOLD, header, embeds, chunk_size=2):
        for item in report.get("curated_news", []):
            orig_idx = int(item.get("original_index", 1)) - 1
            if 0 <= orig_idx < len(raw_news):
                sent_cache.add(raw_news[orig_idx]["link"])
        save_sent_cache(CACHE_FILE, sent_cache)
        print("4. Updated sent gold cache successfully.")


if __name__ == "__main__":
    main()
