import os
import json
import re
import time
from urllib.parse import urlparse
import requests
import feedparser

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def load_sent_cache(cache_path: str) -> set:
    """Loads previously sent article URLs from a JSON cache file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data) if isinstance(data, list) else set()
        except Exception as e:
            print(f"Notice: Cache load skipped for {cache_path} ({e})")
    return set()


def save_sent_cache(cache_path: str, sent_set: set, max_items: int = 250) -> None:
    """Saves sent article URLs, retaining the most recent max_items entries."""
    try:
        items = list(sent_set)[-max_items:]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Notice: Cache save skipped for {cache_path} ({e})")


def to_google_translate_url(url: str) -> str:
    """Creates a Google Translate web proxy link (<host>.translate.goog) for 1-click Thai reading."""
    try:
        parsed = urlparse(url)
        if not parsed.hostname or "translate.goog" in parsed.hostname:
            return url
        clean_host = parsed.hostname.lower().replace("-", "--").replace(".", "-")
        trans_host = f"{clean_host}.translate.goog"
        params = "_x_tr_sl=auto&_x_tr_tl=th&_x_tr_hl=th&_x_tr_pto=wapp"
        query_str = f"{parsed.query}&{params}" if parsed.query else params
        return f"https://{trans_host}{parsed.path}?{query_str}"
    except Exception:
        return url


def format_article_links(source_name: str, original_link: str) -> str:
    """Formats Discord markdown links for original article and 1-click Thai translation."""
    translate_link = to_google_translate_url(original_link)
    link_line = f"📰 **[อ่านข่าวต้นฉบับ ({source_name})]({original_link})**"
    if translate_link and translate_link != original_link:
        link_line += f" · 🌐 **[อ่านแบบแปลไทย]({translate_link})**"
    return link_line


def fetch_rss_feeds(
    feeds_dict: dict,
    sent_cache: set = None,
    max_per_feed: int = 20,
    timeout: int = 10,
) -> list:
    """Fetches RSS feeds safely using browser-like User-Agent and filters out cached URLs."""
    if sent_cache is None:
        sent_cache = set()

    all_news = []
    seen_links = set()

    for source, url in feeds_dict.items():
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)

            for entry in feed.entries[:max_per_feed]:
                link = (entry.get("link") or "").strip()
                if not link or link in seen_links or link in sent_cache:
                    continue
                seen_links.add(link)

                raw_summary = entry.get("summary", "") or entry.get("description", "")
                clean_summary = re.sub(r"<[^>]+>", "", raw_summary).strip()[:400]

                all_news.append({
                    "source": source,
                    "title": (entry.get("title") or "No Title").strip(),
                    "link": link,
                    "published": entry.get("published", ""),
                    "summary": clean_summary,
                })
        except Exception as e:
            print(f"Error fetching from {source}: {e}")

    return all_news


def send_discord_embeds(
    webhook_url: str,
    header_content: str,
    embeds: list,
    chunk_size: int = 2,
) -> bool:
    """Sends Discord embeds in safe chunks with rate-limit delay. Returns True if all chunks succeed."""
    if not embeds or not webhook_url:
        return False

    all_ok = True
    for i in range(0, len(embeds), chunk_size):
        chunk = embeds[i:i + chunk_size]
        payload = {
            "content": header_content if i == 0 else "",
            "embeds": chunk,
        }
        try:
            response = requests.post(webhook_url, json=payload, timeout=15)
            response.raise_for_status()
            print(f"Successfully sent chunk {i // chunk_size + 1} to Discord!")
            time.sleep(2)
        except Exception as e:
            print(f"Error sending chunk {i // chunk_size + 1} to Discord: {e}")
            all_ok = False

    return all_ok
