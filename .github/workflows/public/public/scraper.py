"""
Events Radar — production scraper (JSON output variant).

Runs daily via GitHub Actions. Fetches each configured source page,
passes it to Claude for structured extraction, deduplicates, and
writes events.json + events.csv to the repo. The static webpage
(events_radar_app.html) fetches events.json and re-renders.

Environment variables:
  ANTHROPIC_API_KEY   - required, set as GitHub Actions secret
  OUTPUT_DIR          - optional, defaults to "public"

Local test run:
  export ANTHROPIC_API_KEY=sk-ant-...
  python scraper.py
"""

import os
import json
import hashlib
import sqlite3
from datetime import datetime, date
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from anthropic import Anthropic

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "public"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = OUTPUT_DIR / "events.db"

# ---------------------------------------------------------------------------
# SOURCES
# ---------------------------------------------------------------------------
SOURCES = [
    {"name": "Global Investigations Review", "url": "https://globalinvestigationsreview.com/events"},
    {"name": "ACAMS", "url": "https://www.acams.org/en/events/acams-events-view-all-upcoming-events"},
    {"name": "LCIA", "url": "https://www.lcia.org"},
    {"name": "C5 Anti-Corruption London", "url": "https://www.c5-online.com/ac-london/"},
    {"name": "C5 FCPA Portfolio", "url": "https://www.c5-online.com/conference/anti-corruption-fcpa/"},
    {"name": "Cambridge Economic Crime Symposium", "url": "https://www.crimesymposium.org/register"},
    {"name": "LIDW", "url": "https://register.lidw.co.uk/"},
    {"name": "RUSI CFS", "url": "https://www.rusi.org/explore-our-research/research-groups/centre-for-finance-and-security"},
    {"name": "IAC London Arbitration Calendar", "url": "https://www.iac-london.com/arbitral-events-2026/"},
    {"name": "Herbert Smith Freehills", "url": "https://www.hsfkramer.com/events"},
    {"name": "Gibson Dunn", "url": "https://www.gibsondunn.com/events/"},
    {"name": "WilmerHale", "url": "https://www.wilmerhale.com/en/insights/events"},
]

# ---------------------------------------------------------------------------
# EXTRACTION PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You extract structured event records from HTML scraped from
law firm, regulator, professional body, and conference organiser websites.

Return ONLY valid JSON — no preamble, no markdown fences, no commentary.
Return an array of events. If the page has no events, return [].

RELEVANCE: Only include events whose primary subject matter is disputes,
litigation, arbitration, mediation, internal or regulatory investigations,
financial crime (AML, sanctions, ABC/bribery, fraud, market abuse, tax
evasion, export controls), asset recovery, or enforcement.
Exclude general corporate, M&A, tax, employment, IP, real estate, ESG.

If the event's end date is before today's date, set relevant=false.

SCHEMA per event (all keys required):
{
  "title": "string — event name, no organiser suffix",
  "organiser": "string",
  "start": "YYYY-MM-DD",
  "end": "YYYY-MM-DD (same as start if single-day)",
  "city": "string or 'Virtual'",
  "country": "string or ''",
  "region": "UK | Europe | North America | APAC | MENA | LATAM | Africa | Global",
  "format": "In-person | Virtual | Hybrid",
  "topics": ["array from: Investigations, FCPA, ABC, Bribery, Fraud, AML, Sanctions, Export Controls, Financial Crime, Market Abuse, Tax Evasion, Asset Recovery, Enforcement, Disputes, Litigation, Arbitration, Mediation, Compliance, Policy, Networking"],
  "audience": "Junior | Mixed | Senior",
  "cost": "Free | Paid | Invite-only",
  "costDisplay": "string as shown on page (e.g. '£695', 'Free (members)', 'Register')",
  "url": "registration URL from the page, or the source URL if not found",
  "flags": ["Flagship" if the event is a major marquee event, otherwise empty array],
  "summary": "<=40 words, factual, plain English, no marketing language",
  "confidence": 0.0-1.0,
  "relevant": true or false
}

Today's date will be in the user message; use it to resolve relative dates."""

# ---------------------------------------------------------------------------
# FETCH
# ---------------------------------------------------------------------------
def fetch_html(url: str) -> str:
    headers = {
        "User-Agent": "EventsRadar/1.0 (+https://github.com/yourname/events-radar)",
        "Accept": "text/html,application/xhtml+xml",
    }
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        r = client.get(url, headers=headers)
        r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return str(main)[:60_000]

# ---------------------------------------------------------------------------
# EXTRACT
# ---------------------------------------------------------------------------
def extract_events(html: str, source_url: str, client: Anthropic) -> list[dict]:
    today = date.today().isoformat()
    user_msg = f"today: {today}\nsource_url: {source_url}\n\nHTML:\n{html}"
    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    try:
        events = json.loads(text)
        if not isinstance(events, list):
            return []
        return events
    except json.JSONDecodeError as e:
        print(f"    ! JSON parse error: {e}")
        print(f"    first 200 chars of response: {text[:200]}")
        return []

# ---------------------------------------------------------------------------
# DB (for dedup and persistence across runs)
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    title TEXT, organiser TEXT,
    start TEXT, end TEXT,
    city TEXT, country TEXT,
    region TEXT, format TEXT,
    topics TEXT,            -- JSON array string
    audience TEXT,
    cost TEXT, costDisplay TEXT,
    url TEXT, flags TEXT,   -- JSON array string
    summary TEXT,
    source_name TEXT, source_url TEXT,
    first_seen TEXT, last_seen TEXT,
    confidence REAL
);
CREATE INDEX IF NOT EXISTS idx_start ON events(start);
"""

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn

def event_id(ev: dict) -> str:
    key = f"{ev.get('title','').lower().strip()}|{ev.get('start','')}|{ev.get('city','').lower().strip()}"
    return "EVT-" + hashlib.sha1(key.encode()).hexdigest()[:10].upper()

def upsert(conn, ev: dict, source_name: str, source_url: str):
    if not ev.get("relevant", True):
        return False
    if ev.get("confidence", 1.0) < 0.5:
        return False
    # Validate minimum required fields
    if not ev.get("title") or not ev.get("start"):
        return False

    eid = event_id(ev)
    today = date.today().isoformat()

    existing = conn.execute("SELECT first_seen FROM events WHERE id=?", (eid,)).fetchone()
    first_seen = existing[0] if existing else today

    conn.execute("""
        INSERT OR REPLACE INTO events VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, (
        eid, ev.get("title"), ev.get("organiser"), ev.get("start"), ev.get("end", ev.get("start")),
        ev.get("city"), ev.get("country", ""), ev.get("region"), ev.get("format"),
        json.dumps(ev.get("topics", [])), ev.get("audience", "Mixed"),
        ev.get("cost"), ev.get("costDisplay", ""), ev.get("url", source_url),
        json.dumps(ev.get("flags", [])), ev.get("summary", ""),
        source_name, source_url, first_seen, today, ev.get("confidence", 1.0)
    ))
    return True

# ---------------------------------------------------------------------------
# EXPORT to events.json (the webpage reads this)
# ---------------------------------------------------------------------------
def export_json(conn):
    rows = conn.execute("""
        SELECT id, title, organiser, start, end, city, country, region, format,
               topics, audience, cost, costDisplay, url, flags, summary
        FROM events
        WHERE end >= DATE('now','-7 days')
        ORDER BY start ASC
    """).fetchall()

    events = []
    for r in rows:
        events.append({
            "id": r[0],
            "title": r[1],
            "organiser": r[2],
            "start": r[3],
            "end": r[4],
            "city": r[5],
            "country": r[6],
            "region": r[7],
            "format": r[8],
            "topics": json.loads(r[9] or "[]"),
            "audience": r[10],
            "cost": r[11],
            "costDisplay": r[12],
            "url": r[13],
            "flags": json.loads(r[14] or "[]"),
            "summary": r[15] or ""
        })

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "event_count": len(events),
        "events": events
    }

    out_path = OUTPUT_DIR / "events.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n✓ Wrote {len(events)} events to {out_path}")

    # Also write a CSV for anyone who wants to pull the same data elsewhere
    import csv
    csv_path = OUTPUT_DIR / "events.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id","title","organiser","start","end","city","country","region",
                    "format","topics","audience","cost","costDisplay","url","flags","summary"])
        for e in events:
            w.writerow([
                e["id"], e["title"], e["organiser"], e["start"], e["end"],
                e["city"], e["country"], e["region"], e["format"],
                "; ".join(e["topics"]), e["audience"], e["cost"], e["costDisplay"],
                e["url"], "; ".join(e["flags"]), e["summary"]
            ])
    print(f"✓ Wrote CSV mirror to {csv_path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY environment variable not set")

    client = Anthropic()
    conn = get_db()

    total_upserted = 0
    total_skipped = 0

    for i, source in enumerate(SOURCES, 1):
        print(f"[{i}/{len(SOURCES)}] {source['name']} — {source['url']}")
        try:
            html = fetch_html(source["url"])
        except Exception as e:
            print(f"    ! fetch failed: {e}")
            continue

        try:
            events = extract_events(html, source["url"], client)
        except Exception as e:
            print(f"    ! extraction failed: {e}")
            continue

        print(f"    extracted {len(events)} candidate events")

        for ev in events:
            if upsert(conn, ev, source["name"], source["url"]):
                total_upserted += 1
            else:
                total_skipped += 1

        conn.commit()

    print(f"\nSummary: {total_upserted} events upserted, {total_skipped} skipped")
    export_json(conn)
    conn.close()

if __name__ == "__main__":
    run()
