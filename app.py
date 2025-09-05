# app.py — ValueVault Thin Backend v1
# Run locally: uvicorn app:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time, random, requests
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# -----------------
# App & Middleware
# -----------------
app = FastAPI(title="ValueVault Thin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "us"
DEFAULT_ODDS_FORMAT = "american"

# -------------
# TTL Cache
# -------------
class TTLCache:
    def __init__(self, ttl_seconds=75):
        self.ttl = ttl_seconds
        self.store = {}
    def get(self, key):
        v = self.store.get(key)
        if not v: return None
        exp, val = v
        if time.time() > exp:
            self.store.pop(key, None)
            return None
        return val
    def set(self, key, value):
        self.store[key] = (time.time() + self.ttl, value)

cache = TTLCache(ttl_seconds=75)

# -------------
# HTTP helper with retry
# -------------
def http_get_retry(url, params, timeout=10, retries=2):
    for i in range(retries+1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            msg = f"Upstream {r.status_code}: {r.text[:200]}"
            if i == retries:
                raise HTTPException(status_code=502, detail=msg)
        except Exception as e:
            if i == retries:
                raise HTTPException(status_code=502, detail=str(e))
        time.sleep(0.25 + 0.25*random.random())

# -------------
# Math helpers
# -------------
def amer_to_imp(odds:int)->Optional[float]:
    if odds is None: return None
    return abs(odds)/(abs(odds)+100) if odds<0 else 100/(odds+100)

def ev_pct(win_p: float, price_amer: int) -> Optional[float]:
    if win_p is None: return None
    payout = (100/abs(price_amer)) if price_amer < 0 else (price_amer/100)
    return (win_p * payout - (1.0 - win_p)) * 100.0

BOOK_WEIGHTS = {
    "pinnacle": 1.00, "pinny": 1.00, "circa": 0.90, "betcris": 0.90,
    "draftkings": 0.70, "fanduel": 0.70, "betrivers": 0.65, "betmgm": 0.65,
    "caesars": 0.65,
    "__default__": 0.60
}

def normalize_probs(vec: List[float]) -> List[float]:
    s = sum(vec) or 1.0
    return [v/s for v in vec]

def consensus_fair_probs_by_market(book_outcomes: List[Dict[str,Any]], alpha_prior: float=0.10) -> Optional[Dict[str,float]]:
    # Outcome names (e.g., Team A, Team B, Draw)
    names = []
    for b in book_outcomes:
        for o in b.get("outcomes", []):
            n = o.get("name")
            if n not in names: names.append(n)

    # Build per-book vig-free vectors, then weighted-average by book weight
    per_book_vectors = []
    for b in book_outcomes:
        book_key = (b.get("book") or "").lower()
        w = BOOK_WEIGHTS.get(book_key, BOOK_WEIGHTS["__default__"])
        vec = []
        ok = True
        for name in names:
            price = None
            for o in b.get("outcomes", []):
                if o.get("name") == name:
                    price = o.get("price")
                    break
            if price is None:
                ok = False; break
            p = amer_to_imp(price)
            if p is None: ok = False; break
            vec.append(p)
        if ok:
            fair_vec = normalize_probs(vec)  # vig-free for this book's market
            per_book_vectors.append((w, fair_vec))

    if not per_book_vectors:
        return None

    total_w = sum(w for w,_ in per_book_vectors) or 1.0
    k = len(per_book_vectors[0][1])
    avg = [0.0]*k
    for w, vec in per_book_vectors:
        for i in range(k):
            avg[i] += (w/total_w) * vec[i]

    # Light Bayesian shrink toward uniform (sport-specific priors can be added later)
    if 0.0 < alpha_prior < 1.0:
        prior = [1.0/len(avg)]*len(avg)
        avg = [(1-alpha_prior)*p + alpha_prior*q for p,q in zip(avg, prior)]

    return {name: p for name,p in zip(names, avg)}

def best_available_odds_across_books(book_outcomes: List[Dict[str,Any]], target_name: str):
    best, best_book = None, None
    for b in book_outcomes:
        book = b.get("book")
        for o in b.get("outcomes", []):
            if o.get("name") == target_name and o.get("price") is not None:
                price = o["price"]
                if best is None:
                    best, best_book = price, book
                else:
                    # Improved bettor-friendly comparison
                    if (price > 0 and (best <= 0 or price > best)) or (price < 0 and best < 0 and price > best):
                        best, best_book = price, book
    return best, best_book

# -------------
# Time helper
# -------------
ET_TZ = "America/New_York"

def to_et_str(iso_str: Optional[str]) -> Optional[str]:
    if not iso_str: return None
    try:
        iso = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if ZoneInfo:
            dt = dt.astimezone(ZoneInfo(ET_TZ))
        # %-I is not portable on Windows; use fallback
        try:
            return dt.strftime("%-I:%M %p ET")
        except:  # pragma: no cover
            return dt.strftime("%I:%M %p ET").lstrip("0") + " ET"
    except Exception:
        return None

# -------------
# Schemas
# -------------
class OddsRequest(BaseModel):
    api_key: str
    sport_key: str                  # e.g., "baseball_mlb", "tennis_atp", "americanfootball_nfl"
    markets: List[str]              # ["h2h","spreads","totals"] or props later
    books: Optional[List[str]] = None
    cache_ttl_sec: Optional[int] = 75
    ev_floor_pct: Optional[float] = 1.0
    unit_floor: Optional[float] = 0.25
    unit_cap: Optional[float] = 1.25
    win_floor_pct: Optional[float] = 48.0
    alpha_prior: Optional[float] = 0.10

class ParlayReq(BaseModel):
    rows: List[Dict[str, Any]]
    legs: int = 2
    max_suggestions: int = 5
    block_same_game: bool = True
    prefer_same_book: bool = True

# -------------
# Endpoints
# -------------
@app.get("/")
def root():
    return {"ok": True}

@app.post("/straight")
def get_straight(req: OddsRequest):
    cache.ttl = req.cache_ttl_sec or 75
    key = f"straight:{req.sport_key}:{','.join(req.markets)}:{','.join(req.books or [])}:{req.alpha_prior}"
    cached = cache.get(key)
    if cached: return cached

    params = {
        "apiKey": req.api_key,
        "regions": DEFAULT_REGIONS,
        "oddsFormat": DEFAULT_ODDS_FORMAT,
        "markets": ",".join(req.markets)
    }
    if req.books:
        params["bookmakers"] = ",".join(req.books)

    url = f"{ODDS_API_BASE}/sports/{req.sport_key}/odds"
    data = http_get_retry(url, params)

    rows = []
    for event in data:
        start_iso = event.get("commence_time")
        start_et = to_et_str(start_iso)
        home, away = event.get("home_team"), event.get("away_team")
        eid = event.get("id")
        bms = event.get("bookmakers", [])

        # bucket outcomes by market across books
        market_bucket: Dict[str, List[Dict[str,Any]]] = {}
        for bm in bms:
            book = (bm.get("key") or "").lower()
            for m in bm.get("markets", []):
                mkt = m.get("key")
                outs = m.get("outcomes", [])
                if not outs: continue
                market_bucket.setdefault(mkt, []).append({"book": book, "outcomes": outs})

        for mkt, book_outcomes in market_bucket.items():
            fair = consensus_fair_probs_by_market(book_outcomes, alpha_prior=req.alpha_prior or 0.10)
            if not fair:
                # last-resort fallback: average implied, then normalize
                all_names = list({o["name"] for b in book_outcomes for o in b.get("outcomes", [])})
                agg = {n: [] for n in all_names}
                for b in book_outcomes:
                    for o in b.get("outcomes", []):
                        p = amer_to_imp(o.get("price"))
                        if p is not None:
                            agg[o["name"]].append(p)
                vec = [sum(v)/len(v) if v else 0.0 for v in agg.values()]
                vec = normalize_probs(vec)
                fair = {n: p for n,p in zip(agg.keys(), vec)}

            for outcome_name, p_model in fair.items():
                price, best_book = best_available_odds_across_books(book_outcomes, outcome_name)
                if price is None or p_model is None:
                    continue
                ev = ev_pct(p_model, price)
                if req.ev_floor_pct is not None and ev < req.ev_floor_pct:
                    continue
                if req.win_floor_pct is not None and (p_model*100.0) < req.win_floor_pct:
                    continue

                # Unit sizing (Available EV % tiers)
                unit = 1.0 if ev >= 2.5 else 0.75 if ev >= 1.5 else 0.5
                unit = max(req.unit_floor or 0.25, min(req.unit_cap or 1.25, unit))

                rows.append({
                    "EventID": eid,
                    "StartISO": start_iso,
                    "Start Time (ET)": start_et,
                    "Matchup": f"{away} @ {home}" if away and home else event.get("sport_title", ""),
                    "Market": mkt,
                    "BetSide": outcome_name,
                    "Book": best_book,
                    "Odds": price,
                    "ModelWinPct": round(p_model*100.0, 2),
                    "AvailableEVPct": round(ev, 2),
                    "UnitSize": unit
                })

    rows.sort(key=lambda r: (-r["AvailableEVPct"], r.get("StartISO") or ""))
    payload = {"rows": rows, "count": len(rows)}
    cache.set(key, payload)
    return payload

@app.post("/parlay/suggest")
def parlay_suggest(req: ParlayReq):
    rows = [r for r in req.rows if isinstance(r.get("AvailableEVPct"), (int, float))]
    rows.sort(key=lambda r: (-r["AvailableEVPct"], r.get("StartISO", ""), r.get("Matchup", "")))
    rng = random.Random(42)

    def ev_key(r):
        return (r.get("EventID") or r.get("Matchup"), r.get("Market"))

    suggestions, used = [], set()
    buckets: Dict[str, List[Dict[str,Any]]] = {}
    for r in rows:
        b = (r.get("Book") if req.prefer_same_book else "any") or "any"
        buckets.setdefault(b, []).append(r)

    for book, bucket in buckets.items():
        i = 0
        while i < len(bucket) and len(suggestions) < req.max_suggestions:
            pick, seen_events = [], set()
            j = i
            while j < len(bucket) and len(pick) < req.legs:
                r = bucket[j]
                k = ev_key(r)
                if k in used:
                    j += 1; continue
                if req.block_same_game and r.get("EventID") in seen_events:
                    j += 1; continue
                pick.append(r); used.add(k); seen_events.add(r.get("EventID"))
                j += 1
            if len(pick) == req.legs:
                suggestions.append({
                    "book": book,
                    "legs": pick,
                    "sumEVPct": round(sum(x["AvailableEVPct"] for x in pick), 2)
                })
            i = j
        if len(suggestions) >= req.max_suggestions:
            break

    return {"parlays": suggestions}

# ---- DubClub text builder ----
def fmt_amer(n):
    try:
        n = int(n)
        return f"{n:+d}" if n>0 else f"{n:d}"
    except Exception:
        return str(n)

def fmt_line(r)->str:
    start = r.get("Start Time (ET)") or r.get("StartISO")
    seg = f"{r.get('Matchup','')} — {r.get('Market','')} {r.get('BetSide','')} {fmt_amer(r.get('Odds'))} ({(r.get('Book') or '').upper()}) — {round(float(r.get('UnitSize',1.0)),2)}u"
    return seg + (f" | {start}" if start else "")

@app.post("/post/dubclub")
def post_dubclub(body: Dict[str, Any]):
    rows = body.get("rows", [])
    title_date = body.get("title_date", "")
    sport_label = body.get("sport_label", "Multi-Sport")

    header = f"ValueVaultHQ — {title_date} Final Card\n\n"
    card = [f"**{sport_label}**\n"]
    for r in rows:
        card.append(fmt_line(r))
    footer = ("\n\nNotes:\n"
              "• Odds/lines can move; we post best widely-available numbers.\n"
              "• Informational only • 21+ • Play responsibly.")
    dubclub_text = header + "\n".join(card) + footer

    total_units = round(sum(float(r.get("UnitSize", 0)) for r in rows), 2)
    x_post = (f"Card is live 🔒  {sport_label}\n"
              f"{len(rows)} plays • {total_units}u on board\n"
              "Lineup-confirmed +EV. Details in bio/pinned.")

    return {"dubclub_text": dubclub_text, "x_post": x_post}
