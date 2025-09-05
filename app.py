# app.py — ValueVault Thin Backend (PROPS-first)
# Run locally: uvicorn app:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import time, random, requests
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

# -----------------
# App & Middleware
# -----------------
app = FastAPI(title="ValueVault Props API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_REGIONS = "us2"            # props are best on us2
DEFAULT_ODDS_FORMAT = "american"

# -------------
# TTL Cache
# -------------
class TTLCache:
    def __init__(self, ttl_seconds=90):
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

cache = TTLCache(ttl_seconds=90)

# -------------
# HTTP helper with retry
# -------------
def http_get_retry(url, params, timeout=15, retries=2):
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
    "draftkings": 0.70, "fanduel": 0.70, "betrivers": 0.65,
    "betmgm": 0.65, "caesars": 0.65,
    "__default__": 0.60
}

def to_et_str(iso_str: Optional[str]) -> Optional[str]:
    if not iso_str: return None
    try:
        iso = iso_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso)
        if ZoneInfo:
            dt = dt.astimezone(ZoneInfo("America/New_York"))
        try:
            return dt.strftime("%-I:%M %p ET")
        except:
            return dt.strftime("%I:%M %p ET").lstrip("0") + " ET"
    except Exception:
        return None

# -----------------
# Schemas
# -----------------
class PropsPull(BaseModel):
    api_key: str
    sports: List[str] = Field(..., description="e.g. ['baseball_mlb','basketball_wnba','americanfootball_nfl']")
    markets: List[str] = Field(default=["player_props"], description="Use ['player_props'] for all props per game")
    books: List[str] = Field(default_factory=list, description="['fanduel','draftkings','betmgm','betrivers','caesars']")
    regions: str = DEFAULT_REGIONS
    cache_ttl_sec: int = 90

class PropsRowRaw(BaseModel):
    Date: str
    Start_Time: str
    Sport: str
    Matchup: str
    Player: str
    Prop: str
    Bet: str
    Point: Optional[str]
    Best_Book: Optional[str]
    Odds: Optional[int]
    Event_ID: str
    Bookmaker_Key: Optional[str]
    Market: str
    Outcome_Name: str
    StartISO: str

class PropsRowModel(PropsRowRaw):
    ModelWinPct: Optional[float] = None
    AvailableEVPct: Optional[float] = None
    UnitSize: Optional[float] = None

class PropsResp(BaseModel):
    rows: List[Dict[str, Any]]
    meta: Dict[str, Any] = {}

# -----------------
# Helpers for props parsing
# -----------------
def best_available_odds(book_outcomes: List[Dict[str,Any]], target_name: str, target_desc: str, target_point) -> Tuple[Optional[int], Optional[str]]:
    best, best_book = None, None
    for b in book_outcomes:
        book = b.get("book")
        for o in b.get("outcomes", []) or []:
            if (o.get("name")==target_name and
                (o.get("description") or "").strip().lower()==(target_desc or "").strip().lower() and
                (o.get("point")==target_point) and
                o.get("price") is not None):
                price = o["price"]
                if best is None:
                    best, best_book = price, book
                else:
                    if (price > 0 and (best <= 0 or price > best)) or (price < 0 and best < 0 and price > best):
                        best, best_book = price, book
    return best, best_book

def vig_free_consensus_prob(book_outcomes: List[Dict[str,Any]], player: str, point, side_a="Over", side_b="Under", alpha_prior=0.10) -> Dict[str,float]:
    """Return fair probs for Over/Under (or Yes/No) at a given point."""
    # Build per-book normalized probs for the two sides, then weight by book quality
    per_book = []
    for b in book_outcomes:
        book = (b.get("book") or "").lower()
        w = BOOK_WEIGHTS.get(book, BOOK_WEIGHTS["__default__"])
        pa, pb = None, None
        for o in b.get("outcomes", []) or []:
            if o.get("name")==player and o.get("point")==point:
                desc = (o.get("description") or "").strip().lower()
                if desc == side_a.lower(): pa = amer_to_imp(o.get("price"))
                if desc == side_b.lower(): pb = amer_to_imp(o.get("price"))
        if pa is not None and pb is not None and (pa+pb)>0:
            s = pa+pb
            per_book.append((w, pa/s, pb/s))   # vig-free for this book
    if not per_book:
        return {}
    tot_w = sum(w for w,_,_ in per_book) or 1.0
    a = sum((w/tot_w)*pa for w,pa,_ in per_book)
    b = sum((w/tot_w)*pb for w,_,pb in per_book)
    # Light shrink toward 50/50 for stability
    a = (1-alpha_prior)*a + alpha_prior*0.5
    b = (1-alpha_prior)*b + alpha_prior*0.5
    # Normalize again (tiny drift guard)
    s = (a+b) or 1.0
    return {side_a: a/s, side_b: b/s}

def normalize_raw_row(game: Dict[str,Any], bkey: str, market: Dict[str,Any], out: Dict[str,Any]) -> PropsRowRaw:
    start_iso = game.get("commence_time") or ""
    return PropsRowRaw(
        Date="",
        Start_Time=to_et_str(start_iso) or "",
        Sport=(game.get("sport_key") or game.get("sport_title") or ""),
        Matchup=f"{(game.get('away_team') or '').strip()} @ {(game.get('home_team') or '').strip()}".strip(" @"),
        Player=(out.get("name") or "").strip(),
        Prop=(market.get("key") or market.get("market_key") or ""),
        Bet=(out.get("description") or "").strip(),
        Point=str(out.get("point")) if out.get("point") is not None else "",
        Best_Book=bkey,
        Odds=out.get("price"),
        Event_ID=str(game.get("id") or ""),
        Bookmaker_Key=bkey,
        Market=(market.get("key") or market.get("market_key") or ""),
        Outcome_Name=(out.get("name") or ""),
        StartISO=start_iso
    )

# -----------------
# PROPS RAW
# -----------------
@app.post("/props/raw", response_model=PropsResp)
def props_raw(req: PropsPull):
    cache.ttl = req.cache_ttl_sec or 90
    rows: List[Dict[str,Any]] = []
    games_seen = 0

    for sport in req.sports:
        params = {
            "apiKey": req.api_key,
            "regions": req.regions or DEFAULT_REGIONS,
            "oddsFormat": DEFAULT_ODDS_FORMAT,
            "markets": ",".join(req.markets or ["player_props"]),
        }
        if req.books:
            params["bookmakers"] = ",".join(req.books)
        url = f"{ODDS_API_BASE}/sports/{sport}/odds"
        data = http_get_retry(url, params)
        games_seen += len(data)

        for g in data:
            bms = g.get("bookmakers", []) or []
            for bm in bms:
                book = (bm.get("key") or "").lower()
                for m in bm.get("markets", []) or []:
                    outs = m.get("outcomes", []) or []
                    for o in outs:
                        r = normalize_raw_row(g, book, m, o)
                        rows.append(r.model_dump())

    return PropsResp(rows=rows, meta={"sports": req.sports, "books": req.books, "games_seen": games_seen, "rows": len(rows)})

# -----------------
# PROPS MODEL (vig-free consensus ModelWin% + AvailableEV%)
# -----------------
class PropsModelPull(PropsPull):
    alpha_prior: float = 0.10
    ev_floor_pct: float = 0.5
    unit_floor: float = 0.25
    unit_cap: float = 1.25

@app.post("/props/model", response_model=PropsResp)
def props_model(req: PropsModelPull):
    rows: List[Dict[str,Any]] = []
    games_seen = 0

    for sport in req.sports:
        params = {
            "apiKey": req.api_key,
            "regions": req.regions or DEFAULT_REGIONS,
            "oddsFormat": DEFAULT_ODDS_FORMAT,
            "markets": ",".join(req.markets or ["player_props"]),
        }
        if req.books:
            params["bookmakers"] = ",".join(req.books)
        url = f"{ODDS_API_BASE}/sports/{sport}/odds"
        data = http_get_retry(url, params)
        games_seen += len(data)

        for g in data:
            start_iso = g.get("commence_time") or ""
            start_et = to_et_str(start_iso) or ""
            eid = str(g.get("id") or "")
            home, away = g.get("home_team"), g.get("away_team")
            matchup = f"{(g.get('away_team') or '').strip()} @ {(g.get('home_team') or '').strip()}".strip(" @")

            # Build cross-book buckets per (player, market_key, point)
            buckets: Dict[Tuple[str,str,Any], List[Dict[str,Any]]] = {}
            for bm in g.get("bookmakers", []) or []:
                book = (bm.get("key") or "").lower()
                for m in bm.get("markets", []) or []:
                    mkey = (m.get("key") or m.get("market_key") or "")
                    for o in (m.get("outcomes", []) or []):
                        player = (o.get("name") or "").strip()
                        point = o.get("point")
                        desc = (o.get("description") or "").strip()
                        price = o.get("price")
                        if player and desc and price is not None:
                            k = (player, mkey, point)
                            buckets.setdefault(k, []).append({"book":book, "outcomes":[{"name":player,"description":desc,"point":point,"price":price}]})

            # For each bucket, compute vig-free fair probs for Over/Under (or Yes/No)
            for (player, mkey, point), entries in buckets.items():
                # Detect sides present
                sides = set((o.get("outcomes")[0].get("description") or "").strip().title() for o in entries)
                if {"Over","Under"}.issubset(sides):
                    fair = vig_free_consensus_prob(entries, player, point, side_a="Over", side_b="Under", alpha_prior=req.alpha_prior)
                    sides_order = ["Over","Under"]
                elif {"Yes","No"}.issubset(sides):
                    fair = vig_free_consensus_prob(entries, player, point, side_a="Yes", side_b="No", alpha_prior=req.alpha_prior)
                    sides_order = ["Yes","No"]
                else:
                    continue  # skip odd markets that aren't binary O/U or Yes/No

                for side in sides_order:
                    best_odds, best_book = best_available_odds(entries, player, side, point)
                    if best_odds is None: 
                        continue
                    p = fair.get(side)
                    if p is None: 
                        continue

                    ev = ev_pct(p, best_odds)
                    if ev is None or ev < req.ev_floor_pct:
                        continue

                    unit = 1.0 if ev>=2.5 else 0.75 if ev>=1.5 else 0.5
                    unit = max(req.unit_floor or 0.25, min(req.unit_cap or 1.25, unit))

                    rows.append(PropsRowModel(
                        Date="",
                        Start_Time=start_et,
                        Sport=sport,
                        Matchup=matchup,
                        Player=player,
                        Prop=mkey,
                        Bet=side,
                        Point=str(point) if point is not None else "",
                        Best_Book=best_book,
                        Odds=best_odds,
                        Event_ID=eid,
                        Bookmaker_Key=best_book,
                        Market=mkey,
                        Outcome_Name=player,
                        StartISO=start_iso,
                        ModelWinPct=round(100*p,2),
                        AvailableEVPct=round(ev,2),
                        UnitSize=unit
                    ).model_dump())

    # Sort by EV desc
    rows.sort(key=lambda r: (-float(r.get("AvailableEVPct",0)), r.get("StartISO") or ""))
    return PropsResp(rows=rows, meta={"sports": req.sports, "books": req.books, "games_seen": games_seen, "rows": len(rows)})

# -----------------
# Parlay Suggest (already used by Sheets; safe for props too)
# -----------------
class ParlayReq(BaseModel):
    rows: List[Dict[str, Any]]
    legs: int = 2
    max_suggestions: int = 5
    block_same_game: bool = True
    prefer_same_book: bool = True

@app.post("/parlay/suggest")
def parlay_suggest(req: ParlayReq):
    rows = [r for r in req.rows if isinstance(r.get("AvailableEVPct"), (int, float))]
    rows.sort(key=lambda r: (-r["AvailableEVPct"], r.get("StartISO", ""), r.get("Matchup", "")))
    used = set(); suggestions = []

    def key(r): return (r.get("Event_ID") or r.get("EventID") or r.get("Matchup"), r.get("Player") or r.get("Market"))

    buckets: Dict[str, List[Dict[str,Any]]] = {}
    for r in rows:
        b = (r.get("Best_Book") or r.get("Book") if req.prefer_same_book else "any") or "any"
        buckets.setdefault(b, []).append(r)

    for book, bucket in buckets.items():
        i = 0
        while i < len(bucket) and len(suggestions) < req.max_suggestions:
            pick, seen_events, seen_players = [], set(), set()
            j = i
            while j < len(bucket) and len(pick) < req.legs:
                r = bucket[j]; j += 1
                k = key(r)
                if k in used: continue
                if req.block_same_game and (r.get("Event_ID") or r.get("EventID")) in seen_events:
                    continue
                # simple correlation guard: avoid same player twice
                if (r.get("Player") or "") in seen_players:
                    continue
                pick.append(r); used.add(k)
                seen_events.add(r.get("Event_ID") or r.get("EventID"))
                seen_players.add(r.get("Player") or "")
            if len(pick)==req.legs:
                suggestions.append({"book": book, "legs": pick,
                                    "sumEVPct": round(sum(x.get("AvailableEVPct",0) for x in pick), 2)})
            i = j
        if len(suggestions) >= req.max_suggestions: break

    return {"parlays": suggestions}

# -----------------
# DubClub text builder (unchanged)
# -----------------
def fmt_amer(n):
    try:
        n = int(n)
        return f"{n:+d}" if n>0 else f"{n:d}"
    except Exception:
        return str(n)

def fmt_line(r)->str:
    start = r.get("Start Time (ET)") or r.get("Start_Time") or r.get("StartISO")
    betside = r.get("BetSide") or r.get("Bet") or ""
    point = r.get("Point")
    bet_label = f"{betside} {point}" if point not in (None,"") else betside
    seg = f"{r.get('Matchup','')} — {r.get('Player') or r.get('Outcome_Name','')} {r.get('Prop','')} {bet_label} {fmt_amer(r.get('Odds'))} ({(r.get('Best_Book') or r.get('Book') or '').upper()}) — {round(float(r.get('UnitSize',1.0)),2)}u"
    return seg + (f" | {start}" if start else "")

class DubReq(BaseModel):
    rows: List[Dict[str, Any]] = []
    title_date: str = ""
    sport_label: str = "Props"

@app.post("/post/dubclub")
def post_dubclub(body: DubReq):
    rows = body.rows or []
    header = f"ValueVaultHQ — {body.title_date or ''} Final Card\n\n"
    card = [f"**{body.sport_label}**\n"]
    for r in rows:
        card.append(fmt_line(r))
    footer = ("\n\nNotes:\n"
              "• Odds/lines can move; we post best widely-available numbers.\n"
              "• Informational only • 21+ • Play responsibly.")
    dubclub_text = header + "\n".join(card) + footer

    total_units = round(sum(float(r.get("UnitSize", 0)) for r in rows), 2)
    x_post = (f"Card is live 🔒  {body.sport_label}\n"
              f"{len(rows)} plays • {total_units}u on board\n"
              "Lineup-confirmed +EV. Details in bio/pinned.")
    return {"dubclub_text": dubclub_text, "x_post": x_post}

@app.get("/")
def root(): return {"ok": True}
