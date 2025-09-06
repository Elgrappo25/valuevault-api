# app.py — ValueVault Props API (PROPS-first + Engine consensus edge)
# Run locally: uvicorn app:app --host 0.0.0.0 --port 8000

import os
import time
import math
import requests
from typing import List, Optional, Union, Tuple, Dict, Any, DefaultDict
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

API_NAME = "ValueVault Props API"
API_VERSION = "0.1.1"

# ---------------------------
# FastAPI app & CORS
# ---------------------------
app = FastAPI(title=f"{API_NAME} (PROPS-first)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Small helpers / cache
# ---------------------------

def _as_list(v: Union[str, List[str], None]) -> List[str]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

_cache: Dict[Tuple, Dict[str, Any]] = {}  # key -> {"ts": float, "payload": dict}

def _cache_get(key: Tuple, ttl_minutes: int) -> Optional[dict]:
    if ttl_minutes <= 0:
        return None
    item = _cache.get(key)
    if not item:
        return None
    if (time.time() - item["ts"]) <= ttl_minutes * 60:
        return item["payload"]
    return None

def _cache_set(key: Tuple, payload: dict) -> None:
    _cache[key] = {"ts": time.time(), "payload": payload}

def american_to_prob(odds: Optional[float]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = float(odds)
    except Exception:
        return None
    if o == 0:
        return None
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)

def two_way_no_vig(p_over_raw: Optional[float], p_under_raw: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    Remove vig for a 2-way market when we have both sides' raw implied probs.
    Returns (p_over_novig, p_under_novig). If one side missing, returns (None, None).
    """
    if p_over_raw is None or p_under_raw is None:
        return None, None
    s = p_over_raw + p_under_raw
    if s <= 0:
        return None, None
    # scale so they sum to 1
    return p_over_raw / s, p_under_raw / s

def median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0

# ---------------------------
# Root
# ---------------------------

@app.get("/")
def root():
    return {
        "name": API_NAME,
        "version": API_VERSION,
        "time_utc": _now_utc_iso(),
        "health": "ok",
    }

# ---------------------------
# /props/raw (unchanged)
# ---------------------------

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

class PropsRawReq(BaseModel):
    api_key: Optional[str] = None
    sports: Union[str, List[str]]
    markets: Union[str, List[str]]
    books: Union[str, List[str]]
    regions: Union[str, List[str]] = "us"
    odds_format: str = "american"
    days_from: int = 2
    cache_ttl_minutes: int = 2
    force: bool = False

@app.post("/props/raw")
def props_raw(req: PropsRawReq):
    odds_api_key = req.api_key or os.getenv("ODDS_API_KEY")
    if not odds_api_key:
        raise HTTPException(status_code=400, detail="No Odds API key provided or configured (ODDS_API_KEY).")

    sports = _as_list(req.sports)
    markets = _as_list(req.markets)
    books = [b.lower() for b in _as_list(req.books)]
    regions = [r.lower() for r in _as_list(req.regions)]
    if not sports or not markets or not books or not regions:
        raise HTTPException(status_code=422, detail="sports, markets, books, and regions are required (string or list).")

    cache_key = (
        tuple(sorted(sports)),
        tuple(sorted(markets)),
        tuple(sorted(books)),
        tuple(sorted(regions)),
        req.odds_format,
        req.days_from,
    )
    if not req.force:
        cached = _cache_get(cache_key, req.cache_ttl_minutes)
        if cached:
            return cached

    rows: List[dict] = []
    meta: dict = {"sports": sports, "markets": markets, "books": books, "regions": regions}

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    timeout = (10, 30)

    def _keep_book(key: str) -> bool:
        return key.lower() in books

    for sport in sports:
        events_url = f"{ODDS_API_BASE}/sports/{sport}/events"
        params_events = {"apiKey": odds_api_key, "daysFrom": req.days_from}
        try:
            ev_resp = session.get(events_url, params=params_events, timeout=timeout)
            if ev_resp.status_code != 200:
                raise HTTPException(status_code=ev_resp.status_code,
                                    detail=f"Events fetch failed for {sport}: {ev_resp.text}")
            events = ev_resp.json() or []
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Events fetch error for {sport}: {e}")

        for ev in events:
            event_id = ev.get("id") or ev.get("eventId")
            if not event_id:
                continue

            odds_url = f"{ODDS_API_BASE}/sports/{sport}/events/{event_id}/odds"
            params_odds = {
                "apiKey": odds_api_key,
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": req.odds_format,
            }
            try:
                od_resp = session.get(odds_url, params=params_odds, timeout=timeout)
                if od_resp.status_code == 404:
                    continue
                if od_resp.status_code != 200:
                    meta.setdefault("errors", []).append({
                        "sport": sport,
                        "event_id": event_id,
                        "status": od_resp.status_code,
                        "body": od_resp.text[:300],
                    })
                    continue
                od = od_resp.json()
            except requests.RequestException as e:
                meta.setdefault("errors", []).append({"sport": sport, "event_id": event_id, "error": str(e)})
                continue

            start_iso = od.get("commence_time") or ev.get("commence_time")
            home = od.get("home_team") or ev.get("home_team")
            away = od.get("away_team") or ev.get("away_team")

            for bm in od.get("bookmakers", []):
                if not _keep_book(bm.get("key", "")):
                    continue
                book_key = bm.get("key")
                for m in bm.get("markets", []):
                    market_key = m.get("key")
                    for out in m.get("outcomes", []):
                        player_name = out.get("description") or out.get("participant") or out.get("name")
                        side = out.get("name")  # Over/Under/Yes/No
                        line = out.get("point")
                        odds = out.get("price")
                        rows.append({
                            "start_iso": start_iso,
                            "sport_key": sport,
                            "matchup": f"{away} @ {home}" if home and away else None,
                            "market": market_key,
                            "name": player_name,
                            "bet_side": side,
                            "book": book_key,
                            "line": line,
                            "odds": odds,
                            "event_id": event_id,
                        })

    payload = {"rows": rows, "meta": meta}
    _cache_set(cache_key, payload)
    return payload

# ---------------------------
# Engine: MLB consensus/no-vig model for Win% and Edge%
# ---------------------------

class EngineScoreReq(BaseModel):
    sports: Union[str, List[str]]                     # e.g., "baseball_mlb"
    markets: Union[str, List[str]]                    # e.g., "batter_hits"
    books: Union[str, List[str]]                      # CSV or list of book keys
    regions: Union[str, List[str]] = "us"
    days_from: int = 2                                # how far ahead to scan
    min_win_pct: float = 56.0                         # show only if model win% >= this
    min_edge_pct: float = 0.5                         # show only if edge% >= this
    min_odds: int = -250                              # american odds guard
    max_odds: int = 800
    top_per_event: int = 5
    cache_ttl_minutes: int = 2
    force: bool = False

@app.post("/engine/mlb/score")
def engine_mlb_score(req: EngineScoreReq):
    """
    Consensus/no-vig model:
      - Build Over/Under pairs per (event, player, line, market, book).
      - Convert to no-vig probs per book.
      - Model probability = median of book no-vig probs (consensus).
      - Win% = model probability, Edge% = (model - this book no-vig) * 100.
    """
    odds_api_key = os.getenv("ODDS_API_KEY")
    if not odds_api_key:
        raise HTTPException(status_code=400, detail="No Odds API key provided or configured (ODDS_API_KEY).")

    sports = _as_list(req.sports)
    markets = _as_list(req.markets)
    books = [b.lower() for b in _as_list(req.books)]
    regions = [r.lower() for r in _as_list(req.regions)]
    if not sports or not markets or not books or not regions:
        raise HTTPException(status_code=422, detail="sports, markets, books, and regions are required.")

    # cache on request params
    cache_key = ("engine_mlb_score",
                 tuple(sorted(sports)), tuple(sorted(markets)),
                 tuple(sorted(books)), tuple(sorted(regions)),
                 req.days_from)
    if not req.force:
        cached = _cache_get(cache_key, req.cache_ttl_minutes)
        if cached:
            rows = _filter_rank_and_trim(cached["rows"], req)
            return {"rows": rows, "meta": cached["meta"]}

    meta: dict = {"sports": sports, "markets": markets, "books": books, "regions": regions}
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    timeout = (10, 30)

    # 1) Pull events and odds, aggregate by group
    # group key: (event_id, market_key, player_name, line)
    # store per book: {"over": price, "under": price}
    groups: DefaultDict[Tuple, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    event_info: Dict[str, Dict[str, Any]] = {}  # event_id -> {"start_iso","matchup","sport"}

    for sport in sports:
        # events list
        events_url = f"{ODDS_API_BASE}/sports/{sport}/events"
        params_events = {"apiKey": odds_api_key, "daysFrom": req.days_from}
        ev_resp = session.get(events_url, params=params_events, timeout=timeout)
        if ev_resp.status_code != 200:
            raise HTTPException(status_code=ev_resp.status_code,
                                detail=f"Events fetch failed for {sport}: {ev_resp.text}")
        events = ev_resp.json() or []

        for ev in events:
            event_id = ev.get("id") or ev.get("eventId")
            if not event_id:
                continue
            start_iso = ev.get("commence_time")
            home = ev.get("home_team")
            away = ev.get("away_team")
            event_info[event_id] = {
                "start_iso": start_iso,
                "matchup": f"{away} @ {home}" if home and away else None,
                "sport_key": sport,
            }

            odds_url = f"{ODDS_API_BASE}/sports/{sport}/events/{event_id}/odds"
            params_odds = {
                "apiKey": odds_api_key,
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": "american",
            }
            od_resp = session.get(odds_url, params=params_odds, timeout=timeout)
            if od_resp.status_code == 404:
                continue
            if od_resp.status_code != 200:
                meta.setdefault("errors", []).append({
                    "sport": sport, "event_id": event_id,
                    "status": od_resp.status_code, "body": od_resp.text[:300]
                })
                continue
            od = od_resp.json() or {}

            for bm in od.get("bookmakers", []):
                book_key = bm.get("key", "").lower()
                if book_key not in books:
                    continue
                for m in bm.get("markets", []):
                    market_key = m.get("key")
                    for out in m.get("outcomes", []):
                        player_name = out.get("description") or out.get("participant") or out.get("name")
                        side = (out.get("name") or "").lower()  # "over"/"under"
                        line = out.get("point")
                        price = out.get("price")
                        if side not in ("over", "under"):
                            continue
                        gkey = (event_id, market_key, player_name, line)
                        groups[gkey][book_key][side] = price

    # 2) Build rows with model & edge
    rows: List[dict] = []
    for gkey, by_book in groups.items():
        event_id, market_key, player_name, line = gkey

        # no-vig probs per book for this side
        over_probs = []
        under_probs = []
        per_book_prob = {}  # book -> {"over": p, "under": p}, no-vig

        for book_key, sides in by_book.items():
            p_over_raw = american_to_prob(sides.get("over"))
            p_under_raw = american_to_prob(sides.get("under"))
            p_over_nv, p_under_nv = two_way_no_vig(p_over_raw, p_under_raw)
            if p_over_nv is None or p_under_nv is None:
                continue
            per_book_prob[book_key] = {"over": p_over_nv, "under": p_under_nv}
            over_probs.append(p_over_nv)
            under_probs.append(p_under_nv)

        if not per_book_prob:
            continue

        # consensus model = median across books (more robust than mean)
        p_model_over = median(over_probs)
        p_model_under = median(under_probs)
        if p_model_over is None or p_model_under is None:
            continue

        # produce one row per book & side (so you can see which books are off-consensus)
        for book_key, sides in by_book.items():
            # only if we had a no-vig pair for this book
            if book_key not in per_book_prob:
                continue

            # raw displayed odds for each side from this book (for odds filters & UX)
            price_over = sides.get("over")
            price_under = sides.get("under")

            # odds guards
            def _keep(price: Optional[float]) -> bool:
                return (price is not None) and (req.min_odds <= price <= req.max_odds)

            # OVER row
            if _keep(price_over):
                p_book = per_book_prob[book_key]["over"]
                p_model = p_model_over
                edge_pct = (p_model - p_book) * 100.0
                win_pct = p_model * 100.0
                if win_pct >= req.min_win_pct and edge_pct >= req.min_edge_pct:
                    info = event_info.get(event_id, {})
                    rows.append({
                        "start_iso": info.get("start_iso"),
                        "sport_key": info.get("sport_key"),
                        "matchup": info.get("matchup"),
                        "market": market_key,
                        "player": player_name,
                        "side": "Over",
                        "book": book_key,
                        "line": line,
                        "odds": price_over,
                        "win_pct": round(win_pct, 1),
                        "edge_pct": round(edge_pct, 2),
                        "event_id": event_id,
                    })

            # UNDER row
            if _keep(price_under):
                p_book = per_book_prob[book_key]["under"]
                p_model = p_model_under
                edge_pct = (p_model - p_book) * 100.0
                win_pct = p_model * 100.0
                if win_pct >= req.min_win_pct and edge_pct >= req.min_edge_pct:
                    info = event_info.get(event_id, {})
                    rows.append({
                        "start_iso": info.get("start_iso"),
                        "sport_key": info.get("sport_key"),
                        "matchup": info.get("matchup"),
                        "market": market_key,
                        "player": player_name,
                        "side": "Under",
                        "book": book_key,
                        "line": line,
                        "odds": price_under,
                        "win_pct": round(win_pct, 1),
                        "edge_pct": round(edge_pct, 2),
                        "event_id": event_id,
                    })

    base = {"rows": rows, "meta": meta}
    _cache_set(cache_key, base)

    # final trim: rank by edge within event & limit top_per_event
    rows = _filter_rank_and_trim(rows, req)
    return {"rows": rows, "meta": meta}

def _filter_rank_and_trim(rows: List[dict], req: EngineScoreReq) -> List[dict]:
    # group by event and pick top by edge
    by_event: DefaultDict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)
    trimmed: List[dict] = []
    for ev, lst in by_event.items():
        best = sorted(lst, key=lambda x: x["edge_pct"], reverse=True)[: max(1, req.top_per_event)]
        trimmed.extend(best)
    # global sort for stable table
    trimmed.sort(key=lambda x: (x["start_iso"] or "", -x["edge_pct"]))
    return trimmed

# ---------------------------
# stubs unchanged
# ---------------------------

@app.post("/props/model")
def props_model_stub():
    return {"ok": True, "msg": "stub — implement later"}

@app.post("/parlay/suggest")
def parlay_suggest_stub():
    return {"ok": True, "msg": "stub — implement later"}

@app.post("/post/dubclub")
def post_dubclub_stub():
    return {"ok": True, "msg": "stub — implement later"}
