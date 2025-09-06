# app.py — ValueVault API (Props + Engine)
# Run locally: uvicorn app:app --host 0.0.0.0 --port 8000

import os
import time
from collections import defaultdict
from typing import List, Optional, Union, Tuple, Dict, Any

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone

API_NAME = "ValueVault API"
API_VERSION = "0.2.0"

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# ---------------------------
# FastAPI app & CORS
# ---------------------------
app = FastAPI(title=API_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers & tiny in-memory cache
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

def american_to_prob(odds: Optional[Union[int, float]]) -> Optional[float]:
    """Convert American odds to implied probability (0..1)."""
    if odds is None:
        return None
    try:
        o = int(odds)
    except Exception:
        return None
    if o > 0:
        return 100 / (o + 100)
    else:
        return -o / (-o + 100)

# ---------------------------
# Models
# ---------------------------

class PropsRawReq(BaseModel):
    api_key: Optional[str] = None
    sports: Union[str, List[str]]
    markets: Union[str, List[str]]
    books: Union[str, List[str]]
    regions: Union[str, List[str]] = "us"
    odds_format: str = "american"      # "american" | "decimal"
    days_from: int = 2
    cache_ttl_minutes: int = 2
    force: bool = False

class EngineScoreReq(BaseModel):
    api_key: Optional[str] = None

    sports: Union[str, List[str]]
    markets: Union[str, List[str]]
    books: Union[str, List[str]]
    regions: Union[str, List[str]] = "us"

    odds_format: str = "american"
    days_from: int = 2
    cache_ttl_minutes: int = 2
    force: bool = False

    # Engine filters
    min_win_pct: float = 50.0     # e.g., 56
    min_edge_pct: float = 0.0     # e.g., 0.5
    min_odds: int = -250          # american min
    max_odds: int = 800           # american max
    top_per_event: int = 3        # keep top-N by edge per event

# ---------------------------
# Root (health)
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
# Core: /props/raw
# ---------------------------

@app.post("/props/raw")
def props_raw(req: PropsRawReq):
    """
    Pull player prop markets using per-event endpoint to avoid INVALID_MARKET.
    Returns rows (one per outcome per book).
    """
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
        "props_raw",
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

    def keep_book(key: str) -> bool:
        return key.lower() in books

    for sport in sports:
        # events
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

        # per-event odds with markets filter
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
                        "sport": sport, "event_id": event_id,
                        "status": od_resp.status_code, "body": od_resp.text[:300],
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
                if not keep_book(bm.get("key", "")):
                    continue
                book_key = bm.get("key")
                for m in bm.get("markets", []):
                    market_key = m.get("key")
                    for out in m.get("outcomes", []):
                        player_name = out.get("description") or out.get("participant") or out.get("name")
                        side = out.get("name")  # "Over"/"Under" or "Yes"/"No"
                        line = out.get("point")
                        odds = out.get("price")

                        rows.append({
                            "start_iso": start_iso,
                            "sport_key": sport,
                            "matchup": f"{away} @ {home}" if home and away else None,
                            "market": market_key,
                            "player": player_name,
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
# Engine: /engine/mlb/score
# ---------------------------

@app.post("/engine/mlb/score")
def engine_mlb_score(req: EngineScoreReq):
    """
    Score props and return only "bet-worthy" rows.
    - Builds consensus model probability by de-vigging Over/Under pairs per book and averaging.
    - Computes win_pct and edge_pct per book offer.
    - Applies filters (min_win_pct, min_edge_pct, odds range, top_per_event).
    """
    odds_api_key = req.api_key or os.getenv("ODDS_API_KEY")
    if not odds_api_key:
        raise HTTPException(status_code=400, detail="No Odds API key provided or configured (ODDS_API_KEY).")

    sports = _as_list(req.sports)
    markets = _as_list(req.markets)
    books = [b.lower() for b in _as_list(req.books)]
    regions = [r.lower() for r in _as_list(req.regions)]
    if not sports or not markets or not books or not regions:
        raise HTTPException(status_code=422, detail="sports, markets, books, and regions are required (string or list).")

    # cache key includes filters because they change the output rows
    cache_key = (
        "engine_score",
        tuple(sorted(sports)),
        tuple(sorted(markets)),
        tuple(sorted(books)),
        tuple(sorted(regions)),
        req.odds_format,
        req.days_from,
        req.min_win_pct,
        req.min_edge_pct,
        req.min_odds,
        req.max_odds,
        req.top_per_event,
    )
    if not req.force:
        cached = _cache_get(cache_key, req.cache_ttl_minutes)
        if cached:
            return cached

    rows: List[dict] = []
    meta: dict = {
        "sports": sports, "markets": markets, "books": books, "regions": regions,
        "filters": {
            "min_win_pct": req.min_win_pct,
            "min_edge_pct": req.min_edge_pct,
            "min_odds": req.min_odds,
            "max_odds": req.max_odds,
            "top_per_event": req.top_per_event,
        }
    }

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    timeout = (10, 30)

    def keep_book(key: str) -> bool:
        return key.lower() in books

    def odds_in_range(od: Optional[Union[int, float]]) -> bool:
        if od is None:
            return False
        try:
            o = int(od)
        except Exception:
            return False
        return req.min_odds <= o <= req.max_odds

    for sport in sports:
        # 1) pull events
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
                        "sport": sport, "event_id": event_id,
                        "status": od_resp.status_code, "body": od_resp.text[:300],
                    })
                    continue
                od = od_resp.json()
            except requests.RequestException as e:
                meta.setdefault("errors", []).append({"sport": sport, "event_id": event_id, "error": str(e)})
                continue

            start_iso = od.get("commence_time") or ev.get("commence_time")
            home = od.get("home_team") or ev.get("home_team")
            away = od.get("away_team") or ev.get("away_team")
            matchup = f"{away} @ {home}" if home and away else None

            # 2) Build group for each (player, market, line) across books with Over/Under (or Yes/No)
            # groups[key]["Over"][book] = odds, groups[key]["Under"][book] = odds, etc.
            groups = defaultdict(lambda: {"Over": {}, "Under": {}, "Yes": {}, "No": {}})

            for bm in od.get("bookmakers", []):
                if not keep_book(bm.get("key", "")):
                    continue
                book_key = bm.get("key")
                for mkt in bm.get("markets", []):
                    market_key = mkt.get("key")
                    if market_key not in markets:
                        # Odds API sometimes returns superset; keep explicit
                        pass
                    for out in mkt.get("outcomes", []):
                        player_name = out.get("description") or out.get("participant") or out.get("name")
                        side = out.get("name")  # "Over"/"Under" or "Yes"/"No"
                        line = out.get("point")
                        price = out.get("price")
                        if player_name and side:
                            key = (player_name, market_key, line)
                            # store odds by side and book
                            if side in ("Over", "Under", "Yes", "No"):
                                groups[key][side][book_key] = price

            # 3) For each group, compute consensus probabilities using de-vig normalization per book,
            # then average across books.
            event_rows: List[dict] = []

            for (player_name, market_key, line), sides_dict in groups.items():
                # Determine if it's an Over/Under style or Yes/No style
                is_ou = len(sides_dict["Over"]) > 0 or len(sides_dict["Under"]) > 0
                is_yn = len(sides_dict["Yes"]) > 0 or len(sides_dict["No"]) > 0

                # Build per-book normalized probabilities (remove vig by dividing by sum)
                # We'll compute p_side across books where both sides exist at that book
                norm_p_over: List[float] = []
                norm_p_under: List[float] = []
                norm_p_yes: List[float] = []
                norm_p_no: List[float] = []

                if is_ou:
                    # set of books that have at least one side
                    book_keys = set(sides_dict["Over"].keys()) | set(sides_dict["Under"].keys())
                    for bk in book_keys:
                        o_odds = sides_dict["Over"].get(bk)
                        u_odds = sides_dict["Under"].get(bk)
                        o_p = american_to_prob(o_odds) if o_odds is not None else None
                        u_p = american_to_prob(u_odds) if u_odds is not None else None
                        if o_p is not None and u_p is not None and (o_p + u_p) > 0:
                            # de-vig normalize
                            total = o_p + u_p
                            norm_p_over.append(o_p / total)
                            norm_p_under.append(u_p / total)

                if is_yn:
                    book_keys = set(sides_dict["Yes"].keys()) | set(sides_dict["No"].keys())
                    for bk in book_keys:
                        y_odds = sides_dict["Yes"].get(bk)
                        n_odds = sides_dict["No"].get(bk)
                        y_p = american_to_prob(y_odds) if y_odds is not None else None
                        n_p = american_to_prob(n_odds) if n_odds is not None else None
                        if y_p is not None and n_p is not None and (y_p + n_p) > 0:
                            total = y_p + n_p
                            norm_p_yes.append(y_p / total)
                            norm_p_no.append(n_p / total)

                # consensus model probabilities (0..1)
                p_over = sum(norm_p_over) / len(norm_p_over) if norm_p_over else None
                p_under = sum(norm_p_under) / len(norm_p_under) if norm_p_under else None
                p_yes = sum(norm_p_yes) / len(norm_p_yes) if norm_p_yes else None
                p_no = sum(norm_p_no) / len(norm_p_no) if norm_p_no else None

                # Now produce a row PER available book offer, compute edge vs that book's implied prob
                def maybe_emit(side_name: str, book_to_odds: Dict[str, Any], p_model: Optional[float]):
                    if p_model is None:
                        return
                    for bk, price in book_to_odds.items():
                        if not odds_in_range(price):
                            continue
                        implied = american_to_prob(price)
                        if implied is None:
                            continue
                        edge = p_model - implied
                        win_pct = round(p_model * 100, 1)
                        edge_pct = round(edge * 100, 1)

                        # Filters
                        if win_pct < req.min_win_pct:
                            continue
                        if edge_pct < req.min_edge_pct:
                            continue

                        event_rows.append({
                            "start_iso": start_iso,
                            "sport_key": sport,
                            "matchup": matchup,
                            "market": market_key,
                            "player": player_name,
                            "bet_side": side_name,
                            "book": bk,
                            "line": line,
                            "odds": price,
                            "event_id": event_id,
                            "win_pct": win_pct,
                            "edge_pct": edge_pct,
                            # for debugging/curiosity:
                            "p_model": round(p_model, 4),
                        })

                if is_ou:
                    maybe_emit("Over", sides_dict["Over"], p_over)
                    maybe_emit("Under", sides_dict["Under"], p_under)
                if is_yn:
                    maybe_emit("Yes", sides_dict["Yes"], p_yes)
                    maybe_emit("No", sides_dict["No"], p_no)

            # 4) Keep top N by edge for this event
            if event_rows:
                event_rows.sort(key=lambda r: (r.get("edge_pct") or 0), reverse=True)
                rows.extend(event_rows[: max(req.top_per_event, 0)])

    payload = {"rows": rows, "meta": meta}
    _cache_set(cache_key, payload)
    return payload
