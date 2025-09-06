# app.py — ValueVault Props API (PROPS-first)
# Run locally: uvicorn app:app --host 0.0.0.0 --port 8000

import os
import time
import json
import requests
from typing import List, Optional, Union, Tuple, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone

API_NAME = "ValueVault Props API"
API_VERSION = "0.1.0"

# ---------------------------
# FastAPI app & CORS
# ---------------------------
app = FastAPI(title=f"{API_NAME} (PROPS-first)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # viewer.html file, localhost, etc.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Models & helpers
# ---------------------------

class PropsRawReq(BaseModel):
    # If provided, overrides the env var ODDS_API_KEY
    api_key: Optional[str] = None

    # Accept either a string or a list in the request body
    sports: Union[str, List[str]]
    markets: Union[str, List[str]]
    books: Union[str, List[str]]
    regions: Union[str, List[str]] = "us"

    # Optional knobs (safe defaults)
    odds_format: str = "american"      # "american" | "decimal"
    days_from: int = 2                  # how far ahead to scan events
    cache_ttl_minutes: int = 2          # in-memory cache TTL
    force: bool = False                 # ignore cache on this request


def _as_list(v: Union[str, List[str], None]) -> List[str]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# very small in-memory cache
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

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


@app.post("/props/raw")
def props_raw(req: PropsRawReq):
    """
    Pull player prop markets using the per-event endpoint to avoid INVALID_MARKET.
    Accepts single strings or CSV-like lists in JSON arrays.
    """

    # 1) API key resolution
    odds_api_key = req.api_key or os.getenv("ODDS_API_KEY")
    if not odds_api_key:
        raise HTTPException(status_code=400, detail="No Odds API key provided or configured (ODDS_API_KEY).")

    sports = _as_list(req.sports)
    markets = _as_list(req.markets)
    books = [b.lower() for b in _as_list(req.books)]
    regions = [r.lower() for r in _as_list(req.regions)]
    if not sports or not markets or not books or not regions:
        raise HTTPException(status_code=422, detail="sports, markets, books, and regions are required (string or list).")

    # 2) Cache key & lookup
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

    # 3) Fetch events per sport, then per-event odds with markets filter
    rows: List[dict] = []
    meta: dict = {"sports": sports, "markets": markets, "books": books, "regions": regions}

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    timeout = (10, 30)  # connect, read

    # Helper: ensure we only keep requested books
    def _keep_book(key: str) -> bool:
        return key.lower() in books

    for sport in sports:
        # 3a) list events for the sport
        # Docs: GET /v4/sports/{sport}/events?apiKey=...&daysFrom=N
        events_url = f"{ODDS_API_BASE}/sports/{sport}/events"
        params_events = {
            "apiKey": odds_api_key,
            "daysFrom": req.days_from,
        }
        try:
            ev_resp = session.get(events_url, params=params_events, timeout=timeout)
            if ev_resp.status_code != 200:
                raise HTTPException(status_code=ev_resp.status_code,
                                    detail=f"Events fetch failed for {sport}: {ev_resp.text}")
            events = ev_resp.json() or []
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Events fetch error for {sport}: {e}")

        # 3b) for each event, pull per-event odds with requested player markets
        for ev in events:
            event_id = ev.get("id")
            if not event_id:
                # Older accounts sometimes use 'id' or 'eventId'; handle defensively
                event_id = ev.get("eventId")
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
                    # event vanished or unavailable—skip quietly
                    continue
                if od_resp.status_code != 200:
                    # don't crash whole batch—record soft error in meta and continue
                    meta.setdefault("errors", []).append({
                        "sport": sport,
                        "event_id": event_id,
                        "status": od_resp.status_code,
                        "body": od_resp.text[:300],
                    })
                    continue
                od = od_resp.json()
            except requests.RequestException as e:
                meta.setdefault("errors", []).append({
                    "sport": sport,
                    "event_id": event_id,
                    "error": str(e),
                })
                continue

            # Expected schema (abridged):
            # {
            #   "id": "...",
            #   "commence_time": "2025-09-05T18:21:00Z",
            #   "home_team": "...", "away_team": "...",
            #   "bookmakers": [
            #     {"key":"fanduel","title":"FanDuel","markets":[
            #        {"key":"batter_hits","outcomes":[
            #           {"name":"Over","point":1.5,"price":-110,"description":"Seiya Suzuki"},
            #           {"name":"Under","point":1.5,"price":-115,"description":"Seiya Suzuki"}
            #        ]}, ...
            #     ]}
            #   ]
            # }

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
                        # Player name usually in "description" for player props
                        player_name = out.get("description") or out.get("participant") or out.get("name")
                        side = out.get("name")  # "Over", "Under", "Yes", "No"
                        line = out.get("point")
                        odds = out.get("price")  # american odds when oddsFormat=american

                        row = {
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
                        }
                        rows.append(row)

    payload = {"rows": rows, "meta": meta}
    _cache_set(cache_key, payload)
    return payload


# ---------------------------
# (Optional) stubs for future endpoints you showed in screenshots
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
# ====== ENGINE: MLB score (market devig now; model hooks ready) =================
from typing import List, Dict, Any, Union
import requests

class EngineScoreReq(BaseModel):
    sports: Union[str, List[str]]
    markets: Union[str, List[str]]
    books: Union[str, List[str]]
    regions: Union[str, List[str]] = "us"
    cache_ttl_minutes: int = 2
    force: bool = False
    # thresholds (bettable/simple-view)
    min_win_pct: float = 58.0
    min_edge_pct: float = 2.0
    min_odds: int = -250
    max_odds: int = 800
    top_per_event: int = 3
    # blend weights (we’ll use these when we add a true model)
    w_market: float = 0.7

class EngineScoreOut(BaseModel):
    rows: List[Dict[str, Any]]
    meta: Dict[str, Any]

def _aa_to_dec2(odds: int) -> float:
    return 1 + (odds/100.0) if odds >= 0 else 1 + (100.0/abs(odds))

def _impl_from_aa2(odds: int) -> float:
    return 1.0 / _aa_to_dec2(odds)

def _leg_base2(r: Dict[str, Any]) -> str:
    # event + market + player + line (ignore book & side)
    return "|".join([str(r.get("event_id","")), str(r.get("market","")),
                     str(r.get("name","")), str(r.get("line",""))])

def _devig_market_prob(rows: List[Dict[str, Any]]) -> None:
    """
    Adds to each row:
      p_market (0..1)
    Uses best Over vs Under on the same leg across books (de-vig).
    """
    best: Dict[str, Dict[str, int]] = {}  # base -> {'over':odds, 'under':odds}
    for r in rows:
        base = _leg_base2(r)
        side = str(r.get("bet_side","")).lower()
        bucket = best.get(base, {"over": None, "under": None})
        if side == "over":
            if bucket["over"] is None or _aa_to_dec2(r["odds"]) > _aa_to_dec2(bucket["over"]):
                bucket["over"] = r["odds"]
        else:
            if bucket["under"] is None or _aa_to_dec2(r["odds"]) > _aa_to_dec2(bucket["under"]):
                bucket["under"] = r["odds"]
        best[base] = bucket

    for r in rows:
        base = _leg_base2(r)
        bucket = best.get(base, {})
        over_odds = bucket.get("over")
        under_odds = bucket.get("under")

        p_over_fair = None
        if over_odds is not None and under_odds is not None:
            p_over_raw = _impl_from_aa2(over_odds)
            p_under_raw = _impl_from_aa2(under_odds)
            denom = p_over_raw + p_under_raw
            if denom > 0:
                p_over_fair = p_over_raw / denom

        if p_over_fair is None:
            # fallback if we only have one side from the books
            if over_odds is not None:
                p_over_fair = _impl_from_aa2(over_odds)
            elif under_odds is not None:
                p_over_fair = 1 - _impl_from_aa2(under_odds)
            else:
                p_over_fair = 0.5

        is_over = str(r.get("bet_side","")).lower() == "over"
        r["p_market"] = p_over_fair if is_over else (1 - p_over_fair)

def _keep_best_book_exact(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # exact identity includes side so we keep only the highest paying book per leg+side
    seen: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = _leg_base2(r) + "|" + str(r.get("bet_side","")).lower()
        prev = seen.get(k)
        if prev is None or _aa_to_dec2(r["odds"]) > _aa_to_dec2(prev["odds"]):
            seen[k] = r
    return list(seen.values())

@app.post("/engine/mlb/score", response_model=EngineScoreOut)
def engine_mlb_score(req: EngineScoreReq):
    """
    Step 1 engine:
      - pulls props via /props/raw
      - computes p_market (de-vig)
      - sets p_model = p_market (placeholder)
      - p_final = p_market (blend scaffold in place)
      - adds EV%, filters to bettable list
    """
    # normalize lists
    def _as_list2(v): return v if isinstance(v, list) else [v]
    sports  = _as_list2(req.sports)
    markets = _as_list2(req.markets)
    books   = _as_list2(req.books)
    regions = _as_list2(req.regions)

    # call our own /props/raw (reuses your cache & key)
    base = os.environ.get("PUBLIC_BASE_URL", "").strip()
    if not base:
        # If you haven’t set PUBLIC_BASE_URL yet, try default Render URL pattern.
        # Replace 'valuevault-api' if your service slug differs.
        base = "https://valuevault-api.onrender.com"

    payload = {
        "sports": sports, "markets": markets, "books": books, "regions": regions,
        "cache_ttl_minutes": req.cache_ttl_minutes, "force": req.force
    }
    try:
        rr = requests.post(f"{base}/props/raw", json=payload, timeout=90)
        if rr.status_code != 200:
            raise HTTPException(status_code=rr.status_code, detail=f"/props/raw failed: {rr.text[:300]}")
        rows = rr.json().get("rows", [])
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error calling /props/raw: {e}")

    if not rows:
        return {"rows": [], "meta": {"input_rows": 0, "returned": 0, "hidden": 0}}

    # market fair probability
    _devig_market_prob(rows)

    # model stub: for now mirror market; next step we will compute from MLB / weather
    for r in rows:
        r["p_model"] = r["p_market"]

    # blend (scaffold): p_final = w_market * p_market + (1 - w_market) * p_model
    w = max(0.0, min(1.0, req.w_market))
    for r in rows:
        p_final = w * r["p_market"] + (1 - w) * r["p_model"]
        r["p_final"] = p_final
        r["_win_prob"] = p_final
        r["_roi"] = p_final * _aa_to_dec2(int(r["odds"])) - 1.0

    # filter to bettable
    def _in_odds_range(o: Any) -> bool:
        try:
            v = int(o); return req.min_odds <= v <= req.max_odds
        except:
            return False

    filtered = [r for r in rows
                if r["_win_prob"] >= (req.min_win_pct/100.0)
                and r["_roi"] >= (req.min_edge_pct/100.0)
                and _in_odds_range(r.get("odds"))]

    # keep single best book per exact leg
    filtered = _keep_best_book_exact(filtered)

    # cap per event (keep the view tight)
    if req.top_per_event and req.top_per_event > 0:
        by_evt: Dict[str, List[Dict[str, Any]]] = {}
        for r in filtered:
            by_evt.setdefault(r.get("event_id",""), []).append(r)
        capped: List[Dict[str, Any]] = []
        for evt, lst in by_evt.items():
            lst.sort(key=lambda x: (x["_roi"], x["_win_prob"]), reverse=True)
            capped.extend(lst[:req.top_per_event])
        filtered = capped

    # final sort
    filtered.sort(key=lambda x: (x["_roi"], x["_win_prob"]), reverse=True)

    return {
        "rows": filtered,
        "meta": {
            "input_rows": len(rows),
            "returned": len(filtered),
            "hidden": max(0, len(rows)-len(filtered)),
            "note": "p_model currently mirrors p_market (placeholder). Next step: add MLB stats & weather."
        }
    }
# ====== END ENGINE ============================================================
