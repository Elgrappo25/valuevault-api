# app.py — ValueVault API (Snapshot Engine: one pull, many filters)
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
API_VERSION = "0.3.0"

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
# Helpers
# ---------------------------

def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _as_list(v: Union[str, List[str], None]) -> List[str]:
    if v is None:
        return []
    return v if isinstance(v, list) else [v]

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

def _key_tuple(items: List[str]) -> Tuple[str, ...]:
    return tuple(sorted([str(x).lower() for x in items]))

# ---------------------------
# Snapshot storage (in-memory)
# ---------------------------
# A snapshot is the big "scored table" we can filter without new API calls.
# Keyed by: (sports, markets, books, regions, days_from)
SnapshotKey = Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], int]
SNAPSHOTS: Dict[SnapshotKey, Dict[str, Any]] = {}
# value: {"ts": float, "rows": List[dict], "meta": {...}}

def get_snapshot(key: SnapshotKey, ttl_minutes: int) -> Optional[Dict[str, Any]]:
    snap = SNAPSHOTS.get(key)
    if not snap:
        return None
    if ttl_minutes <= 0:
        return snap
    if (time.time() - snap["ts"]) <= ttl_minutes * 60:
        return snap
    return None

def set_snapshot(key: SnapshotKey, rows: List[dict], meta: dict):
    SNAPSHOTS[key] = {"ts": time.time(), "rows": rows, "meta": meta}

# ---------------------------
# Models
# ---------------------------

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

class EngineScoreReq(BaseModel):
    api_key: Optional[str] = None
    sports: Union[str, List[str]]
    markets: Union[str, List[str]]
    books: Union[str, List[str]]
    regions: Union[str, List[str]] = "us"

    odds_format: str = "american"
    days_from: int = 2

    # SNAPSHOT knobs
    use_snapshot: bool = True           # if True, reuse snapshot instead of re-pulling
    snapshot_ttl_minutes: int = 10      # how long a snapshot is considered fresh
    force: bool = False                 # if True, rebuild the snapshot now (one pull)

    # FILTER knobs (do not affect snapshot — only what we return)
    min_win_pct: float = 50.0
    min_edge_pct: float = 0.0
    min_odds: int = -250
    max_odds: int = 800
    top_per_event: int = 5

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
# Core: /props/raw  (unchanged raw pull)
# ---------------------------

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

    rows: List[dict] = []
    meta: dict = {"sports": sports, "markets": markets, "books": books, "regions": regions}
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    timeout = (10, 30)

    def keep_book(key: str) -> bool:
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

    return {"rows": rows, "meta": meta}

# ---------------------------
# SNAPSHOT Engine (one pull, many filters)
# ---------------------------

def _build_scored_snapshot(
    odds_api_key: str,
    sports: List[str],
    markets: List[str],
    books: List[str],
    regions: List[str],
    days_from: int,
    odds_format: str = "american",
) -> Tuple[List[dict], dict]:
    """
    Pull odds once and compute a scored table (win_pct, edge_pct per offer).
    Returns (rows, meta) to be stored as a snapshot.
    """
    rows: List[dict] = []
    meta: dict = {"sports": sports, "markets": markets, "books": books, "regions": regions, "days_from": days_from}

    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    timeout = (10, 30)

    def keep_book(key: str) -> bool:
        return key.lower() in books

    for sport in sports:
        # 1) events
        events_url = f"{ODDS_API_BASE}/sports/{sport}/events"
        params_events = {"apiKey": odds_api_key, "daysFrom": days_from}
        try:
            ev_resp = session.get(events_url, params=params_events, timeout=timeout)
            if ev_resp.status_code != 200:
                meta.setdefault("errors", []).append({"sport": sport, "stage": "events", "status": ev_resp.status_code})
                continue
            events = ev_resp.json() or []
        except requests.RequestException as e:
            meta.setdefault("errors", []).append({"sport": sport, "stage": "events", "error": str(e)})
            continue

        for ev in events:
            event_id = ev.get("id") or ev.get("eventId")
            if not event_id:
                continue

            odds_url = f"{ODDS_API_BASE}/sports/{sport}/events/{event_id}/odds"
            params_odds = {
                "apiKey": odds_api_key,
                "regions": ",".join(regions),
                "markets": ",".join(markets),
                "oddsFormat": odds_format,
            }
            try:
                od_resp = session.get(odds_url, params=params_odds, timeout=timeout)
                if od_resp.status_code == 404:
                    continue
                if od_resp.status_code != 200:
                    meta.setdefault("errors", []).append({
                        "sport": sport, "stage": "per_event_odds", "event_id": event_id,
                        "status": od_resp.status_code
                    })
                    continue
                od = od_resp.json()
            except requests.RequestException as e:
                meta.setdefault("errors", []).append({"sport": sport, "stage": "per_event_odds", "event_id": event_id, "error": str(e)})
                continue

            start_iso = od.get("commence_time") or ev.get("commence_time")
            home = od.get("home_team") or ev.get("home_team")
            away = od.get("away_team") or ev.get("away_team")
            matchup = f"{away} @ {home}" if home and away else None

            # Build groups by (player, market, line)
            groups = defaultdict(lambda: {"Over": {}, "Under": {}, "Yes": {}, "No": {}})

            for bm in od.get("bookmakers", []):
                if not keep_book(bm.get("key", "")):
                    continue
                book_key = bm.get("key")
                for mkt in bm.get("markets", []):
                    market_key = mkt.get("key")
                    for out in mkt.get("outcomes", []):
                        player_name = out.get("description") or out.get("participant") or out.get("name")
                        side = out.get("name")
                        line = out.get("point")
                        price = out.get("price")
                        if player_name and side in ("Over","Under","Yes","No"):
                            key = (player_name, market_key, line)
                            groups[key][side][book_key] = price

            # For each group, devig per-book and average across books to get p_model
            def consensus_from_pair(side_a: Dict[str, Any], side_b: Dict[str, Any]) -> Optional[Tuple[float, float]]:
                """Return (p_a, p_b) in [0..1] averaged across books, or (None, None)."""
                norm_a: List[float] = []
                norm_b: List[float] = []
                bk_all = set(side_a.keys()) | set(side_b.keys())
                for bk in bk_all:
                    a_od, b_od = side_a.get(bk), side_b.get(bk)
                    pa = american_to_prob(a_od) if a_od is not None else None
                    pb = american_to_prob(b_od) if b_od is not None else None
                    if pa is None or pb is None:
                        continue
                    total = pa + pb
                    if total > 0:
                        norm_a.append(pa / total)
                        norm_b.append(pb / total)
                if not norm_a and not norm_b:
                    return None, None
                p_a = sum(norm_a)/len(norm_a) if norm_a else None
                p_b = sum(norm_b)/len(norm_b) if norm_b else None
                return p_a, p_b

            event_rows: List[dict] = []

            for (player_name, market_key, line), sides in groups.items():
                # OU
                p_over, p_under = consensus_from_pair(sides["Over"], sides["Under"])
                # YN
                p_yes, p_no = consensus_from_pair(sides["Yes"], sides["No"])

                def emit(side_label: str, book_to_odds: Dict[str, Any], p_model: Optional[float]):
                    if p_model is None:
                        return
                    for bk, price in book_to_odds.items():
                        implied = american_to_prob(price)
                        if implied is None:
                            continue
                        edge = p_model - implied
                        event_rows.append({
                            "start_iso": start_iso,
                            "sport_key": sport,
                            "matchup": matchup,
                            "market": market_key,
                            "player": player_name,
                            "bet_side": side_label,
                            "book": bk,
                            "line": line,
                            "odds": price,
                            "event_id": event_id,
                            "win_pct": round(p_model * 100, 1),
                            "edge_pct": round(edge * 100, 1),
                            "p_model": round(p_model, 4),
                        })

                emit("Over", sides["Over"], p_over)
                emit("Under", sides["Under"], p_under)
                emit("Yes", sides["Yes"], p_yes)
                emit("No", sides["No"], p_no)

            # keep everything in snapshot; top-per-event is applied later at filter time
            rows.extend(event_rows)

    return rows, meta

def _filter_and_rank(rows: List[dict], min_win_pct: float, min_edge_pct: float, min_odds: int, max_odds: int, top_per_event: int) -> List[dict]:
    # filter
    filtered: List[dict] = []
    for r in rows:
        odds = r.get("odds")
        try:
            o = int(odds) if odds is not None else None
        except Exception:
            o = None
        if o is None:
            continue
        if not (min_odds <= o <= max_odds):
            continue
        win = r.get("win_pct")
        edge = r.get("edge_pct")
        if win is None or edge is None:
            continue
        if win < min_win_pct or edge < min_edge_pct:
            continue
        filtered.append(r)

    # rank: Win % desc, then Edge % desc
    filtered.sort(key=lambda x: ((x.get("win_pct") or 0), (x.get("edge_pct") or 0)), reverse=True)

    # apply top-per-event
    by_event: Dict[str, List[dict]] = defaultdict(list)
    for r in filtered:
        ev = str(r.get("event_id") or "")
        by_event[ev].append(r)

    out: List[dict] = []
    cap = max(int(top_per_event or 0), 0)
    for ev, lst in by_event.items():
        if cap:
            out.extend(lst[:cap])
        else:
            out.extend(lst)
    return out

@app.post("/engine/mlb/score")
def engine_score(req: EngineScoreReq):
    """
    One pull (snapshot), many filters.
    - If use_snapshot and a fresh snapshot exists → reuse it.
    - Else (or force=True) → pull once now, build snapshot.
    - Then apply filters and return "bet-worthy" rows.
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

    snap_key: SnapshotKey = (_key_tuple(sports), _key_tuple(markets), _key_tuple(books), _key_tuple(regions), int(req.days_from))

    rows: List[dict]
    meta: dict

    snap = get_snapshot(snap_key, req.snapshot_ttl_minutes) if req.use_snapshot and not req.force else None
    if snap:
        rows = snap["rows"]
        meta = dict(snap.get("meta", {}))
        meta["snapshot_reused"] = True
        meta["snapshot_ts"] = snap["ts"]
    else:
        # Build a fresh snapshot now (one Odds API pull)
        rows, meta = _build_scored_snapshot(
            odds_api_key=odds_api_key,
            sports=sports, markets=markets, books=books, regions=regions,
            days_from=req.days_from, odds_format=req.odds_format,
        )
        set_snapshot(snap_key, rows, meta)
        meta = dict(meta)
        meta["snapshot_reused"] = False
        meta["snapshot_ts"] = SNAPSHOTS[snap_key]["ts"]

    # Now filter & rank without touching external API
    out_rows = _filter_and_rank(
        rows=rows,
        min_win_pct=req.min_win_pct,
        min_edge_pct=req.min_edge_pct,
        min_odds=req.min_odds,
        max_odds=req.max_odds,
        top_per_event=req.top_per_event,
    )

    return {"rows": out_rows, "meta": meta}

@app.get("/engine/mlb/snapshot/status")
def snapshot_status(
    sports: str,
    markets: str,
    books: str,
    regions: str,
    days_from: int = 2,
    snapshot_ttl_minutes: int = 10
):
    """
    Check whether a snapshot exists & freshness for given scope.
    Query params are CSV: sports, markets, books, regions
    """
    s = _key_tuple([x.strip() for x in sports.split(",") if x.strip()])
    m = _key_tuple([x.strip() for x in markets.split(",") if x.strip()])
    b = _key_tuple([x.strip() for x in books.split(",") if x.strip()])
    r = _key_tuple([x.strip() for x in regions.split(",") if x.strip()])
    key: SnapshotKey = (s, m, b, r, int(days_from))
    snap = SNAPSHOTS.get(key)
    fresh = False
    if snap:
        if snapshot_ttl_minutes <= 0:
            fresh = True
        else:
            fresh = (time.time() - snap["ts"]) <= snapshot_ttl_minutes * 60
    return {
        "exists": bool(snap),
        "fresh": fresh,
        "ts": snap["ts"] if snap else None,
        "time_utc": _now_utc_iso(),
        "key": {
            "sports": list(s),
            "markets": list(m),
            "books": list(b),
            "regions": list(r),
            "days_from": days_from
        }
    }
