# app.py — ValueVault Props API (props-first, per-event odds fix)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import requests
from datetime import datetime, timezone

# -----------------------------------------------------------------------------
# App & CORS
# -----------------------------------------------------------------------------

app = FastAPI(
    title="ValueVault Props API",
    version="0.1.0",
    description="Props-first backend that pulls player props via per-event endpoints"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock down to your domains if you want
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Odds API helpers (per-event props)
# -----------------------------------------------------------------------------

ODDS_BASE = "https://api.the-odds-api.com/v4"


def _get_today_event_ids(api_key: str, sport_key: str, tz: str = "America/New_York") -> List[str]:
    """
    Fetch today's event IDs for a sport.
    Uses /sports/{sport}/events (NOT the odds endpoint).
    """
    url = f"{ODDS_BASE}/sports/{sport_key}/events"
    params = {
        "apiKey": api_key,
        "dateFormat": "iso",
        "daysFrom": 0,   # today
        "daysTo": 0,     # today
        "tz": tz,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    events = r.json() or []
    return [e["id"] for e in events if isinstance(e, dict) and "id" in e]


def _get_props_for_event(
    api_key: str,
    sport_key: str,
    event_id: str,
    markets: List[str],
    regions: str,
    books: List[str]
) -> Dict[str, Any]:
    """
    Fetch player-prop odds for a single event using:
    /sports/{sport}/events/{eventId}/odds?markets=...&regions=...
    """
    url = f"{ODDS_BASE}/sports/{sport_key}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "markets": ",".join(markets),
        "regions": regions,
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    if books:
        params["bookmakers"] = ",".join(books)

    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def _normalize_game_props(game: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten a single event's props into rows ready for sheets / modeling.
    """
    rows: List[Dict[str, Any]] = []
    if not isinstance(game, dict):
        return rows

    event_id = game.get("id")
    commence = game.get("commence_time")
    home = game.get("home_team")
    away = game.get("away_team")

    for bm in game.get("bookmakers", []) or []:
        book = bm.get("title")
        for mkt in bm.get("markets", []) or []:
            market_key = mkt.get("key")
            for out in mkt.get("outcomes", []) or []:
                rows.append({
                    "event_id": event_id,
                    "start_iso": commence,
                    "home": home,
                    "away": away,
                    "book": book,
                    "market": market_key,
                    "name": out.get("description") or out.get("name"),  # player name or label
                    "bet_side": out.get("name"),                        # Over/Under/Yes/No
                    "line": out.get("point"),
                    "odds": out.get("price"),
                })
    return rows


def fetch_props_raw_for_sport(
    api_key: str,
    sport_key: str,
    markets: List[str],
    books: List[str],
    regions: str,
    tz: str = "America/New_York"
) -> List[Dict[str, Any]]:
    """
    Gather props across today's events for the given sport.
    """
    rows: List[Dict[str, Any]] = []
    event_ids = _get_today_event_ids(api_key, sport_key, tz)
    for eid in event_ids:
        data = _get_props_for_event(api_key, sport_key, eid, markets, regions, books)
        if not data:
            continue
        # per-event endpoint can return a dict (single game) or list (rare)
        games = [data] if isinstance(data, dict) else data
        for g in games:
            rows.extend(_normalize_game_props(g))
    return rows


# -----------------------------------------------------------------------------
# Pydantic models (requests / responses)
# -----------------------------------------------------------------------------

class PropsRawReq(BaseModel):
    api_key: str = Field(..., description="The Odds API key")
    sports: List[str] = Field(..., description="e.g., ['baseball_mlb']")
    markets: List[str] = Field(..., description="e.g., ['batter_hits','batter_total_bases']")
    books: List[str] = Field(default_factory=list, description="e.g., ['fanduel','draftkings']")
    regions: str = Field("us", description="Odds API regions, e.g., 'us'")
    tz: str = Field("America/New_York", description="Time zone for event listing")


class PropsRawResp(BaseModel):
    rows: List[Dict[str, Any]]
    meta: Dict[str, Any]


class ModelReq(BaseModel):
    rows: List[Dict[str, Any]]  # input rows from /props/raw
    # Optional knobs for modeling (simple baseline)
    min_odds: Optional[int] = None  # e.g., exclude odds shorter than -140, etc.


class ParlayReq(BaseModel):
    rows: List[Dict[str, Any]]        # rows with at least event_id, name, market, odds, book
    parlay_size: int = 3              # how many legs
    max_same_game: int = 1            # prevent too many legs from same event_id
    max_per_player: int = 1           # prevent stacking the same player
    distinct_markets_only: bool = True


class ParlayResp(BaseModel):
    picks: List[Dict[str, Any]]
    meta: Dict[str, Any]


class DubReq(BaseModel):
    picks: List[Dict[str, Any]]
    header: Optional[str] = "Tonight's Props"
    footer: Optional[str] = "#ValueVault"


# -----------------------------------------------------------------------------
# Utility: American odds -> implied probability (no vig removal)
# -----------------------------------------------------------------------------

def american_to_prob(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = int(odds)
    except Exception:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "ValueVault Props API",
        "version": app.version,
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "health": "ok"
    }


@app.post("/props/raw", response_model=PropsRawResp)
def props_raw(req: PropsRawReq):
    """
    Pull player prop markets using the **per-event** endpoint.
    Fixes INVALID_MARKET that happens on the game-odds endpoint.
    """
    try:
        all_rows: List[Dict[str, Any]] = []
        for sport_key in req.sports:
            rows = fetch_props_raw_for_sport(
                api_key=req.api_key,
                sport_key=sport_key,
                markets=req.markets,
                books=req.books,
                regions=req.regions,
                tz=req.tz,
            )
            # tag sport so downstream filters are easy
            for r in rows:
                r["sport_key"] = sport_key
            all_rows.extend(rows)

        return {
            "rows": all_rows,
            "meta": {
                "count": len(all_rows),
                "sports": req.sports,
                "markets": req.markets,
                "books": req.books,
                "regions": req.regions,
            },
        }
    except requests.HTTPError as e:
        # Bubble up Odds API errors cleanly
        detail = getattr(e.response, "text", str(e))
        raise HTTPException(status_code=e.response.status_code if e.response else 502, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/props/model")
def props_model(req: ModelReq):
    """
    Very simple baseline "model": add implied prob from offered odds,
    optionally filter by min_odds, and surface a 'model_win_pct'.
    (You can replace with a stronger model later.)
    """
    rows_out: List[Dict[str, Any]] = []
    for r in req.rows:
        odds = r.get("odds")
        if req.min_odds is not None and isinstance(odds, int):
            # Example filter: keep only odds >= min_odds if positive,
            # or odds <= min_odds if negative; to keep it simple we skip here.
            pass

        p = american_to_prob(odds)
        r2 = dict(r)
        r2["model_win_pct"] = round(float(p) * 100.0, 2) if p is not None else None
        rows_out.append(r2)

    return {
        "rows": rows_out,
        "meta": {"count": len(rows_out)}
    }


def _pick_greedy_parlay(
    rows: List[Dict[str, Any]],
    size: int,
    max_same_game: int,
    max_per_player: int,
    distinct_markets_only: bool
) -> List[Dict[str, Any]]:
    """
    Simple greedy selector:
    - Sort by model_win_pct descending (fallback to implied prob).
    - Enforce constraints: same game cap, same player cap, distinct market if requested.
    """
    # Score rows
    def score_row(r: Dict[str, Any]) -> float:
        if r.get("model_win_pct") is not None:
            return float(r["model_win_pct"])
        p = american_to_prob(r.get("odds"))
        return (p or 0.0) * 100.0

    candidates = sorted(rows, key=score_row, reverse=True)

    chosen: List[Dict[str, Any]] = []
    by_game: Dict[str, int] = {}
    by_player: Dict[str, int] = {}
    markets_used: set = set()

    for r in candidates:
        if len(chosen) >= size:
            break
        eid = str(r.get("event_id"))
        player = str(r.get("name"))
        market = str(r.get("market"))

        if max_same_game > 0 and by_game.get(eid, 0) >= max_same_game:
            continue
        if max_per_player > 0 and by_player.get(player, 0) >= max_per_player:
            continue
        if distinct_markets_only and market in markets_used:
            continue

        chosen.append(r)
        by_game[eid] = by_game.get(eid, 0) + 1
        by_player[player] = by_player.get(player, 0) + 1
        markets_used.add(market)

    return chosen


@app.post("/parlay/suggest", response_model=ParlayResp)
def parlay_suggest(req: ParlayReq):
    picks = _pick_greedy_parlay(
        rows=req.rows,
        size=req.parlay_size,
        max_same_game=req.max_same_game,
        max_per_player=req.max_per_player,
        distinct_markets_only=req.distinct_markets_only,
    )
    return {
        "picks": picks,
        "meta": {
            "count": len(picks),
            "parlay_size": req.parlay_size
        }
    }


@app.post("/post/dubclub")
def post_dubclub(req: DubReq):
    """
    Format a DubClub-ready text block. (No external call—just returns the string.)
    """
    lines = [req.header or "Props"]
    for i, p in enumerate(req.picks, 1):
        # Example line: "1) Mike Trout OVER 1.5 TB (-110) — DK"
        who = p.get("name") or "Unknown"
        side = p.get("bet_side") or ""
        line = p.get("line")
        market = p.get("market") or ""
        odds = p.get("odds")
        book = p.get("book") or ""
        parts = [f"{i}) {who}"]
        # Friendly market
        if side:
            parts.append(str(side).upper())
        if line is not None:
            parts.append(str(line))
        if market:
            parts.append(f"[{market}]")
        if odds is not None:
            parts.append(f"({odds})")
        if book:
            parts.append(f"— {book}")
        lines.append(" ".join(parts))
    if req.footer:
        lines.append(req.footer)
    return {"text": "\n".join(lines)}
