from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
import requests
from supabase import create_client
from datetime import date, timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("nba_model.pkl", "rb") as f:
    model = pickle.load(f)

_supabase = None

BALLDONTLIE_KEY = os.getenv("BALLDONTLIE_KEY") or "1e22e8aa-e47c-462a-84fb-7e33b30f049b"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY") or "28fd62b9b4b6485c8916e444d9ae2af2"

def get_supabase():
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL") or "https://qvhupmldwenihkxipeqi.supabase.co"
        key = os.getenv("SUPABASE_KEY") or "sb_publishable_OC1q3aS2LqEpB7UU6y_Y3Q_vpojrHdT"
        _supabase = create_client(url, key)
    return _supabase

ROLLING_FEATURES = [
    'fg_pct', 'fg3_pct', 'ft_pct',
    'reb', 'oreb', 'dreb',
    'ast', 'stl', 'blk', 'tov', 'pf',
    'pts', 'net_rating'
]
WINDOW = 15

class MatchupRequest(BaseModel):
    home_team: str
    away_team: str

def get_team_rolling_stats(team_abbr: str):
    sb = get_supabase()
    query = (
        sb.table("games")
        .select("game_date, fg_pct, fg3_pct, ft_pct, reb, oreb, dreb, ast, stl, blk, tov, pf, pts, matchup, game_id")
        .ilike("team_abbreviation", team_abbr)
        .eq("season_id", "22024")
        .order("game_date", desc=True)
        .limit(WINDOW + 5)
    )
    result = query.execute()
    if not result.data or len(result.data) < 5:
        return None
    games = result.data
    game_ids = [g["game_id"] for g in games]
    opp_result = (
        sb.table("games")
        .select("game_id, pts")
        .in_("game_id", game_ids)
        .not_.ilike("team_abbreviation", team_abbr)
        .execute()
    )
    opp_pts_map = {r["game_id"]: r["pts"] for r in (opp_result.data or [])}
    rows = []
    for g in games:
        opp_pts = opp_pts_map.get(g["game_id"])
        net_rating = (g["pts"] - opp_pts) if opp_pts is not None else None
        rows.append({
            "fg_pct": g["fg_pct"], "fg3_pct": g["fg3_pct"], "ft_pct": g["ft_pct"],
            "reb": g["reb"], "oreb": g["oreb"], "dreb": g["dreb"],
            "ast": g["ast"], "stl": g["stl"], "blk": g["blk"],
            "tov": g["tov"], "pf": g["pf"], "pts": g["pts"], "net_rating": net_rating,
        })
    avgs = {}
    for feat in ROLLING_FEATURES:
        vals = [r[feat] for r in rows if r[feat] is not None]
        avgs[feat] = sum(vals) / len(vals) if vals else 0.0
    return avgs

def get_team_scoring_stats(team_abbr: str):
    sb = get_supabase()
    result = (
        sb.table("games")
        .select("pts, game_id")
        .ilike("team_abbreviation", team_abbr)
        .eq("season_id", "22024")
        .order("game_date", desc=True)
        .limit(20)
        .execute()
    )
    if not result.data or len(result.data) < 5:
        return None
    pts_list = [r["pts"] for r in result.data if r["pts"] is not None]
    game_ids = [r["game_id"] for r in result.data]
    opp_result = (
        sb.table("games")
        .select("game_id, pts")
        .in_("game_id", game_ids)
        .not_.ilike("team_abbreviation", team_abbr)
        .execute()
    )
    opp_pts_map = {r["game_id"]: r["pts"] for r in (opp_result.data or [])}
    opp_pts_list = [opp_pts_map[r["game_id"]] for r in result.data if opp_pts_map.get(r["game_id"])]
    return {
        "pts_mean": float(np.mean(pts_list)),
        "pts_std": float(np.std(pts_list)),
        "pts_allowed_mean": float(np.mean(opp_pts_list)) if opp_pts_list else None,
        "pts_allowed_std": float(np.std(opp_pts_list)) if opp_pts_list else None,
    }

def build_feature_vector(home_stats: dict, away_stats: dict) -> list:
    diff = [home_stats[f] - away_stats[f] for f in ROLLING_FEATURES]
    raw_home = [home_stats[f] for f in ROLLING_FEATURES]
    return diff + raw_home + [1]

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/today-games")
def today_games():
    try:
        today = date.today().strftime("%Y-%m-%d")
        response = requests.get(
            f"https://api.balldontlie.io/v1/games?dates[]={today}",
            headers={"Authorization": BALLDONTLIE_KEY},
            timeout=10
        )
        data = response.json()
        result = []
        for game in data.get("data", []):
            home = game["home_team"]["abbreviation"]
            away = game["visitor_team"]["abbreviation"]
            game_time = game.get("time") or ""
            period = game["period"]
            if game_time.lower() == "final":
                status = "Final"
            elif period > 0 and game_time:
                status = f"Q{period} {game_time}"
            elif period > 0:
                status = f"Q{period}"
            else:
                status = "Today"
            result.append({
                "game_id": str(game["id"]),
                "home_team": home,
                "away_team": away,
                "status": status,
            })
        return {"games": result, "date_used": today}
    except Exception as e:
        return {"error": str(e), "games": []}

@app.get("/live-scores")
def live_scores():
    try:
        results = []
        for days_ago in range(0, 6):
            check_date = (date.today() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            response = requests.get(
                f"https://api.balldontlie.io/v1/games?dates[]={check_date}&per_page=25",
                headers={"Authorization": BALLDONTLIE_KEY},
                timeout=10
            )
            data = response.json()
            for game in data.get("data", []):
                home_score = game["home_team_score"]
                away_score = game["visitor_team_score"]
                period = game["period"]
                game_time = (game.get("time") or "").strip()
                if home_score == 0 and away_score == 0 and period == 0:
                    continue
                if game_time.lower() == "final" or (period >= 4 and game_time == ""):
                    status = "Final"
                elif period > 0 and game_time and game_time.lower() != "final":
                    status = f"Q{period} {game_time}"
                elif period > 0:
                    status = f"Q{period}"
                else:
                    status = "Final"
                results.append({
                    "home": game["home_team"]["abbreviation"],
                    "homeScore": home_score,
                    "away": game["visitor_team"]["abbreviation"],
                    "awayScore": away_score,
                    "status": status,
                    "date": game["date"]
                })
            if len(results) >= 8:
                break
        results.sort(key=lambda x: x["date"], reverse=True)
        return {"games": results[:12]}
    except Exception as e:
        return {"error": str(e), "games": []}

@app.get("/news")
def get_news(q: str = Query(default="NBA playoffs")):
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": q,
                "sortBy": "publishedAt",
                "pageSize": 6,
                "language": "en",
                "apiKey": NEWSAPI_KEY,
            },
            timeout=10
        )
        data = response.json()
        articles = []
        for a in data.get("articles", []):
            if not a.get("urlToImage") or not a.get("description"):
                continue
            if "[Removed]" in (a.get("title") or ""):
                continue
            title = a.get("title", "").lower()
            if any(w in title for w in ["playoff", "finals", "series", "game "]):
                category = "Playoffs"
            elif any(w in title for w in ["trade", "sign", "draft", "contract"]):
                category = "Transactions"
            elif any(w in title for w in ["stat", "analytic", "record", "triple"]):
                category = "Analysis"
            elif any(w in title for w in ["injur", "health", "return"]):
                category = "Injury"
            else:
                category = "NBA"
            articles.append({
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "image": a.get("urlToImage", ""),
                "source": a.get("source", {}).get("name", ""),
                "publishedAt": a.get("publishedAt", ""),
                "category": category,
            })
            if len(articles) >= 3:
                break
        return {"articles": articles}
    except Exception as e:
        return {"error": str(e), "articles": []}

@app.get("/recent-games")
def recent_games():
    sb = get_supabase()
    result = (
        sb.table("games")
        .select("team_abbreviation, pts, wl, matchup, game_date, game_id")
        .eq("season_id", "22024")
        .order("game_date", desc=True)
        .limit(30)
        .execute()
    )
    if not result.data:
        return {"games": []}
    games_map = {}
    for row in result.data:
        gid = row["game_id"]
        if gid not in games_map:
            games_map[gid] = {}
        if "vs." in row["matchup"]:
            games_map[gid]["home"] = row
        else:
            games_map[gid]["away"] = row
    paired = []
    for gid, teams in games_map.items():
        if "home" in teams and "away" in teams:
            paired.append({
                "game_id": gid,
                "home": teams["home"]["team_abbreviation"],
                "homeScore": teams["home"]["pts"],
                "away": teams["away"]["team_abbreviation"],
                "awayScore": teams["away"]["pts"],
                "status": "Final",
                "date": teams["home"]["game_date"],
            })
    paired.sort(key=lambda x: x["date"], reverse=True)
    return {"games": paired[:10]}

@app.post("/predict-matchup")
def predict_matchup(request: MatchupRequest):
    home_stats = get_team_rolling_stats(request.home_team)
    away_stats = get_team_rolling_stats(request.away_team)
    if not home_stats:
        return {"error": f"Could not find stats for {request.home_team}"}
    if not away_stats:
        return {"error": f"Could not find stats for {request.away_team}"}
    features = build_feature_vector(home_stats, away_stats)
    prob = model.predict_proba([features])[0]
    home_win = round(float(prob[1]) * 100, 1)
    away_win = round(float(prob[0]) * 100, 1)
    return {
        "home_win_probability": home_win,
        "away_win_probability": away_win,
        "home_stats": home_stats,
        "away_stats": away_stats,
        "model_version": "2.0"
    }

@app.post("/predict-score")
def predict_score(request: MatchupRequest):
    home_stats = get_team_rolling_stats(request.home_team)
    away_stats = get_team_rolling_stats(request.away_team)
    if not home_stats:
        return {"error": f"Could not find stats for {request.home_team}"}
    if not away_stats:
        return {"error": f"Could not find stats for {request.away_team}"}
    home_scoring = get_team_scoring_stats(request.home_team)
    away_scoring = get_team_scoring_stats(request.away_team)
    if not home_scoring or not away_scoring:
        return {"error": "Could not get scoring stats for simulation"}
    SIMULATIONS = 1000
    home_offense = home_scoring["pts_mean"]
    home_defense = home_scoring["pts_allowed_mean"]
    away_offense = away_scoring["pts_mean"]
    away_defense = away_scoring["pts_allowed_mean"]
    home_expected = ((home_offense + away_defense) / 2) + 2.5
    away_expected = (away_offense + home_defense) / 2
    home_std = home_scoring["pts_std"]
    away_std = away_scoring["pts_std"]
    np.random.seed(None)
    home_scores = np.random.normal(home_expected, home_std, SIMULATIONS)
    away_scores = np.random.normal(away_expected, away_std, SIMULATIONS)
    home_scores = np.clip(home_scores, 85, 155)
    away_scores = np.clip(away_scores, 85, 155)
    home_wins = int(np.sum(home_scores > away_scores))
    home_win_pct = round(home_wins / SIMULATIONS * 100, 1)
    away_win_pct = round(100 - home_win_pct, 1)
    predicted_home = round(float(np.mean(home_scores)), 1)
    predicted_away = round(float(np.mean(away_scores)), 1)
    predicted_total = round(predicted_home + predicted_away, 1)
    predicted_spread = round(predicted_home - predicted_away, 1)
    if predicted_spread > 0:
        spread_label = f"{request.home_team} -{abs(predicted_spread)}"
        spread_favor = request.home_team
    elif predicted_spread < 0:
        spread_label = f"{request.away_team} -{abs(predicted_spread)}"
        spread_favor = request.away_team
    else:
        spread_label = "PICK"
        spread_favor = "EVEN"
    return {
        "home_team": request.home_team,
        "away_team": request.away_team,
        "predicted_home_score": predicted_home,
        "predicted_away_score": predicted_away,
        "predicted_total": predicted_total,
        "predicted_spread": predicted_spread,
        "spread_label": spread_label,
        "spread_favor": spread_favor,
        "home_win_probability": home_win_pct,
        "away_win_probability": away_win_pct,
        "simulations_run": SIMULATIONS,
        "model_version": "2.0"
    }

@app.post("/predict")
def predict_legacy(request: MatchupRequest):
    return predict_matchup(request)

@app.get("/highlights")
def get_highlights():
    try:
        YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "key": YOUTUBE_API_KEY,
                "channelId": "UCWJ2lWNubArHWmf3FIHbfcQ",
                "part": "snippet",
                "order": "date",
                "maxResults": 3,
                "q": "highlights",
                "type": "video",
            },
            timeout=10
        )
        data = response.json()
        videos = []
        for item in data.get("items", []):
            vid_id = item["id"]["videoId"]
            title = item["snippet"]["title"]
            published = item["snippet"]["publishedAt"][:10]
            videos.append({
                "id": vid_id,
                "title": title,
                "label": f"{published} · NBA"
            })
        return {"videos": videos}
    except Exception as e:
        return {"error": str(e), "videos": []}
    
@app.get("/debug-env")
def debug_env():
    return {
        "SUPABASE_URL": os.getenv("SUPABASE_URL", "NOT FOUND"),
        "SUPABASE_KEY_LENGTH": len(os.getenv("SUPABASE_KEY", ""))
    }
