from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
from supabase import create_client
from nba_api.stats.endpoints import scoreboardv2, leaguegamefinder
from datetime import date

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

def get_supabase():
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL") or "https://qvhupmldwenihkxipeqi.supabase.co"
        key = os.getenv("SUPABASE_KEY") or "sb_publishable_OC1q3aS2LqEpB7UU6y_Y3Q_vpojrHdT"
        _supabase = create_client(url, key)
    return _supabase

# ── Feature config (must match training) ─────────────────────────────────────
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
            "fg_pct": g["fg_pct"],
            "fg3_pct": g["fg3_pct"],
            "ft_pct": g["ft_pct"],
            "reb": g["reb"],
            "oreb": g["oreb"],
            "dreb": g["dreb"],
            "ast": g["ast"],
            "stl": g["stl"],
            "blk": g["blk"],
            "tov": g["tov"],
            "pf": g["pf"],
            "pts": g["pts"],
            "net_rating": net_rating,
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
        finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=today,
            date_to_nullable=today,
            league_id_nullable="00"
        )
        df = finder.get_data_frames()[0]

        games_map = {}
        for _, row in df.iterrows():
            gid = str(row["GAME_ID"])
            matchup = str(row["MATCHUP"])
            team = str(row["TEAM_ABBREVIATION"])
            if gid not in games_map:
                games_map[gid] = {}
            if "vs." in matchup:
                games_map[gid]["home_team"] = team
                games_map[gid]["status"] = "Today"
            elif "@" in matchup:
                games_map[gid]["away_team"] = team

        result = []
        for gid, g in games_map.items():
            if "home_team" in g and "away_team" in g:
                result.append({
                    "game_id": gid,
                    "home_team": g["home_team"],
                    "away_team": g["away_team"],
                    "status": g.get("status", "Today"),
                })

        return {"games": result, "date_used": today, "raw_rows": len(df)}
    except Exception as e:
        return {"error": str(e), "games": [], "date_used": str(date.today())}

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

@app.get("/debug-env")
def debug_env():
    return {
        "SUPABASE_URL": os.getenv("SUPABASE_URL", "NOT FOUND"),
        "SUPABASE_KEY_LENGTH": len(os.getenv("SUPABASE_KEY", ""))
    }