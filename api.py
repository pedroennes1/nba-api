from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
from supabase import create_client
from nba_api.stats.endpoints import scoreboardv2
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
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
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

def get_team_rolling_stats(team_abbr: str, before_date: str = None):
    """
    Returns rolling 15-game averages for a team.
    If before_date is provided (YYYY-MM-DD), only uses games before that date.
    """
    sb = get_supabase()

    # Fetch recent games for this team — grab more than WINDOW so we have enough
    query = (
        sb.table("games")
        .select("game_date, fg_pct, fg3_pct, ft_pct, reb, oreb, dreb, ast, stl, blk, tov, pf, pts, matchup, game_id")
        .ilike("team_abbreviation", team_abbr)
        .eq("season_id", "22024")
        .order("game_date", desc=True)
        .limit(WINDOW + 5)
    )

    if before_date:
        query = query.lt("game_date", before_date)

    result = query.execute()

    if not result.data or len(result.data) < 5:
        return None

    games = result.data

    # For net_rating we need opponent pts — fetch those too
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

    # Average across last WINDOW games (already sorted desc, so rows[0] is most recent)
    avgs = {}
    for feat in ROLLING_FEATURES:
        vals = [r[feat] for r in rows if r[feat] is not None]
        avgs[feat] = sum(vals) / len(vals) if vals else 0.0

    return avgs

def build_feature_vector(home_stats: dict, away_stats: dict) -> list:
    """
    Builds the feature vector in the same order used during training:
    [diff_features..., raw_home_features..., is_home]
    """
    diff = [home_stats[f] - away_stats[f] for f in ROLLING_FEATURES]
    raw_home = [home_stats[f] for f in ROLLING_FEATURES]
    return diff + raw_home + [1]  # is_home = 1 for home team

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/today-games")
def today_games():
    try:
        scoreboard = scoreboardv2.ScoreboardV2(
            game_date=date.today().strftime("%m/%d/%Y"),
            league_id="00"
        )
        df = scoreboard.get_data_frames()[0]
        result = []
        for _, game in df.iterrows():
            gamecode = str(game["GAMECODE"])
            teams_part = gamecode.split("/")[1] if "/" in gamecode else ""
            away_team = teams_part[:3] if len(teams_part) >= 6 else ""
            home_team = teams_part[3:] if len(teams_part) >= 6 else ""
            result.append({
                "game_id": str(game["GAME_ID"]),
                "home_team": home_team,
                "away_team": away_team,
                "status": str(game["GAME_STATUS_TEXT"]),
            })
        return {"games": result}
    except Exception as e:
        return {"error": str(e), "games": []}

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

# Keep /predict endpoint for backwards compatibility (uses home team stats only, no matchup context)
@app.post("/predict")
def predict_legacy(request: MatchupRequest):
    return predict_matchup(request)

@app.get("/debug-env")
def debug_env():
    return {
        "SUPABASE_URL": os.getenv("SUPABASE_URL", "NOT FOUND"),
        "SUPABASE_KEY_LENGTH": len(os.getenv("SUPABASE_KEY", ""))
    }
