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

# Load model
with open("nba_model.pkl", "rb") as f:
    model = pickle.load(f)

# Supabase client - lazy initialization
_supabase = None

def get_supabase():
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise Exception("Supabase credentials not found")
        _supabase = create_client(url, key)
    return _supabase

class GameStats(BaseModel):
    fg_pct: float
    fg3_pct: float
    ft_pct: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float
    pf: float

class MatchupRequest(BaseModel):
    home_team: str
    away_team: str

def get_team_stats(team_abbr):
    result = get_supabase().table("games")\
        .select("fg_pct, fg3_pct, ft_pct, reb, ast, stl, blk, tov, pf")\
        .ilike("team_abbreviation", team_abbr)\
        .eq("season_id", "22024")\
        .execute()

    if not result.data:
        return None

    stats = result.data
    return {
        "fg_pct": sum(s["fg_pct"] for s in stats) / len(stats),
        "fg3_pct": sum(s["fg3_pct"] for s in stats) / len(stats),
        "ft_pct": sum(s["ft_pct"] for s in stats) / len(stats),
        "reb": sum(s["reb"] for s in stats) / len(stats),
        "ast": sum(s["ast"] for s in stats) / len(stats),
        "stl": sum(s["stl"] for s in stats) / len(stats),
        "blk": sum(s["blk"] for s in stats) / len(stats),
        "tov": sum(s["tov"] for s in stats) / len(stats),
        "pf": sum(s["pf"] for s in stats) / len(stats),
    }

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
    
@app.post("/predict")
def predict(stats: GameStats):
    features = [[
        stats.fg_pct, stats.fg3_pct, stats.ft_pct,
        stats.reb, stats.ast, stats.stl,
        stats.blk, stats.tov, stats.pf
    ]]
    prob = model.predict_proba(features)[0]
    return {
        "win_probability": round(float(prob[1]) * 100, 1),
        "loss_probability": round(float(prob[0]) * 100, 1)
    }

@app.post("/predict-matchup")
def predict_matchup(request: MatchupRequest):
    home_stats = get_team_stats(request.home_team)
    away_stats = get_team_stats(request.away_team)

    if not home_stats or not away_stats:
        return {"error": "Could not find stats for one or both teams"}

    home_features = [[
        home_stats["fg_pct"], home_stats["fg3_pct"], home_stats["ft_pct"],
        home_stats["reb"], home_stats["ast"], home_stats["stl"],
        home_stats["blk"], home_stats["tov"], home_stats["pf"]
    ]]
    away_features = [[
        away_stats["fg_pct"], away_stats["fg3_pct"], away_stats["ft_pct"],
        away_stats["reb"], away_stats["ast"], away_stats["stl"],
        away_stats["blk"], away_stats["tov"], away_stats["pf"]
    ]]

    home_prob = model.predict_proba(home_features)[0]
    away_prob = model.predict_proba(away_features)[0]

    home_win = round(float(home_prob[1]) * 100, 1)
    away_win = round(float(away_prob[1]) * 100, 1)
    total = home_win + away_win

    return {
        "home_win_probability": round((home_win / total) * 100, 1),
        "away_win_probability": round((away_win / total) * 100, 1),
        "home_stats": home_stats,
        "away_stats": away_stats
    }

@app.get("/health")
def health():
    return {"status": "ok"}