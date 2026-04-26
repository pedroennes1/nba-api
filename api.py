from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Allow Next.js to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("nba_model.pkl", "rb") as f:
    model = pickle.load(f)

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

@app.get("/health")
def health():
    return {"status": "ok"}