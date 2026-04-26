from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import pandas as pd
import time

# ── 1. Get all NBA teams ──────────────────────────────────────────
all_teams = teams.get_teams()
df_teams = pd.DataFrame(all_teams)
print(f"✓ Found {len(df_teams)} teams")

# ── 2. Fetch game logs for the last 5 seasons ────────────────────
seasons = ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21"]
all_games = []

for season in seasons:
    print(f"  Fetching {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00"   # 00 = NBA
    )
    games = gamefinder.get_data_frames()[0]
    all_games.append(games)
    time.sleep(1)   # be polite to the API

df_games = pd.concat(all_games, ignore_index=True)
print(f"✓ Fetched {len(df_games)} game records across {len(seasons)} seasons")

# ── 3. Preview & save locally ────────────────────────────────────
print("\nSample columns:", df_games.columns.tolist())
print(df_games.head(3))

df_teams.to_csv("nba_teams.csv", index=False)
df_games.to_csv("nba_games.csv", index=False)
print("\n✓ Saved nba_teams.csv and nba_games.csv")