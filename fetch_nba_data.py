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
    retries = 3
    for attempt in range(retries):
        try:
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                timeout=60
            )
            games = gamefinder.get_data_frames()[0]
            all_games.append(games)
            print(f"  ✓ {season} done")
            break
        except Exception as e:
            print(f"  Attempt {attempt+1} failed: {e}")
            time.sleep(10)

    time.sleep(3)

df_games = pd.concat(all_games, ignore_index=True)
print(f"✓ Fetched {len(df_games)} game records")

df_teams.to_csv("nba_teams.csv", index=False)
df_games.to_csv("nba_games.csv", index=False)
print("✓ Saved CSVs")