from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
from nba_api.library.http import NBAStatsHTTP
import pandas as pd
import time

# Add headers to mimic a real browser
NBAStatsHTTP.HEADERS = {
    'Host': 'stats.nba.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'x-nba-stats-origin': 'stats',
    'x-nba-stats-token': 'true',
    'Referer': 'https://www.nba.com/',
    'Connection': 'keep-alive',
    'Pragma': 'no-cache',
    'Cache-Control': 'no-cache',
}

# ── 1. Get all NBA teams ──────────────────────────────────────────
all_teams = teams.get_teams()
df_teams = pd.DataFrame(all_teams)
print(f"✓ Found {len(df_teams)} teams")

# ── 2. Fetch game logs ────────────────────────────────────────────
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
            time.sleep(15)
    time.sleep(5)

df_games = pd.concat(all_games, ignore_index=True)
print(f"✓ Fetched {len(df_games)} game records")

df_teams.to_csv("nba_teams.csv", index=False)
df_games.to_csv("nba_games.csv", index=False)
print("✓ Saved CSVs")