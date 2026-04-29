from nba_api.stats.endpoints import leaguedashplayerstats
from supabase import create_client
from dotenv import load_dotenv
import pandas as pd
import os
import time

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

seasons = ["2024-25", "2023-24", "2022-23"]
all_players = []

for season in seasons:
    print(f"Fetching {season}...")
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season
    )
    df = stats.get_data_frames()[0]
    df['season'] = season
    all_players.append(df)
    time.sleep(1)

df_players = pd.concat(all_players, ignore_index=True)
df_players.columns = df_players.columns.str.lower()
df_players = df_players.fillna(0)

print(f"Fetched {len(df_players)} player records")
print(f"Columns: {df_players.columns.tolist()}")

df_players.to_csv("nba_players.csv", index=False)
print("Saved nba_players.csv")