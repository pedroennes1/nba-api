import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

df = pd.read_csv("nba_players.csv")
df = df.fillna(0)

# Keep only the columns our table has
columns = ['player_id', 'player_name', 'team_abbreviation', 'age', 'gp',
           'pts', 'reb', 'ast', 'stl', 'blk', 'fg_pct', 'fg3_pct',
           'ft_pct', 'tov', 'min', 'plus_minus', 'season']

df = df[columns]

print(f"Uploading {len(df)} player records...")
batch_size = 500
total = 0

for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size].to_dict(orient="records")
    try:
        supabase.table("players").insert(batch).execute()
        total += len(batch)
        print(f"  {total}/{len(df)} uploaded...")
    except Exception as e:
        print(f"  Error: {e}")
        break

print(f"Done! {total} player records in Supabase")