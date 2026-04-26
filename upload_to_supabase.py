import pandas as pd
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ── Upload games in batches ───────────────────────────────────────
df_games = pd.read_csv("nba_games.csv")
df_games = df_games.fillna(0)

# Lowercase all column names to match Supabase table
df_games.columns = df_games.columns.str.lower()

print(f"Uploading {len(df_games)} game records...")
batch_size = 500
total = 0

for i in range(0, len(df_games), batch_size):
    batch = df_games.iloc[i:i+batch_size].to_dict(orient="records")
    try:
        supabase.table("games").insert(batch).execute()
        total += len(batch)
        print(f"  {total}/{len(df_games)} uploaded...")
    except Exception as e:
        print(f"  ✗ Batch failed: {e}")
        break

print(f"\n✓ Done! {total} game records in Supabase")