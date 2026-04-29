from nba_api.stats.static import teams
import requests
import pandas as pd
import time
import os
import urllib.parse

SCRAPER_KEY = os.getenv("SCRAPER_API_KEY")

def scraper_get(url):
    encoded_url = urllib.parse.quote(url, safe='')
    proxy_url = f"http://api.scraperapi.com?api_key={SCRAPER_KEY}&url={encoded_url}"
    response = requests.get(proxy_url, timeout=120)
    return response.json()

all_teams = teams.get_teams()
df_teams = pd.DataFrame(all_teams)
print(f"Found {len(df_teams)} teams")

seasons = ["2024-25", "2023-24", "2022-23", "2021-22", "2020-21"]
all_games = []

for season in seasons:
    print(f"Fetching {season}...")
    url = f"https://stats.nba.com/stats/leaguegamefinder?LeagueID=00&Season={season}&SeasonType=Regular+Season"
    retries = 3
    for attempt in range(retries):
        try:
            data = scraper_get(url)
            headers = data['resultSets'][0]['headers']
            rows = data['resultSets'][0]['rowSet']
            df = pd.DataFrame(rows, columns=headers)
            all_games.append(df)
            print(f"Done {season} - {len(df)} games")
            break
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(15)
    time.sleep(3)

df_games = pd.concat(all_games, ignore_index=True)
df_games.columns = df_games.columns.str.lower()
print(f"Fetched {len(df_games)} records")

df_teams.to_csv("nba_teams.csv", index=False)
df_games.to_csv("nba_games.csv", index=False)
print("Saved CSVs")
