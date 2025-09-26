import io
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import requests

st.set_page_config(page_title="NFL Rivalry (open data) — v2", layout="wide")

PRIMARY_URLS = [
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules.csv.gz",
    "https://cdn.jsdelivr.net/gh/nflverse/nflfastR-data@master/schedules.csv.gz",
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules.csv",
    "https://cdn.jsdelivr.net/gh/nflverse/nflfastR-data@master/schedules.csv",
]

FRANCHISE_ALIASES = {
    "ARI": {"ARI", "CRD", "PHX"}, "ATL": {"ATL"}, "BAL": {"BAL"}, "BUF": {"BUF"}, "CAR": {"CAR"},
    "CHI": {"CHI"}, "CIN": {"CIN"}, "CLE": {"CLE"}, "DAL": {"DAL"}, "DEN": {"DEN"}, "DET": {"DET"},
    "GB": {"GB", "GNB"}, "HOU": {"HOU"}, "IND": {"IND"}, "JAX": {"JAX", "JAC"}, "KC": {"KC", "KAN"},
    "LAC": {"LAC", "SD"}, "LAR": {"LAR", "STL", "RAM"}, "LV": {"LV", "OAK", "LA", "RAI"},
    "MIA": {"MIA"}, "MIN": {"MIN"}, "NE": {"NE", "NWE"}, "NO": {"NO", "NOR"}, "NYG": {"NYG"},
    "NYJ": {"NYJ"}, "PHI": {"PHI"}, "PIT": {"PIT"}, "SEA": {"SEA"}, "SF": {"SF", "SFO"},
    "TB": {"TB", "TAM"}, "TEN": {"TEN", "OTI", "HOUOIL"}, "WAS": {"WAS", "WFT", "WSH"},
}

TEAM_NAMES = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
    "LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","LV":"Las Vegas Raiders","MIA":"Miami Dolphins",
    "MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
    "NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SEA":"Seattle Seahawks",
    "SF":"San Francisco 49ers","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans (incl. Oilers)",
    "WAS":"Washington Commanders (incl. WFT)",
}
ALL_FRANCHISEES = list(TEAM_NAMES.keys())

def _fetch_csv_from_urls(urls) -> pd.DataFrame:
    last_err = None
    headers = {"User-Agent": "Mozilla/5.0 (RivalryApp/1.0)"}
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            bio = io.BytesIO(r.content)
            df = pd.read_csv(bio, compression="infer", low_memory=False)
            if not df.empty:
                return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to download schedules from known mirrors. Last error: {last_err}")

@st.cache_data(ttl=6*60*60, show_spinner=True)
def load_schedules() -> pd.DataFrame:
    df = _fetch_csv_from_urls(PRIMARY_URLS)
    keep_cols = ["game_id","season","week","game_type","home_team","away_team","home_score","away_score",
                 "start_time","overtime","neutral_site","div_game"]
    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols].copy()
    for col in ["home_score","away_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    def norm_team(t):
        if pd.isna(t): return t
        up = str(t).upper()
        for fran, aliases in FRANCHISE_ALIASES.items():
            if up in aliases: return fran
        return up
    if "home_team" in df.columns: df["home_franchise"] = df["home_team"].map(norm_team)
    if "away_team" in df.columns: df["away_franchise"] = df["away_team"].map(norm_team)
    def result_row(r):
        if ("home_score" not in r) or ("away_score" not in r): return np.nan
        if pd.isna(r["home_score"]) or pd.isna(r["away_score"]): return np.nan
        if r["home_score"]>r["away_score"]: return "HOME"
        if r["home_score"]<r["away_score"]: return "AWAY"
        return "TIE"
    df["winner_side"] = df.apply(result_row, axis=1)
    df["margin"] = df["home_score"] - df["away_score"] if {"home_score","away_score"}.issubset(df.columns) else np.nan
    df["date"] = pd.to_datetime(df["start_time"], errors="coerce").dt.date if "start_time" in df.columns else pd.NaT
    return df

def head_to_head(df, t1, t2, season_min, season_max, game_types):
    q = df.copy()
    if "season" in q.columns:
        if season_min is not None: q = q[q["season"]>=season_min]
        if season_max is not None: q = q[q["season"]<=season_max]
    if "game_type" in q.columns and game_types:
        q = q[q["game_type"].isin(game_types)]
    mask = ((q["home_franchise"]==t1)&(q["away_franchise"]==t2)) | ((q["home_franchise"]==t2)&(q["away_franchise"]==t1))
    h2h = q[mask].sort_values(["season","week"]).reset_index(drop=True)
    def wlt():
        if h2h.empty: return 0,0,0
        t1w = ((h2h["home_franchise"]==t1)&(h2h["winner_side"]=="HOME")).sum() + ((h2h["away_franchise"]==t1)&(h2h["winner_side"]=="AWAY")).sum()
        t2w = ((h2h["home_franchise"]==t2)&(h2h["winner_side"]=="HOME")).sum() + ((h2h["away_franchise"]==t2)&(h2h["winner_side"]=="AWAY")).sum()
        ties = (h2h["winner_side"]=="TIE").sum()
        return int(t1w), int(t2w), int(t
