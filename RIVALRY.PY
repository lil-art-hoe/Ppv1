import pandas as pd
import numpy as np
import streamlit as st
from functools import lru_cache

st.set_page_config(page_title="NFL Rivalry Dashboard (open data)", layout="wide")

# ---------------------------
# Data source (open, no scraping)
# ---------------------------
NFL_SCHEDULES_URL = "https://github.com/nflverse/nflfastR-data/raw/master/schedules.csv.gz"

# Normalize old abbreviations into a current "franchise" tag so relocations roll up cleanly.
FRANCHISE_ALIASES = {
    "ARI": {"ARI", "CRD", "PHX"},                 # Cardinals historic codes sometimes appear as CRD/PHX
    "ATL": {"ATL"},
    "BAL": {"BAL"},
    "BUF": {"BUF"},
    "CAR": {"CAR"},
    "CHI": {"CHI"},
    "CIN": {"CIN"},
    "CLE": {"CLE"},
    "DAL": {"DAL"},
    "DEN": {"DEN"},
    "DET": {"DET"},
    "GB":  {"GB", "GNB"},                         # some old refs use GNB
    "HOU": {"HOU"},                               # Texans (NOT old Oilers)
    "IND": {"IND"},
    "JAX": {"JAX", "JAC"},
    "KC":  {"KC", "KAN"},
    "LAC": {"LAC", "SD"},                         # Chargers
    "LAR": {"LAR", "STL", "RAM"},                 # Rams (STL era)
    "LV":  {"LV", "OAK", "LA", "RAI"},            # Raiders (OAK/LA eras)
    "MIA": {"MIA"},
    "MIN": {"MIN"},
    "NE":  {"NE", "NWE"},
    "NO":  {"NO", "NOR"},
    "NYG": {"NYG"},
    "NYJ": {"NYJ"},
    "PHI": {"PHI"},
    "PIT": {"PIT"},
    "SEA": {"SEA"},
    "SF":  {"SF", "SFO"},
    "TB":  {"TB", "TAM"},
    "TEN": {"TEN", "OTI", "HOUOIL"},              # Titans / Oilers rollup
    "WAS": {"WAS", "WFT", "WSH"},                 # Washington (WFT era)
}

# Friendly display names
TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB":  "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC":  "Kansas City Chiefs",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "LV":  "Las Vegas Raiders",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE":  "New England Patriots",
    "NO":  "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks",
    "SF":  "San Francisco 49ers",
    "TB":  "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans (incl. Oilers)",
    "WAS": "Washington Commanders (incl. WFT)",
}

ALL_FRANCHISEES = list(TEAM_NAMES.keys())

@st.cache_data(ttl=6*60*60, show_spinner=True)
def load_schedules() -> pd.DataFrame:
    df = pd.read_csv(NFL_SCHEDULES_URL, compression="gzip", low_memory=False)
    # Keep only columns we care about
    keep = [
        "game_id","season","week","game_type",
        "home_team","away_team","home_score","away_score",
        "start_time","overtime","neutral_site","div_game"
    ]
    df = df[keep].copy()

    # Make sure scores are numeric (some pregame rows may be NaN)
    for col in ["home_score","away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize team codes to franchise codes
    def norm_team(t):
        if pd.isna(t):
            return t
        up = str(t).upper()
        for fran, aliases in FRANCHISE_ALIASES.items():
            if up in aliases:
                return fran
        # Fallback: pass-through
        return up

    df["home_franchise"] = df["home_team"].map(norm_team)
    df["away_franchise"] = df["away_team"].map(norm_team)

    # Outcome columns
    def result_row(r):
        if pd.isna(r["home_score"]) or pd.isna(r["away_score"]):
            return np.nan
        if r["home_score"] > r["away_score"]:
            return "HOME"
        if r["home_score"] < r["away_score"]:
            return "AWAY"
        return "TIE"

    df["winner_side"] = df.apply(result_row, axis=1)
    df["margin"] = df["home_score"] - df["away_score"]
    # unified date
    df["date"] = pd.to_datetime(df["start_time"], errors="coerce").dt.date

    return df

def head_to_head(df: pd.DataFrame, t1: str, t2: str, season_min: int|None, season_max: int|None, game_types: set[str]):
    # Filter seasons/types
    q = df.copy()
    if season_min is not None:
        q = q[q["season"] >= season_min]
    if season_max is not None:
        q = q[q["season"] <= season_max]
    if game_types:
        q = q[q["game_type"].isin(game_types)]

    # Only games between these franchises (either home/away orientation)
    mask = ((q["home_franchise"] == t1) & (q["away_franchise"] == t2)) | \
           ((q["home_franchise"] == t2) & (q["away_franchise"] == t1))
    h2h = q[mask].sort_values(["season","week"]).reset_index(drop=True)

    # Compute records for display (from t1 POV)
    def wlt_counts():
        if h2h.empty:
            return 0,0,0
        # t1 wins if (home==t1 and winner_side==HOME) or (away==t1 and winner_side==AWAY)
        t1_wins = ((h2h["home_franchise"]==t1) & (h2h["winner_side"]=="HOME")).sum() + \
                  ((h2h["away_franchise"]==t1) & (h2h["winner_side"]=="AWAY")).sum()
        t2_wins = ((h2h["home_franchise"]==t2) & (h2h["winner_side"]=="HOME")).sum() + \
                  ((h2h["away_franchise"]==t2) & (h2h["winner_side"]=="AWAY")).sum()
        ties = (h2h["winner_side"]=="TIE").sum()
        return int(t1_wins), int(t2_wins), int(ties)

    t1_w, t2_w, ties = wlt_counts()

    # Points for/against from t1 POV
    if h2h.empty:
        pf = pa = pd.NA
        margin_avg = pd.NA
    else:
        t1_is_home = (h2h["home_franchise"]==t1)
        t1_points = np.where(t1_is_home, h2h["home_score"], h2h["away_score"])
        t2_points = np.where(t1_is_home, h2h["away_score"], h2h["home_score"])
        pf = int(np.nansum(t1_points))
        pa = int(np.nansum(t2_points))
        margin_avg = float(np.nanmean(t1_points - t2_points))

    # Last-10 (from most recent)
    last10 = h2h.dropna(subset=["winner_side"]).tail(10).copy()
    def as_wld(row):
        if row["winner_side"]=="TIE":
            return "T"
        is_t1_home = row["home_franchise"]==t1
        t1_won = (row["winner_side"]=="HOME" and is_t1_home) or (row["winner_side"]=="AWAY" and not is_t1_home)
        return "W" if t1_won else "L"
    last10_marks = "".join(last10.apply(as_wld, axis=1).tolist())

    # Current streak (from most recent)
    def current_streak():
        # Walk backwards until result flips
        recent = h2h.dropna(subset=["winner_side"]).copy()
        if recent.empty:
            return "—"
        recent = recent.iloc[::-1]
        first = recent.iloc[0]
        first_mark = as_wld(first)
        n = 1
        for _, r in recent.iloc[1:].iterrows():
            mark = as_wld(r)
            if mark == first_mark:
                n += 1
            else:
                break
        return f"{first_mark}{n}"

    # Table view
    table = h2h[[
        "date","season","week","game_type",
        "home_franchise","away_franchise","home_score","away_score","margin","neutral_site","overtime","div_game","game_id"
    ]].copy()
    # Pretty labels
    table.rename(columns={
        "game_type":"Type",
        "home_franchise":"Home",
        "away_franchise":"Away",
        "home_score":"H",
        "away_score":"A",
        "neutral_site":"Neutral",
        "overtime":"OT",
        "div_game":"Div",
        "margin":"H-A",
    }, inplace=True)

    return {
        "games": h2h,
        "table": table,
        "t1w": t1_w, "t2w": t2_w, "ties": ties,
        "pf": pf, "pa": pa, "margin_avg": margin_avg,
        "last10": last10_marks if last10_marks else "—",
        "streak": current_streak()
    }

# ---------------------------
# UI
# ---------------------------
st.title("NFL Rivalry (Head-to-Head) — Open Data")
st.caption("Data: nflverse/nflfastR (schedules). This app rolls up relocations/renames into franchise buckets (e.g., OAK/LA/LV Raiders).")

df = load_schedules()

colA, colB = st.columns(2)
with colA:
    team1 = st.selectbox("Team A", ALL_FRANCHISEES, format_func=lambda x: TEAM_NAMES.get(x, x), index=TEAM_NAMES.keys().__iter__().__length_hint__() if False else 15)
with colB:
    team2 = st.selectbox("Team B", [t for t in ALL_FRANCHISEES if t != team1], format_func=lambda x: TEAM_NAMES.get(x, x))

fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    season_min = st.number_input("From season", value=1999, min_value=1920, max_value=int(df["season"].max()))
with fcol2:
    season_max = st.number_input("To season", value=int(df["season"].max()), min_value=season_min, max_value=int(df["season"].max()))
with fcol3:
    gtypes = st.multiselect("Game types", ["REG","POST","PRE"], default=["REG","POST"])

res = head_to_head(df, team1, team2, int(season_min), int(season_max), set(gtypes))

# ---------------------------
# KPI header
# ---------------------------
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total Meetings", f"{len(res['games'])}")
k2.metric(f"{TEAM_NAMES[team1]} Wins", f"{res['t1w']}")
k3.metric(f"{TEAM_NAMES[team2]} Wins", f"{res['t2w']}")
k4.metric("Ties", f"{res['ties']}")
k5.metric(f"{TEAM_NAMES[team1]} Points (PF/PA)", f"{res['pf']} / {res['pa']}")
k6.metric("Avg Margin (A - B)", f"{res['margin_avg']:.2f}" if pd.notna(res["margin_avg"]) else "—")

s1, s2 = st.columns(2)
s1.metric("Last 10 (A’s perspective)", res["last10"])
s2.metric("Current Streak (A’s perspective)", res["streak"])

st.divider()

# ---------------------------
# Charts
# ---------------------------
import altair as alt

left, right = st.columns(2)

with left:
    st.subheader("Results over time")
    if res["games"].empty:
        st.info("No games in the selected range.")
    else:
        # From A's perspective
        g = res["games"].copy()
        g["is_t1_home"] = (g["home_franchise"]==team1)
        g["t1_points"] = np.where(g["is_t1_home"], g["home_score"], g["away_score"])
        g["t2_points"] = np.where(g["is_t1_home"], g["away_score"], g["home_score"])
        g["t1_margin"] = g["t1_points"] - g["t2_points"]
        g["result_mark"] = np.where(g["t1_margin"]>0, "W", np.where(g["t1_margin"]<0, "L", "T"))
        chart = alt.Chart(g.dropna(subset=["date","t1_margin"])).mark_circle(size=80).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("t1_margin:Q", title=f"Margin ({TEAM_NAMES[team1]} − {TEAM_NAMES[team2]})"),
            tooltip=["season","week","game_type","home_franchise","away_franchise","home_score","away_score","t1_margin"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("Margin distribution (A’s perspective)")
    if res["games"].empty:
        st.info("No data.")
    else:
        g = res["games"].copy()
        g["is_t1_home"] = (g["home_franchise"]==team1)
        g["t1_points"] = np.where(g["is_t1_home"], g["home_score"], g["away_score"])
        g["t2_points"] = np.where(g["is_t1_home"], g["away_score"], g["home_score"])
        g["t1_margin"] = g["t1_points"] - g["t2_points"]
        hist = alt.Chart(g.dropna(subset=["t1_margin"])).mark_bar().encode(
            x=alt.X("t1_margin:Q", bin=alt.Bin(maxbins=25), title="Margin"),
            y=alt.Y("count()", title="Games"),
            tooltip=["count()"]
        ).properties(height=300)
        st.altair_chart(hist, use_container_width=True)

st.divider()

# ---------------------------
# Table & Download
# ---------------------------
st.subheader("Game list")
if res["table"].empty:
    st.info("No games to show with current filters.")
else:
    st.dataframe(
        res["table"],
        use_container_width=True,
        hide_index=True
    )
    csv = res["table"].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv,
        file_name=f"h2h_{team1}_vs_{team2}_{season_min}-{season_max}.csv",
        mime="text/csv"
    )

st.caption("Tip: ‘A’s perspective’ = Team A in the selectors above.")
