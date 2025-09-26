import io, os
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import requests

st.set_page_config(page_title="NFL Rivalry (open data) — Streamlit Cloud", layout="wide")

# ---------------------------
# Data sources (open, no scraping)
# ---------------------------
LOCAL_PATHS = ["data/schedules.csv.gz", "data/schedules.csv"]
PRIMARY_URLS = [
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules.csv.gz",
    "https://cdn.jsdelivr.net/gh/nflverse/nflfastR-data@master/schedules.csv.gz",
    "https://raw.githubusercontent.com/nflverse/nflfastR-data/master/schedules.csv",
    "https://cdn.jsdelivr.net/gh/nflverse/nflfastR-data@master/schedules.csv",
]

FRANCHISE_ALIASES = {
    "ARI": {"ARI", "CRD", "PHX"},
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
    "GB":  {"GB", "GNB"},
    "HOU": {"HOU"},
    "IND": {"IND"},
    "JAX": {"JAX", "JAC"},
    "KC":  {"KC", "KAN"},
    "LAC": {"LAC", "SD"},
    "LAR": {"LAR", "STL", "RAM"},
    "LV":  {"LV", "OAK", "LA", "RAI"},
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
    "TEN": {"TEN", "OTI", "HOUOIL"},
    "WAS": {"WAS", "WFT", "WSH"},
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

def _load_local() -> pd.DataFrame | None:
    for p in LOCAL_PATHS:
        if os.path.exists(p):
            try:
                return pd.read_csv(p, compression="infer", low_memory=False)
            except Exception:
                pass
    return None

def _fetch_remote(urls) -> pd.DataFrame:
    headers = {"User-Agent": "Mozilla/5.0 (RivalryApp/1.0)"}
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content), compression="infer", low_memory=False)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Download failed from all mirrors. Last error: {last_err}")

@st.cache_data(ttl=6*60*60, show_spinner=True)
def load_schedules() -> pd.DataFrame:
    df = _load_local()
    if df is None:
        df = _fetch_remote(PRIMARY_URLS)

    keep = ["game_id","season","week","game_type","home_team","away_team","home_score","away_score",
            "start_time","overtime","neutral_site","div_game"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].copy()
    for col in ["home_score","away_score"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")

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

    def mark(r):
        if r["winner_side"]=="TIE": return "T"
        is_home = r["home_franchise"]==t1
        won = (r["winner_side"]=="HOME" and is_home) or (r["winner_side"]=="AWAY" and not is_home)
        return "W" if won else "L"

    t1_w = ((h2h["home_franchise"]==t1)&(h2h["winner_side"]=="HOME")).sum() + ((h2h["away_franchise"]==t1)&(h2h["winner_side"]=="AWAY")).sum()
    t2_w = ((h2h["home_franchise"]==t2)&(h2h["winner_side"]=="HOME")).sum() + ((h2h["away_franchise"]==t2)&(h2h["winner_side"]=="AWAY")).sum()
    ties = (h2h["winner_side"]=="TIE").sum()

    if h2h.empty or not {"home_score","away_score"}.issubset(h2h.columns):
        pf = pa = pd.NA; margin_avg = pd.NA
    else:
        t1_home = (h2h["home_franchise"]==t1)
        t1_pts = np.where(t1_home, h2h["home_score"], h2h["away_score"])
        t2_pts = np.where(t1_home, h2h["away_score"], h2h["home_score"])
        pf, pa = int(np.nansum(t1_pts)), int(np.nansum(t2_pts))
        margin_avg = float(np.nanmean(t1_pts - t2_pts))

    last10 = h2h.dropna(subset=["winner_side"]).tail(10).copy()
    last10_marks = "".join(last10.apply(mark, axis=1).tolist()) if len(last10) else "—"

    # current streak
    def streak():
        recent = h2h.dropna(subset=["winner_side"]).copy()
        if recent.empty: return "—"
        recent = recent.iloc[::-1]
        first = recent.iloc[0]; first_mark = mark(first); n=1
        for _, r in recent.iloc[1:].iterrows():
            m = mark(r)
            if m==first_mark: n+=1
            else: break
        return f"{first_mark}{n}"

    show = [c for c in ["date","season","week","game_type","home_franchise","away_franchise","home_score","away_score","margin","neutral_site","overtime","div_game","game_id"] if c in h2h.columns]
    table = h2h[show].copy().rename(columns={"game_type":"Type","home_franchise":"Home","away_franchise":"Away","home_score":"H","away_score":"A","neutral_site":"Neutral","overtime":"OT","div_game":"Div","margin":"H-A"})

    return {"games":h2h,"table":table,"t1w":int(t1_w),"t2w":int(t2_w),"ties":int(ties),
            "pf":pf,"pa":pa,"margin_avg":margin_avg,"last10":last10_marks,"streak":streak()}

st.title("NFL Rivalry (Head-to-Head) — Open Data")
st.caption("If GitHub is rate-limited, add a snapshot to /data via the included GitHub Action.")

df = load_schedules()

ALL = list(TEAM_NAMES.keys())
default_a = ALL.index("KC") if "KC" in ALL else 0
default_b = ALL.index("LV") if "LV" in ALL else (1 if len(ALL)>1 else 0)

c1, c2 = st.columns(2)
with c1:
    team1 = st.selectbox("Team A", ALL, format_func=lambda x: TEAM_NAMES.get(x,x), index=default_a)
with c2:
    choices = [t for t in ALL if t!=team1]
    team2 = st.selectbox("Team B", choices, format_func=lambda x: TEAM_NAMES.get(x,x),
                         index=min(choices.index("LV") if "LV" in choices else 0, len(choices)-1))

f1, f2, f3 = st.columns(3)
with f1:
    min_season_val = int(max(1999, int(df["season"].min()) if "season" in df.columns and len(df) else 1999))
    season_min = st.number_input("From season", value=min_season_val, min_value=1920, max_value=int(df["season"].max()) if "season" in df.columns else 2024)
with f2:
    season_max = st.number_input("To season", value=int(df["season"].max()) if "season" in df.columns else 2024,
                                 min_value=int(season_min),
                                 max_value=int(df["season"].max()) if "season" in df.columns else 2024)
with f3:
    gtypes = st.multiselect("Game types", ["REG","POST","PRE"], default=["REG","POST"])

res = head_to_head(df, team1, team2, int(season_min), int(season_max), set(gtypes))

k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Total Meetings", f"{len(res['games'])}")
k2.metric(f"{TEAM_NAMES[team1]} Wins", f"{res['t1w']}")
k3.metric(f"{TEAM_NAMES[team2]} Wins", f"{res['t2w']}")
k4.metric("Ties", f"{res['ties']}")
k5.metric(f"{TEAM_NAMES[team1]} Points (PF/PA)", f"{res['pf']} / {res['pa']}")
k6.metric("Avg Margin (A - B)", f"{res['margin_avg']:.2f}" if pd.notna(res["margin_avg"]) else "—")

s1,s2 = st.columns(2)
s1.metric("Last 10 (A’s perspective)", res["last10"])
s2.metric("Current Streak (A’s perspective)", res["streak"])

st.divider()

left, right = st.columns(2)
with left:
    st.subheader("Results over time")
    if res["games"].empty:
        st.info("No games in the selected range.")
    else:
        g = res["games"].copy()
        g["is_t1_home"] = (g["home_franchise"]==team1)
        g["t1_points"] = np.where(g["is_t1_home"], g["home_score"], g["away_score"])
        g["t2_points"] = np.where(g["is_t1_home"], g["away_score"], g["home_score"])
        g["t1_margin"] = g["t1_points"] - g["t2_points"]
        chart = alt.Chart(g.dropna(subset=["date","t1_margin"])).mark_circle(size=80).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("t1_margin:Q", title=f"Margin ({TEAM_NAMES[team1]} − {TEAM_NAMES[team2]})"),
            tooltip=["season","week","game_type","Home:home_franchise:N","Away:away_franchise:N","home_score","away_score","t1_margin"]
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

st.subheader("Game list")
if res["table"].empty:
    st.info("No games to show with current filters.")
else:
    st.dataframe(res["table"], use_container_width=True, hide_index=True)
    csv = res["table"].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name=f"h2h_{team1}_vs_{team2}_{season_min}-{season_max}.csv", mime="text/csv")

st.caption("Tip: ‘A’s perspective’ = Team A in the selectors above.")