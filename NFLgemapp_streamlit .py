import streamlit as st
import pandas as pd
import math
from datetime import date, datetime
from typing import Dict, List, Set

st.set_page_config(page_title="NFL Gematria Matchup", page_icon="🏈", layout="wide")

@st.cache_data
def load_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    if "team_full" not in df.columns or "aliases" not in df.columns:
        raise ValueError("CSV must include columns: team_full, aliases (semicolon-separated)")
    return df

def _gematria_maps():
    o = {chr(i + 65): i + 1 for i in range(26)}
    ro = {chr(90 - i): i + 1 for i in range(26)}
    def _reduce(n: int) -> int:
        return 1 + ((n - 1) % 9) if n > 0 else 0
    red = {k: _reduce(v) for k, v in o.items()}
    rred = {k: _reduce(v) for k, v in ro.items()}
    return o, red, ro, rred

ORD, RED, RORD, RRED = _gematria_maps()

def _clean_text(s: str) -> str:
    return "".join(ch for ch in s.upper() if "A" <= ch <= "Z")

def gematria_scores(s: str) -> Dict[str, int]:
    t = _clean_text(s)
    def score(m): return sum(m.get(ch, 0) for ch in t)
    return {
        "ordinal": score(ORD),
        "reduction": score(RED),
        "reverse_ordinal": score(RORD),
        "reverse_reduction": score(RRED),
    }

@st.cache_data
def primes_first_n(n=1000) -> List[int]:
    limit = 90000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:limit + 1:step] = [False] * (((limit - start) // step) + 1)
    ps = [i for i, ok in enumerate(sieve) if ok]
    return ps[:n]

PRIMES = primes_first_n(1000)
PRIME_INDEX = {i + 1: p for i, p in enumerate(PRIMES)}
PRIME_TO_INDEX = {p: i + 1 for i, p in enumerate(PRIMES)}

def digit_sum_once(n: int) -> int:
    return sum(int(d) for d in str(abs(n)) if d.isdigit())

def date_numbers(dt: date) -> Dict[str, int]:
    m, d, yyyy = dt.month, dt.day, dt.year
    yy = yyyy % 100

    nums = {
        "(m)+(d)+(20)+(yy)": m + d + 20 + yy,
        "(m)+(d)+y-digit-sum": m + d + sum(int(x) for x in str(yyyy)),
        "m+d+y-digit-sum (all digits)": sum(int(x) for x in f"{m}{d}{yyyy}"),
        "(m)+(d)+(yy)": m + d + yy,
        "m+d+yy-digit-sum (all digits)": sum(int(x) for x in f"{m}{d}{yy}"),
        "day_of_year": int(datetime(dt.year, dt.month, dt.day).strftime("%j")),
        "days_left_in_year": (366 if datetime(dt.year, 12, 31).timetuple().tm_yday == 366 else 365) - int(datetime(dt.year, dt.month, dt.day).strftime("%j")),
        "(m)+(d)": m + d,
        "m+d+(20)+(yy)": m + d + 20 + yy,
        "(m)+(d)+y-last-two-digits": m + d + (yy // 10) + (yy % 10),
        "m+d+(yy)": m + d + yy,
    }

    digits_all = [int(c) for c in f"{m}{d}{yyyy}" if c != "0"]
    digits_no_century = [int(c) for c in f"{m}{d}{yy}" if c != "0"]
    nums["product_all_digits(m,d,20,yy)"] = math.prod(digits_all) if digits_all else 0
    nums["product_all_digits(m,d,yy)"] = math.prod(digits_no_century) if digits_no_century else 0
    return nums

def aliases_for_team(team_query: str, teams_df: pd.DataFrame) -> List[str]:
    row = teams_df[teams_df["team_full"].str.lower() == team_query.lower()]
    if row.empty:
        row = teams_df[teams_df["team_full"].str.lower().str.contains(team_query.lower())]
    if row.empty:
        return [team_query]
    row = row.iloc[0]
    aliases = [a.strip() for a in str(row["aliases"]).split(";") if a.strip()]
    seen = set()
    ordered = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered

def build_name_table(name_list: List[str]) -> pd.DataFrame:
    rows = []
    for n in name_list:
        scores = gematria_scores(n)
        rows.append({"name": n, **scores})
    return pd.DataFrame(rows)

def find_matches(home_df: pd.DataFrame, away_df: pd.DataFrame, date_nums: Dict[str, int], venue_df: pd.DataFrame | None = None) -> pd.DataFrame:
    def values_from(df: pd.DataFrame) -> Dict[str, Set[int]]:
        v = {"ordinal": set(), "reduction": set(), "reverse_ordinal": set(), "reverse_reduction": set()}
        for _, r in df.iterrows():
            for k in v.keys():
                v[k].add(int(r[k]))
        return v

    home_vals = values_from(home_df) if home_df is not None else {"ordinal": set(), "reduction": set(), "reverse_ordinal": set(), "reverse_reduction": set()}
    away_vals = values_from(away_df) if away_df is not None else {"ordinal": set(), "reduction": set(), "reverse_ordinal": set(), "reverse_reduction": set()}
    venue_vals = values_from(venue_df) if venue_df is not None else {"ordinal": set(), "reduction": set(), "reverse_ordinal": set(), "reverse_reduction": set()}

    date_values_set: Set[int] = set(date_nums.values())
    matches = []

    def record(match_type, detail, value, context):
        matches.append({"type": match_type, "detail": detail, "value": int(value), "context": context})

    for sys in home_vals.keys():
        inter = home_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter):
            record("home-away direct", sys, val, {"system": sys})

    for sys in venue_vals.keys():
        inter_home = venue_vals[sys].intersection(home_vals[sys])
        for val in sorted(inter_home):
            record("venue-home direct", sys, val, {"system": sys})
        inter_away = venue_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter_away):
            record("venue-away direct", sys, val, {"system": sys})

    all_team_vals = set().union(*home_vals.values()).union(*away_vals.values()).union(*venue_vals.values())
    for val in sorted(all_team_vals):
        if val in date_values_set:
            record("value-date direct", "any system", val, {})

    for val in sorted(all_team_vals):
        if val in PRIME_INDEX:
            p = PRIME_INDEX[val]
            if p in date_values_set:
                record("prime-index", f"nth prime where n={val}", p, {"n": val, "prime": p})
            s = digit_sum_once(p)
            if s in date_values_set:
                record("prime-digit-sum", f"sum(digits(prime(n))) where n={val}", s, {"n": val, "prime": p})

    for dv in sorted(date_values_set):
        if dv in PRIME_INDEX:
            p = PRIME_INDEX[dv]
            if p in all_team_vals:
                record("date->prime-index->value", "prime(date_value) equals team/venue value", p, {"n": dv, "prime": p})
            s = digit_sum_once(p)
            if s in all_team_vals:
                record("date->prime-digit-sum->value", "sum(digits(prime(date_value))) equals team/venue value", s, {"n": dv, "prime": p})

    return pd.DataFrame(matches)

st.sidebar.header("Settings")
default_path = "nfl_teams_aliases_3.csv"  # points to the 3-alias CSV
file = st.sidebar.file_uploader("Upload team aliases CSV", type=["csv"], help="CSV columns: team_full, aliases (semicolon-separated).")
csv_path_info = st.sidebar.text_input("Or type a CSV path", value=default_path)

teams_df = None
try:
    if file is not None:
        teams_df = load_csv(file)
    else:
        teams_df = load_csv(csv_path_info)
except Exception as e:
    st.sidebar.error(f"CSV load error: {e}")

st.title("🏈 NFL Gematria Matchup")
st.caption("Simple version (uses exactly what you provide in 'aliases').")

if teams_df is None:
    st.warning("Please upload a valid CSV or provide a correct path in the sidebar.")
    st.stop()

colA, colB = st.columns(2)
with colA:
    home_team = st.selectbox("Home team", options=sorted(teams_df["team_full"].unique()))
with colB:
    away_team = st.selectbox("Away team", options=sorted(teams_df["team_full"].unique()))

game_date = st.date_input("Game date", value=date.today())
venue = st.text_input("Venue (City / Stadium)", value="", placeholder="e.g., SoFi Stadium, Inglewood, CA")

home_aliases = aliases_for_team(home_team, teams_df)
away_aliases = aliases_for_team(away_team, teams_df)
home_df = build_name_table(home_aliases)
away_df = build_name_table(away_aliases)

venue_df = None
if venue.strip():
    venue_df = build_name_table([venue.strip()])

date_vals = date_numbers(game_date)

st.subheader("Team Alias Tables")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Home team aliases & gematria**")
    st.dataframe(home_df, use_container_width=True)
with c2:
    st.markdown("**Away team aliases & gematria**")
    st.dataframe(away_df, use_container_width=True)

if venue_df is not None:
    st.markdown("**Venue gematria**")
    st.dataframe(venue_df, use_container_width=True)

st.subheader("Date Numbers")
st.dataframe(pd.DataFrame({"formula": list(date_vals.keys()), "value": list(date_vals.values())}), use_container_width=True)

matches_df = find_matches(home_df, away_df, date_vals, venue_df=venue_df)
st.subheader("Matches")
if matches_df.empty:
    st.info("No matches found with the current inputs.")
else:
    st.dataframe(matches_df, use_container_width=True)

st.caption("Notes: Punctuation and case are ignored in gematria. A 'prime-index' match means the team/venue value equals n and a date value equals the nth prime. 'Prime-digit-sum' means the sum of digits of that prime equals a date value.")


def _prettify_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return matches_df

    # Extract context fields if present
    def _ctx_val(ctx, key):
        try:
            return ctx.get(key) if isinstance(ctx, dict) else None
        except Exception:
            return None

    rows = []
    for _, r in matches_df.iterrows():
        ctx = r.get("context", {})
        row = {
            "type": r.get("type"),
            "detail": r.get("detail"),
            "Matched Number": int(r.get("value")) if pd.notna(r.get("value")) else r.get("value"),
            "System": _ctx_val(ctx, "system"),
            "Team/Venue Value (n)": _ctx_val(ctx, "n"),
            "Prime # (if applicable)": _ctx_val(ctx, "prime"),
        }

        t = row["type"]
        if t == "home-away direct":
            mt = "Home & Away Same Value"
            exp = f"Both teams share {row['Matched Number']} in {row['System']}."
        elif t == "venue-home direct":
            mt = "Venue ↔ Home Same Value"
            exp = f"Venue and Home share {row['Matched Number']} in {row['System']}."
        elif t == "venue-away direct":
            mt = "Venue ↔ Away Same Value"
            exp = f"Venue and Away share {row['Matched Number']} in {row['System']}."
        elif t == "value-date direct":
            mt = "Team/Venue Value = Date Value"
            exp = f"A team/venue gematria value equals a date-derived number."
        elif t == "prime-index":
            mt = "Prime Index Match"
            exp = f"Team/Venue value n={row['Team/Venue Value (n)']} → nth prime={row['Prime # (if applicable)']} equals a date value."
        elif t == "prime-digit-sum":
            mt = "Prime Digit-Sum Match"
            exp = f"Team/Venue value n={row['Team/Venue Value (n)']} → nth prime={row['Prime # (if applicable)']} → digit sum equals date value {row['Matched Number']}."
        elif t == "date->prime-index->value":
            mt = "Date as Prime Index"
            exp = f"Date value n={row['Team/Venue Value (n)']} → nth prime={row['Prime # (if applicable)']} equals a team/venue value."
        elif t == "date->prime-digit-sum->value":
            mt = "Date as Prime Digit-Sum"
            exp = f"Date value n={row['Team/Venue Value (n)']} → nth prime={row['Prime # (if applicable)']} → digit sum equals a team/venue value {row['Matched Number']}."
        else:
            mt = t or "Match"
            exp = r.get("detail", "") or "Match"

        row["Match Type"] = mt
        row["Explanation"] = exp
        rows.append(row)

    out = pd.DataFrame(rows, columns=[
        "Match Type", "Matched Number", "Explanation",
        "System", "Team/Venue Value (n)", "Prime # (if applicable)"
    ])
    # Drop duplicates for readability
    out = out.drop_duplicates().reset_index(drop=True)
    return out
