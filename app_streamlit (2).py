
import streamlit as st
import pandas as pd
import math
from datetime import date, datetime
from typing import Dict, List, Set
import re
import sys
import platform

st.set_page_config(page_title="NFL Gematria Matchup", page_icon="🏈", layout="wide")

# -------------
# Utilities
# -------------
@st.cache_data
def load_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    # Expected columns: city, nickname, abbr, team_full, aliases
    if "team_full" not in df.columns or "aliases" not in df.columns:
        raise ValueError("CSV must include columns: team_full, aliases (semicolon-separated)")
    return df

# Gematria maps
def _gematria_maps():
    o = {chr(i + 65): i + 1 for i in range(26)}  # A=1..Z=26
    ro = {chr(90 - i): i + 1 for i in range(26)}  # Z=1..A=26
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

# First 1000 primes
@st.cache_data
def primes_first_n(n=1000) -> List[int]:
    limit = 90000  # safe upper bound
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
PRIME_INDEX = {i + 1: p for i, p in enumerate(PRIMES)}  # 1-based index -> prime
PRIME_TO_INDEX = {p: i + 1 for i, p in enumerate(PRIMES)}  # prime -> index

def digit_sum_once(n: int) -> int:
    return sum(int(d) for d in str(abs(n)) if d.isdigit())

# Date formulas per user's screenshot
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


def aliases_for_team(team_query: str, teams_df: pd.DataFrame, autofill_to_six: bool = False) -> List[str]:
    row = teams_df[teams_df["team_full"].str.lower() == team_query.lower()]
    if row.empty:
        row = teams_df[teams_df["team_full"].str.lower().str.contains(team_query.lower())]
    if row.empty:
        return [team_query]
    row = row.iloc[0]

    # Split aliases from a single 'aliases' column (accept ';' or ',')
    if "aliases" in row:
        aliases = [a.strip() for a in re.split(r"[;,]", str(row["aliases"])) if a.strip()]
    else:
        aliases = []

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for a in aliases:
        if a not in seen:
            seen.add(a)
            ordered.append(a)

    # Optionally auto-fill up to 6 using known columns
    if autofill_to_six and len(ordered) < 6:
        def _initials(s: str) -> str:
            parts = [p for p in str(s).replace("-", " ").split() if p]
            return "".join(p[0] for p in parts).upper()

        team_full = row["team_full"] if "team_full" in row else None
        city = row["city"] if "city" in row else None
        nickname = row["nickname"] if "nickname" in row else None
        abbr = row["abbr"] if "abbr" in row else None

        # If city/nickname missing, try to infer from team_full
        if (not city or not nickname) and isinstance(team_full, str):
            parts = team_full.split()
            if len(parts) >= 2:
                nickname = nickname or parts[-1]
                city = city or " ".join(parts[:-1])

        candidates = []
        # Preferred canonical 6
        if team_full: candidates.append(str(team_full))
        if abbr: candidates.append(str(abbr))
        if nickname: candidates.append(str(nickname))
        if city: candidates.append(_initials(city))
        if city: candidates.append(str(city))
        if team_full: candidates.append(str(team_full))  # ensure "City Nickname" present

        for c in candidates:
            if c and c not in seen:
                ordered.append(c); seen.add(c)
            if len(ordered) >= 6:
                break

    return ordered[:6] if autofill_to_six else ordered


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

    # Direct matches between home & away by system
    for sys in home_vals.keys():
        inter = home_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter):
            record("home-away direct", sys, val, {"system": sys})

    # Include venue by comparing against both home and away sets
    for sys in venue_vals.keys():
        inter_home = venue_vals[sys].intersection(home_vals[sys])
        for val in sorted(inter_home):
            record("venue-home direct", sys, val, {"system": sys})
        inter_away = venue_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter_away):
            record("venue-away direct", sys, val, {"system": sys})

    # Team (or venue) values vs date numbers
    all_team_vals = set().union(*home_vals.values()).union(*away_vals.values()).union(*venue_vals.values())
    for val in sorted(all_team_vals):
        if val in date_values_set:
            record("value-date direct", "any system", val, {})

    # Prime index matching
    for val in sorted(all_team_vals):
        if val in PRIME_INDEX:
            p = PRIME_INDEX[val]
            if p in date_values_set:
                record("prime-index", f"nth prime where n={val}", p, {"n": val, "prime": p})
            s = digit_sum_once(p)
            if s in date_values_set:
                record("prime-digit-sum", f"sum(digits(prime(n))) where n={val}", s, {"n": val, "prime": p})

    # Reverse: date value acts as n
    for dv in sorted(date_values_set):
        if dv in PRIME_INDEX:
            p = PRIME_INDEX[dv]
            if p in all_team_vals:
                record("date->prime-index->value", "prime(date_value) equals team/venue value", p, {"n": dv, "prime": p})
            s = digit_sum_once(p)
            if s in all_team_vals:
                record("date->prime-digit-sum->value", "sum(digits(prime(date_value))) equals team/venue value", s, {"n": dv, "prime": p})

    return pd.DataFrame(matches)

# -------------
# Sidebar
# -------------
st.sidebar.header("Settings")

default_path = "Untitled spreadsheet - nfl_teams_aliases_core6.csv"
file = st.sidebar.file_uploader("Upload team aliases CSV", type=["csv"], help="Use your cleaned CSV with 6 aliases per team.")
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
st.caption("Enter two NFL teams and a venue/date. The app computes gematria values (ordinal, reduction, reverse ordinal, reverse reduction), date numbers, and prime-based matches.")

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

# Compute
home_aliases = aliases_for_team(home_team, teams_df, autofill_to_six=autofill)
away_aliases = aliases_for_team(away_team, teams_df, autofill_to_six=autofill)
home_df = build_name_table(home_aliases)
away_df = build_name_table(away_aliases)

venue_df = None
if venue.strip():
    # Treat the whole venue string as a single 'alias' item
    venue_df = build_name_table([venue.strip()])

date_vals = date_numbers(game_date)

# Layout
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

# Matches
matches_df = find_matches(home_df, away_df, date_vals, venue_df=venue_df)
st.subheader("Matches")
if matches_df.empty:
    st.info("No matches found with the current inputs.")
else:
    st.dataframe(matches_df, use_container_width=True)

st.caption("Notes: Punctuation and case are ignored in gematria. A 'prime-index' match means the team/venue value equals n and a date value equals the nth prime. 'Prime-digit-sum' means the sum of digits of that prime equals a date value.")

    def _render_matches(matches_df: pd.DataFrame):
        pretty = _prettify_matches(matches_df)
        view = st.radio("Matches view", ["Cards", "Table"], key="matches_view")

        # Color palette per match type
        PALETTE = {
            "Home & Away Same Value": ("#065f46", "#ecfdf5", "#10b981"),
            "Venue ↔ Home Same Value": ("#3f1dcb", "#eef2ff", "#6366f1"),
            "Venue ↔ Away Same Value": ("#3f1dcb", "#eef2ff", "#6366f1"),
            "Team/Venue Value = Date Value": ("#1f2937", "#f3f4f6", "#6b7280"),
            "Prime Index Match": ("#083344", "#ecfeff", "#06b6d4"),
            "Prime Digit-Sum Match": ("#7c2d12", "#fff7ed", "#f97316"),
            "Date as Prime Index": ("#052e16", "#f0fdf4", "#22c55e"),
            "Date as Prime Digit-Sum": ("#4a044e", "#fdf4ff", "#d946ef"),
        }

        if view == "Table":
            st.dataframe(pretty, use_container_width=True)
            return

        for _, row in pretty.iterrows():
            mt = row.get("Match Type", "Match")
            text_c, bg_c, acc_c = PALETTE.get(mt, ("#111827", "#f9fafb", "#9ca3af"))
            badge = f'<span style="background:{acc_c};color:white;padding:2px 8px;border-radius:999px;font-size:12px;">{mt}</span>'
            matched = row.get("Matched Number", "")
            system = row.get("System", "") or "Any"
            nval = row.get("Team/Venue Value (n)", "")
            prime = row.get("Prime # (if applicable)", "")
            expl = row.get("Explanation", "")

            html = f'''
            <div style="border-left:6px solid {acc_c}; background:{bg_c}; padding:10px 12px; margin:8px 0; border-radius:10px;">
                <div style="display:flex;gap:8px;align-items:center;margin-bottom:6px;">{badge}
                    <span style="color:{text_c};font-weight:600;margin-left:4px;">Matched Number:</span>
                    <span style="font-variant-numeric:tabular-nums;color:{text_c};">{matched}</span>
                </div>
                <div style="color:{text_c};font-size:13px;margin-bottom:4px;">{expl}</div>
                <div style="color:{text_c};opacity:.85;font-size:12px;">
                    <b>System:</b> {system}
                    {'&nbsp;·&nbsp;<b>n:</b> ' + str(nval) if nval not in (None, '') else ''}
                    {'&nbsp;·&nbsp;<b>Prime:</b> ' + str(prime) if prime not in (None, '') else ''}
                </div>
            </div>
            '''
            st.markdown(html, unsafe_allow_html=True)
    