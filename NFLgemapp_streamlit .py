
import streamlit as st
import pandas as pd
import math
from datetime import date, datetime
from typing import Dict, List, Set
import re

st.set_page_config(page_title="NFL Gematria (All CSV Columns)", page_icon="🏈", layout="wide")

# --------------------
# Loading
# --------------------
@st.cache_data
def load_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    # Require at least team_full; recommend aliases but not required now
    if "team_full" not in df.columns:
        raise ValueError("CSV must include at least 'team_full'. Other columns are optional.")
    return df

# --------------------
# Gematria
# --------------------
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
    return "".join(ch for ch in str(s).upper() if "A" <= ch <= "Z")

def gematria_scores(s: str) -> Dict[str, int]:
    t = _clean_text(s)
    def score(m): return sum(m.get(ch, 0) for ch in t)
    return {
        "ordinal": score(ORD),
        "reduction": score(RED),
        "reverse_ordinal": score(RORD),
        "reverse_reduction": score(RRED),
    }

# --------------------
# Primes + dates
# --------------------
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
        "(m)+(d)+y-digit-sum)": m + d + sum(int(x) for x in str(yyyy)),
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

# --------------------
# Name extraction from ALL columns
# --------------------
def extract_names_from_row(row: pd.Series) -> List[Dict[str, str]]:
    """
    Returns list of dicts: {"source": <column_name or 'aliases'>, "name": <string>}
    Includes every non-empty text cell. If the column is 'aliases', splits on ';' or ','.
    Dedupes by normalized text (gematria-cleaned), but keeps a merged 'source' list.
    """
    names = []
    # First pass: collect raw entries
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            continue
        if isinstance(val, (int, float)):
            # ignore pure numeric cells
            continue
        text = str(val).strip()
        if not text:
            continue

        if str(col).lower() == "aliases":
            parts = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
            for p in parts:
                names.append({"source": "aliases", "name": p})
        else:
            names.append({"source": str(col), "name": text})

    # Second pass: dedupe by cleaned canonical form, merge sources
    by_key = {}
    for item in names:
        key = _clean_text(item["name"])
        if not key:
            continue
        if key not in by_key:
            by_key[key] = {"name": item["name"], "sources": [item["source"]]}
        else:
            if item["source"] not in by_key[key]["sources"]:
                by_key[key]["sources"].append(item["source"])

    out = []
    for key, obj in by_key.items():
        out.append({"source": ", ".join(obj["sources"]), "name": obj["name"]})
    return out

def build_name_table_from_row(row: pd.Series) -> pd.DataFrame:
    items = extract_names_from_row(row)
    rows = []
    for it in items:
        scores = gematria_scores(it["name"])
        rows.append({"source": it["source"], "name": it["name"], **scores})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["source", "name"]).reset_index(drop=True)
    return df

# --------------------
# Matching logic
# --------------------
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

    all_vals = set().union(*home_vals.values()).union(*away_vals.values()).union(*venue_vals.values())
    for val in sorted(all_vals):
        if val in date_values_set:
            record("value-date direct", "any system", val, {})

    for val in sorted(all_vals):
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
            if p in all_vals:
                record("date->prime-index->value", "prime(date_value) equals team/venue value", p, {"n": dv, "prime": p})
            s = digit_sum_once(p)
            if s in all_vals:
                record("date->prime-digit-sum->value", "sum(digits(prime(date_value))) equals team/venue value", s, {"n": dv, "prime": p})

    return pd.DataFrame(matches)

# --------------------
# Pretty Matches
# --------------------
def _prettify_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return matches_df

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
    ]).drop_duplicates().reset_index(drop=True)
    return out



# --------------------
# Highlighting utilities (per-integer colors)
# --------------------
import colorsys

def _int_to_hex_color(n: int) -> str:
    # Deterministic distinct-ish color per integer using the golden angle on the hue wheel.
    # colorsys uses HLS: (h, l, s). We'll keep good contrast for table backgrounds.
    h = (n * 0.61803398875) % 1.0          # wrap hue
    l = 0.38                                # lightness (0..1)
    s = 0.70                                # saturation (0..1)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

def _collect_values(highlights):
    vals = set()
    for tb, obj in highlights.items():
        # any-set
        for v in obj.get("any", set()):
            vals.add(int(v))
        # per-system
        for sysname, vset in obj.get("by_system", {}).items():
            for typ, s in vset.items():
                for v in s:
                    vals.add(int(v))
    return vals

def _build_value_colors(highlights):
    # Map every participating integer to a unique color
    values = sorted(_collect_values(highlights))
    return {v: _int_to_hex_color(v) for v in values}

def _empty_highlights():
    systems = ["ordinal", "reduction", "reverse_ordinal", "reverse_reduction"]
    tables = ["home", "away", "venue", "date"]
    return {t: {"by_system": {s: {} for s in systems}, "any": set()} for t in tables}

def _compute_highlights(matches_df: pd.DataFrame):
    systems = ["ordinal", "reduction", "reverse_ordinal", "reverse_reduction"]
    tables = ["home", "away", "venue", "date"]
    highlights = {t: {"by_system": {s: {} for s in systems}, "any": set()} for t in tables}
    # Each by_system dict holds {match_type: set(values)} though we won't color by type anymore.
    if matches_df is None or matches_df.empty:
        return highlights

    def add_any(tables_sel, val, typ):
        for tb in tables_sel:
            highlights[tb]["any"].add(int(val))

    def add_sys(tables_sel, sys_name, val, typ):
        sys_name = str(sys_name)
        for tb in tables_sel:
            d = highlights[tb]["by_system"][sys_name]
            if typ not in d:
                d[typ] = set()
            d[typ].add(int(val))

    for _, r in matches_df.iterrows():
        typ = r.get("type")
        val = r.get("value")
        ctx = r.get("context", {}) if isinstance(r.get("context", {}), dict) else {}
        sys_name = ctx.get("system", None)
        n = ctx.get("n", None)  # team/venue value
        # Routing per type
        if typ == "home-away direct" and sys_name is not None:
            add_sys(["home", "away"], sys_name, val, typ)
        elif typ == "venue-home direct" and sys_name is not None:
            add_sys(["venue", "home"], sys_name, val, typ)
        elif typ == "venue-away direct" and sys_name is not None:
            add_sys(["venue", "away"], sys_name, val, typ)
        elif typ == "value-date direct":
            add_any(["home", "away", "venue", "date"], val, typ)
        elif typ == "prime-index":
            if n is not None:
                add_any(["home", "away", "venue"], n, typ)   # n in team tables
            if val is not None:
                add_any(["date"], val, typ)                  # nth prime in date table
        elif typ == "prime-digit-sum":
            if n is not None:
                add_any(["home", "away", "venue"], n, typ)   # n in team tables
            if val is not None:
                add_any(["date"], val, typ)                  # digit sum in date table
        elif typ == "date->prime-index->value":
            add_any(["home", "away", "venue"], val, typ)     # prime equals team/venue value
            if n is not None:
                add_any(["date"], n, typ)                    # date n
        elif typ == "date->prime-digit-sum->value":
            add_any(["home", "away", "venue"], val, typ)     # digit sum equals team/venue value
            if n is not None:
                add_any(["date"], n, typ)                    # date n
        # else ignore
    return highlights

def _style_for_value(v, sys_name, table_label, highlights, color_for):
    try:
        vv = int(v)
    except Exception:
        return ""
    # Check system-scoped first
    bysys = highlights.get(table_label, {}).get("by_system", {}).get(sys_name, {})
    in_sys = any(vv in s for s in bysys.values())
    if in_sys or vv in highlights.get(table_label, {}).get("any", set()):
        color = color_for.get(vv, "#4b5563")  # fallback gray
        return f"background-color:{color};color:white;font-weight:600"
    return ""

def style_df_with_highlights(df: pd.DataFrame, table_label: str, highlights, color_for):
    disp = df.drop(columns=["source"], errors="ignore").copy()
    systems = ["ordinal", "reduction", "reverse_ordinal", "reverse_reduction"]
    styler = disp.style
    for sys_name in systems:
        if sys_name in disp.columns:
            styler = styler.apply(lambda col: [ _style_for_value(v, sys_name, table_label, highlights, color_for) for v in col ], subset=[sys_name])
    return styler

def style_date_df_with_highlights(df: pd.DataFrame, highlights, color_for):
    disp = df.copy()
    if "value" in disp.columns:
        styler = disp.style.apply(lambda col: [ _style_for_value(v, "ordinal", "date", highlights, color_for) for v in col ], subset=["value"])
        return styler
    return disp.style
# --------------------
# UI
# --------------------
st.sidebar.header("Settings")
default_path = "nfl_teams_aliases_3_with_state.csv"
file = st.sidebar.file_uploader("Upload team CSV", type=["csv"], help="Any columns are allowed. At minimum include team_full.")
csv_path_info = st.sidebar.text_input("Or type a CSV path", value=default_path)
highlight_tables = st.sidebar.checkbox("Highlight matches in tables", value=True)

teams_df = None
try:
    if file is not None:
        teams_df = load_csv(file)
    else:
        teams_df = load_csv(csv_path_info)
except Exception as e:
    st.sidebar.error(f"CSV load error: {e}")

st.title("🏈 NFL Gematria — All CSV Columns")
st.caption("Build gematria tables from every text column in your CSV (city, state, abbr, nickname, abbr_nickname, team_full, aliases, etc.).")

if teams_df is None:
    st.warning("Please upload a valid CSV or provide a correct path in the sidebar.")
    st.stop()

colA, colB = st.columns(2)
with colA:
    home_team = st.selectbox("Home team", options=sorted(teams_df["team_full"].astype(str).unique()))
with colB:
    away_team = st.selectbox("Away team", options=sorted(teams_df["team_full"].astype(str).unique()))

game_date = st.date_input("Game date", value=date.today())
venue = st.text_input("Venue (City / Stadium)", value="", placeholder="e.g., SoFi Stadium, Inglewood, CA")

home_row = teams_df[teams_df["team_full"].astype(str).str.lower() == str(home_team).lower()].iloc[0]
away_row = teams_df[teams_df["team_full"].astype(str).str.lower() == str(away_team).lower()].iloc[0]

home_df = build_name_table_from_row(home_row)
away_df = build_name_table_from_row(away_row)

venue_df = None
if venue.strip():
    venue_df = pd.DataFrame([{"source": "venue", "name": venue.strip(), **gematria_scores(venue.strip())}])

date_vals = date_numbers(game_date)

matches_df = find_matches(home_df, away_df, date_vals, venue_df=venue_df)

hl = _compute_highlights(matches_df)
colors_map = _build_value_colors(hl)
st.subheader("Team Values (from all CSV columns)")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Home team**")
    if highlight_tables:
        st.table(style_df_with_highlights(home_df, "home", hl, colors_map))
    else:
        st.dataframe(home_df.drop(columns=["source"], errors="ignore"), use_container_width=True)
with c2:
    st.markdown("**Away team**")
    if highlight_tables:
        st.table(style_df_with_highlights(away_df, "away", hl, colors_map))
    else:
        st.dataframe(away_df.drop(columns=["source"], errors="ignore"), use_container_width=True)

if venue_df is not None:
    st.markdown("**Venue gematria**")
    if highlight_tables:
        st.table(style_df_with_highlights(venue_df, "venue", hl, colors_map))
    else:
        st.dataframe(venue_df.drop(columns=["source"], errors="ignore"), use_container_width=True)

st.subheader("Date Numbers")
date_df = pd.DataFrame({"formula": list(date_vals.keys()), "value": list(date_vals.values())})
if highlight_tables:
    st.table(style_date_df_with_highlights(date_df, hl, colors_map))
else:
    st.dataframe(date_df, use_container_width=True)

matches_df = find_matches(home_df, away_df, date_vals, venue_df=venue_df)
st.subheader("Matches")
if matches_df.empty:
    st.info("No matches found with the current inputs.")
else:
    st.dataframe(_prettify_matches(matches_df), use_container_width=True)

st.caption("Notes: All non-empty text cells are included. 'aliases' is split on ';' or ','. Duplicates are removed based on case/punctuation-insensitive comparison.")
