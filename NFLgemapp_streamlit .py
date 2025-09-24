
import streamlit as st
import pandas as pd
import math
from datetime import date, datetime
from typing import Dict, List, Set
import re, colorsys

st.set_page_config(page_title="NFL Gematria (All CSV Columns)", page_icon="🏈", layout="wide")

# --------------------
# Loading
# --------------------
@st.cache_data
def load_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
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

def _zero_free_int(n: int) -> int:
    s = ''.join(ch for ch in str(int(n)) if ch != '0')
    return int(s) if s else 0

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

# --------------------
# Name extraction (use every text column)
# --------------------
def extract_names_from_row(row: pd.Series) -> List[Dict[str, str]]:
    names = []
    for col in row.index:
        val = row[col]
        if pd.isna(val):
            continue
        if isinstance(val, (int, float)):
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
    # dedupe ignoring punctuation/case
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
    return [{"source": ", ".join(v["sources"]), "name": v["name"]} for v in by_key.values()]

def build_name_table_from_row(row: pd.Series) -> pd.DataFrame:
    items = extract_names_from_row(row)
    rows = []
    for it in items:
        scores = gematria_scores(it["name"])
        rows.append({"source": it["source"], "name": it["name"], **scores})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["name"]).reset_index(drop=True)
    return df

# --------------------
# Matching logic (zero-insensitive)
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

    date_values = list(date_nums.values())
    date_set_raw: Set[int] = set(int(x) for x in date_values)
    date_set_zf: Set[int] = set(_zero_free_int(int(x)) for x in date_values)

    matches = []

    def record(match_type, detail, value, context):
        matches.append({"type": match_type, "detail": detail, "value": int(value), "context": context})

    # Direct home-away matches by system
    for sys in home_vals.keys():
        inter = home_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter):
            record("home-away direct", sys, val, {"system": sys})

    # Venue vs Home/Away
    for sys in venue_vals.keys():
        inter_home = venue_vals[sys].intersection(home_vals[sys])
        for val in sorted(inter_home):
            record("venue-home direct", sys, val, {"system": sys})
        inter_away = venue_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter_away):
            record("venue-away direct", sys, val, {"system": sys})

    # Aggregate values
    all_team_vals = set().union(*home_vals.values()).union(*away_vals.values()).union(*venue_vals.values())
    all_team_vals_zf = set(_zero_free_int(v) for v in all_team_vals)

    # Team/Venue value equals Date value (zero-insensitive)
    for val in sorted(all_team_vals):
        if (val in date_set_raw) or (_zero_free_int(val) in date_set_zf):
            record("value-date direct", "any system", val, {})

    # Prime index & digit-sum
    for val in sorted(all_team_vals):
        if val in PRIME_INDEX:
            p = PRIME_INDEX[val]
            ds = digit_sum_once(p)
            if (p in date_set_raw) or (_zero_free_int(p) in date_set_zf):
                record("prime-index", f"nth prime where n={val}", p, {"n": val, "prime": p})
            if (ds in date_set_raw) or (_zero_free_int(ds) in date_set_zf):
                record("prime-digit-sum", f"sum(digits(prime(n))) where n={val}", ds, {"n": val, "prime": p})

    # Reverse: date as n (zero-free n)
    for dv in sorted(date_set_raw):
        n = _zero_free_int(dv)
        if n in PRIME_INDEX:
            p = PRIME_INDEX[n]
            ds = digit_sum_once(p)
            if (p in all_team_vals) or (_zero_free_int(p) in all_team_vals_zf):
                record("date->prime-index->value", "prime(date_value) equals team/venue value", p, {"n": n, "prime": p})
            if (ds in all_team_vals) or (_zero_free_int(ds) in all_team_vals_zf):
                record("date->prime-digit-sum->value", "sum(digits(prime(date_value))) equals team/venue value", ds, {"n": n, "prime": p})

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
def _int_to_hex_color(n: int) -> str:
    h = (n * 0.61803398875) % 1.0
    l = 0.38
    s = 0.70
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

def _compute_highlights(matches_df: pd.DataFrame):
    systems = ["ordinal", "reduction", "reverse_ordinal", "reverse_reduction"]
    tables = ["home", "away", "venue", "date"]
    highlights = {t: {"by_system": {s: {} for s in systems}, "any": set()} for t in tables}
    if matches_df is None or matches_df.empty:
        return highlights

    def add_any(tables_sel, val):
        for tb in tables_sel:
            highlights[tb]["any"].add(_zero_free_int(int(val)))

    def add_sys(tables_sel, sys_name, val, typ):
        sys_name = str(sys_name)
        for tb in tables_sel:
            d = highlights[tb]["by_system"][sys_name]
            if typ not in d:
                d[typ] = set()
            d[typ].add(_zero_free_int(int(val)))

    for _, r in matches_df.iterrows():
        typ = r.get("type")
        val = r.get("value")
        ctx = r.get("context", {}) if isinstance(r.get("context", {}), dict) else {}
        sys_name = ctx.get("system", None)
        n = ctx.get("n", None)
        if typ == "home-away direct" and sys_name is not None:
            add_sys(["home", "away"], sys_name, val, typ)
        elif typ == "venue-home direct" and sys_name is not None:
            add_sys(["venue", "home"], sys_name, val, typ)
        elif typ == "venue-away direct" and sys_name is not None:
            add_sys(["venue", "away"], sys_name, val, typ)
        elif typ == "value-date direct":
            add_any(["home", "away", "venue", "date"], val)
        elif typ == "prime-index":
            if n is not None:
                add_any(["home", "away", "venue"], n)
            if val is not None:
                add_any(["date"], val)
        elif typ == "prime-digit-sum":
            if n is not None:
                add_any(["home", "away", "venue"], n)
            if val is not None:
                add_any(["date"], val)
        elif typ == "date->prime-index->value":
            add_any(["home", "away", "venue"], val)
            if n is not None:
                add_any(["date"], n)
        elif typ == "date->prime-digit-sum->value":
            add_any(["home", "away", "venue"], val)
            if n is not None:
                add_any(["date"], n)
    return highlights

def _collect_values(highlights):
    vals = set()
    for tb, obj in highlights.items():
        vals |= set(obj.get("any", set()))
        for sysname, d in obj.get("by_system", {}).items():
            for s in d.values():
                vals |= set(s)
    return set(_zero_free_int(v) for v in vals)

def _build_value_colors(highlights):
    values = sorted(_collect_values(highlights))
    return {v: _int_to_hex_color(v) for v in values}

def _style_for_value(v, sys_name, table_label, highlights, color_for):
    try:
        vv = int(v)
    except Exception:
        return ""
    vvz = _zero_free_int(vv)
    bysys = highlights.get(table_label, {}).get("by_system", {}).get(sys_name, {})
    in_sys = any(vvz in s for s in bysys.values())
    if in_sys or vvz in highlights.get(table_label, {}).get("any", set()):
        color = color_for.get(vvz, "#4b5563")
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
    # Only highlight numbers that actually participate in matches
    date_any = set(highlights.get("date", {}).get("any", set()))
    date_sys_union = set()
    for sysname, d in highlights.get("date", {}).get("by_system", {}).items():
        for s in d.values():
            date_sys_union |= set(s)
    allowed = date_any | date_sys_union
    def _style_date_cell(v):
        try:
            vv = int(v)
        except Exception:
            return ""
        vvz = _zero_free_int(vv)
        if vvz in allowed:
            color = color_for.get(vvz, "#4b5563")
            return f"background-color:{color};color:white;font-weight:600"
        return ""
    if "value" in disp.columns:
        styler = disp.style.apply(lambda col: [_style_date_cell(v) for v in col], subset=["value"])
        return styler
    return disp.style

def build_prime_hits(matches_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=["prime", "n", "digit_sum", "from"])
    for _, r in matches_df.iterrows():
        typ = r.get("type")
        ctx = r.get("context", {}) if isinstance(r.get("context", {}), dict) else {}
        pval = ctx.get("prime", None)
        n = ctx.get("n", None)
        if pval is None:
            continue
        ds = digit_sum_once(int(pval))
        rows.append({"prime": int(pval), "n": int(n) if n is not None else None, "digit_sum": int(ds), "from": str(typ)})
    if not rows:
        return pd.DataFrame(columns=["prime", "n", "digit_sum", "from"])
    dfp = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return dfp

def style_primes_df_with_highlights(df: pd.DataFrame, color_for):
    disp = df.copy()
    def _style_num(v):
        try:
            vv = int(v)
        except Exception:
            return ""
        vvz = _zero_free_int(vv)
        color = color_for.get(vvz, None)
        return f"background-color:{color};color:white;font-weight:600" if color else ""
    def _fmt_num(v):
        try:
            if pd.isna(v):
                return ""
            return f"{int(v)}"
        except Exception:
            return v
    styler = disp.style
    for col in ["prime", "n", "digit_sum"]:
        if col in disp.columns:
            styler = styler.apply(lambda c: [_style_num(v) for v in c], subset=[col])
            styler = styler.format(_fmt_num, subset=[col])
    return styler

def _tables_for_number(vz, highlights):
    present = []
    for tb in ["home", "away", "venue", "date"]:
        anyset = highlights.get(tb, {}).get("any", set())
        bysys = highlights.get(tb, {}).get("by_system", {})
        in_any = vz in anyset
        in_sys = any(vz in s for s in bysys.values()) if isinstance(bysys, dict) else False
        if in_any or in_sys:
            present.append(tb)
    return present

def build_match_summary(matches_df: pd.DataFrame, highlights, color_for: dict) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=["number", "color", "appears_in", "categories", "prime_links", "systems"])

    rows = {}
    def add_row(vz, **kw):
        if vz not in rows:
            rows[vz] = {"number": vz, "color": color_for.get(vz, "#4b5563"),
                        "appears_in": set(), "categories": set(), "prime_links": set(), "systems": {"home": set(), "away": set(), "venue": set()}}
        r = rows[vz]
        for tb in _tables_for_number(vz, highlights):
            r["appears_in"].add(tb)
        if "cat" in kw and kw["cat"]:
            r["categories"].add(kw["cat"])
        if "prime" in kw and kw["prime"] is not None and "n" in kw:
            p = int(kw["prime"]); n = int(kw["n"]) if kw["n"] is not None else None
            ds = digit_sum_once(p)
            r["prime_links"].add(f"n={n} → prime={p} (ds={ds})")
        if "system" in kw and kw["system"] and "who" in kw:
            who = kw["who"]; sysn = kw["system"]
            if who in r["systems"]:
                r["systems"][who].add(sysn)

    for _, row in matches_df.iterrows():
        typ = row.get("type")
        val = int(row.get("value"))
        vz = _zero_free_int(val)
        ctx = row.get("context", {}) if isinstance(row.get("context", {}), dict) else {}
        sysn = ctx.get("system", None)
        n = ctx.get("n", None)
        prime = ctx.get("prime", None)
        add_row(vz, cat=typ, system=sysn, who=("home" if typ=="venue-home direct" else "away" if typ=="venue-away direct" else None), n=n, prime=prime)
        if typ in ("prime-index", "prime-digit-sum", "date->prime-index->value", "date->prime-digit-sum->value"):
            if n is not None:
                add_row(_zero_free_int(int(n)), cat=typ, system=sysn, who=None, n=n, prime=prime)
            if prime is not None:
                add_row(_zero_free_int(int(prime)), cat=typ, system=None, who=None, n=n, prime=prime)
            if typ in ("prime-digit-sum", "date->prime-digit-sum->value") and prime is not None:
                add_row(_zero_free_int(digit_sum_once(int(prime))), cat=typ, system=None, who=None, n=n, prime=prime)

    out_rows = []
    for vz, r in rows.items():
        appears = ", ".join(k.capitalize() for k in ["home","away","venue","date"] if k in r["appears_in"])
        cats = ", ".join(sorted(r["categories"]))
        prime_links = " | ".join(sorted(r["prime_links"])) if r["prime_links"] else ""
        systems_str = "; ".join([f'{k.capitalize()}: {", ".join(sorted(v))}' for k, v in r["systems"].items() if v])
        out_rows.append({"Number": vz, "Color": r["color"], "Appears In": appears, "Categories": cats, "Systems": systems_str, "Prime Links": prime_links})
    df = pd.DataFrame(out_rows).sort_values(by="Number").reset_index(drop=True)
    return df

def style_summary_with_colors(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    disp = df.copy()
    def _style_color(v):
        return f"background-color:{v};" if isinstance(v, str) and v.startswith("#") else ""
    def _fmt_num(v):
        try:
            if pd.isna(v):
                return ""
            return f"{int(v)}"
        except Exception:
            return v
    styler = disp.style
    if "Color" in disp.columns:
        styler = styler.apply(lambda c: [_style_color(v) for v in c], subset=["Color"])
    if "Number" in disp.columns:
        styler = styler.format(_fmt_num, subset=["Number"])
    return styler

def style_matches_df_with_highlights(df: pd.DataFrame, color_for: dict) -> "pd.io.formats.style.Styler":
    disp = df.copy()
    cols_to_color = ["Matched Number", "Team/Venue Value (n)", "Prime # (if applicable)"]
    def _style_num(v):
        try:
            if pd.isna(v):
                return ""
            vv = int(v)
        except Exception:
            return ""
        vvz = _zero_free_int(vv)
        color = color_for.get(vvz, None)
        return f"background-color:{color};color:white;font-weight:600" if color else ""
    def _fmt_num(v):
        try:
            if pd.isna(v):
                return ""
            return f"{int(v)}"
        except Exception:
            return v
    styler = disp.style
    for col in cols_to_color:
        if col in disp.columns:
            styler = styler.apply(lambda c: [_style_num(v) for v in c], subset=[col])
            styler = styler.format(_fmt_num, subset=[col])
    return styler

# --------------------
# UI
# --------------------
st.sidebar.header("Settings")
default_path = "nfl_teams_aliases_3_with_state.csv"
file = st.sidebar.file_uploader("Upload team CSV", type=["csv"], help="Any columns are allowed. At minimum include team_full.")
csv_path_info = st.sidebar.text_input("Or type a CSV path", value=default_path)
highlight_tables = st.sidebar.checkbox("Highlight matches in tables", value=True)
collapse_all = st.sidebar.checkbox("Collapse all sections", value=False)

teams_df = None
try:
    if file is not None:
        teams_df = load_csv(file)
    else:
        teams_df = load_csv(csv_path_info)
except Exception as e:
    st.sidebar.error(f"CSV load error: {e}")

st.title("🏈 NFL Gematria — All CSV Columns")
st.caption("Build gematria tables from every text column in your CSV (city, state, abbr, nickname, abbr_nickname, team_full, aliases, etc.). You can also add **Quarterbacks** for both teams; they will be included in matching & highlights.")

if teams_df is None:
    st.warning("Please upload a valid CSV or provide a correct path in the sidebar.")
    st.stop()

colA, colB = st.columns(2)
with colA:
    home_team = st.selectbox("Home team", options=sorted(teams_df["team_full"].astype(str).unique()))
with colB:
    away_team = st.selectbox("Away team", options=sorted(teams_df["team_full"].astype(str).unique()))

game_date = st.date_input("Game date", value=date.today())
home_qb = st.text_input("Home QB (optional)", value="", placeholder="e.g., Kyler Murray")
away_qb = st.text_input("Away QB (optional)", value="", placeholder="e.g., Geno Smith")
venue = st.text_input("Venue (City / Stadium)", value="", placeholder="e.g., SoFi Stadium, Inglewood, CA")

home_row = teams_df[teams_df["team_full"].astype(str).str.lower() == str(home_team).lower()].iloc[0]
away_row = teams_df[teams_df["team_full"].astype(str).str.lower() == str(away_team).lower()].iloc[0]

home_df = build_name_table_from_row(home_row)
away_df = build_name_table_from_row(away_row)

def _append_qb(df, qb_name: str):
    qb = qb_name.strip()
    if qb:
        scores = gematria_scores(qb)
        row = {"source": "QB", "name": qb, **scores}
        if "name" in df.columns and not df["name"].str.lower().eq(qb.lower()).any():
            return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

home_df = _append_qb(home_df, home_qb)
away_df = _append_qb(away_df, away_qb)

venue_df = None
if venue.strip():
    venue_df = pd.DataFrame([{"source": "venue", "name": venue.strip(), **gematria_scores(venue.strip())}])

date_vals = date_numbers(game_date)

matches_df = find_matches(home_df, away_df, date_vals, venue_df=venue_df)

# Remove direct team-vs-team matches as requested
if isinstance(matches_df, pd.DataFrame) and 'type' in matches_df.columns:
    matches_df = matches_df[matches_df['type'] != 'home-away direct'].reset_index(drop=True)

hl = _compute_highlights(matches_df)
colors_map = _build_value_colors(hl)
primes_df = build_prime_hits(matches_df)

st.subheader("Team Values (from all CSV columns)")
c1, c2 = st.columns(2)
with c1:
    with st.expander("Home team", expanded=not collapse_all):
        if highlight_tables:
            st.table(style_df_with_highlights(home_df, "home", hl, colors_map))
        else:
            st.dataframe(home_df.drop(columns=["source"], errors="ignore"), use_container_width=True)
with c2:
    with st.expander("Away team", expanded=not collapse_all):
        if highlight_tables:
            st.table(style_df_with_highlights(away_df, "away", hl, colors_map))
        else:
            st.dataframe(away_df.drop(columns=["source"], errors="ignore"), use_container_width=True)

if venue_df is not None:
    with st.expander("Venue gematria", expanded=not collapse_all):
        if highlight_tables:
            st.table(style_df_with_highlights(venue_df, "venue", hl, colors_map))
        else:
            st.dataframe(venue_df.drop(columns=["source"], errors="ignore"), use_container_width=True)

st.subheader("Date Numbers")
date_df = pd.DataFrame({"formula": list(date_vals.keys()), "value": list(date_vals.values())})
with st.expander("Date Numbers", expanded=not collapse_all):
    if highlight_tables:
        st.table(style_date_df_with_highlights(date_df, hl, colors_map))
    else:
        st.dataframe(date_df, use_container_width=True)

st.subheader("Prime Hits")
if primes_df is None or primes_df.empty:
    st.caption("No prime-related matches.")
else:
    with st.expander("Prime Hits", expanded=not collapse_all):
        st.table(style_primes_df_with_highlights(primes_df, colors_map))

st.subheader("Grouped by Number (Color)")
summary_df = build_match_summary(matches_df, hl, colors_map)
if summary_df is None or summary_df.empty:
    st.caption("No matches to summarize.")
else:
    with st.expander("Grouped by Number (Color)", expanded=not collapse_all):
        st.table(style_summary_with_colors(summary_df))

st.subheader("Matches")
if matches_df.empty:
    st.info("No matches found with the current inputs.")
else:
    with st.expander("Matches", expanded=not collapse_all):
        st.table(style_matches_df_with_highlights(_prettify_matches(matches_df), colors_map))

st.caption("Notes: All non-empty text cells are included. 'aliases' is split on ';' or ','. Duplicates are removed based on case/punctuation-insensitive comparison.")
