
import streamlit as st
import pandas as pd
import math, re, colorsys, unicodedata
from datetime import date, datetime
from difflib import get_close_matches
from typing import Dict, List, Set

st.set_page_config(page_title="Gematria — NFL/Any Team | English & Hebrew", page_icon="🏈", layout="wide")
@st.cache_data
def load_csv(file_or_path) -> pd.DataFrame:
    df = pd.read_csv(file_or_path)
    if "team_full" not in df.columns:
        raise ValueError("CSV must include at least 'team_full'. Other columns are optional.")
    return df

def _zero_free_int(n: int) -> int:
    s = ''.join(ch for ch in str(int(n)) if ch != '0')
    return int(s) if s else 0

def _clean_text_ascii_letters(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    try:
        s = s.encode("ascii", "ignore").decode("ascii")
    except Exception:
        s = str(s)
    s = s.upper()
    return re.sub(r"[^A-Z]", "", s)

def digit_sum_once(n: int) -> int:
    return sum(int(d) for d in str(abs(n)) if d.isdigit())

def _english_maps():
    o = {chr(i + 65): i + 1 for i in range(26)}
    ro = {chr(90 - i): i + 1 for i in range(26)}
    def _reduce(v): return 1 + ((v - 1) % 9) if v > 0 else 0
    red  = {k: _reduce(v) for k, v in o.items()}
    rred = {k: _reduce(v) for k, v in ro.items()}
    return o, red, ro, rred

ORD_EN, RED_EN, RORD_EN, RRED_EN = _english_maps()

def gematria_english(s: str) -> Dict[str, int]:
    t = _clean_text_ascii_letters(s)
    def score(m): return sum(m.get(ch, 0) for ch in t)
    return {
        "ordinal": score(ORD_EN),
        "reduction": score(RED_EN),
        "reverse_ordinal": score(RORD_EN),
        "reverse_reduction": score(RRED_EN),
    }

HEB_LETTERS = list("אבגדהוזחטיכלמנסעפצקרשת")
HEB_FINALS = {"ך":"כ", "ם":"מ", "ן":"נ", "ף":"פ", "ץ":"צ"}
HEB_STD = {"א":1,"ב":2,"ג":3,"ד":4,"ה":5,"ו":6,"ז":7,"ח":8,"ט":9,"י":10,"כ":20,"ל":30,"מ":40,"נ":50,"ס":60,"ע":70,"פ":80,"צ":90,"ק":100,"ר":200,"ש":300,"ת":400,"ך":20,"ם":40,"ן":50,"ף":80,"ץ":90}
HEB_GADOL = HEB_STD.copy(); HEB_GADOL.update({"ך":500,"ם":600,"ן":700,"ף":800,"ץ":900})
HEB_ORDINAL = {}; 
for i, ch in enumerate(HEB_LETTERS, start=1): HEB_ORDINAL[ch] = i
for fin, base in HEB_FINALS.items(): HEB_ORDINAL[fin] = HEB_ORDINAL[base]

def _hebrew_only(s: str) -> str:
    return "".join(ch for ch in str(s) if ("\u0590" <= ch <= "\u05FF"))

def _hebrew_reduce_to_katan(val: int) -> int:
    v = val % 10
    return 9 if v == 0 and val > 0 else v

def gematria_hebrew_text(hs: str, system: str) -> int:
    total = 0
    for ch in hs:
        if system == "Hechrechi": v = HEB_STD.get(ch, 0)
        elif system == "Gadol":   v = HEB_GADOL.get(ch, 0)
        elif system == "Katan":   v = _hebrew_reduce_to_katan(HEB_STD.get(ch, 0))
        elif system == "Siduri":  v = HEB_ORDINAL.get(ch, 0)
        else: v = 0
        total += v
    return total

def gematria_hebrew_all_systems(hs: str) -> Dict[str, int]:
    return {
        "hebrew_hechrechi": gematria_hebrew_text(hs, "Hechrechi"),
        "hebrew_gadol":     gematria_hebrew_text(hs, "Gadol"),
        "hebrew_katan":     gematria_hebrew_text(hs, "Katan"),
        "hebrew_siduri":    gematria_hebrew_text(hs, "Siduri"),
    }

TRANSLIT_RULES = [(r"tch","טש"),(r"tsch","טש"),(r"sch","ש"),(r"sh","ש"),(r"ch","ח"),(r"th","ת"),(r"ph","פ"),(r"ck","ק"),(r"ng","נג")]
TRANSLIT_SINGLE = {"a":"א","e":"ה","i":"י","o":"ו","u":"ו","b":"ב","c":"ק","d":"ד","f":"פ","g":"ג","h":"ה","j":"ג","k":"ק","l":"ל","m":"מ","n":"נ","p":"פ","q":"ק","r":"ר","s":"ס","t":"ט","v":"ב","w":"ו","x":"קס","y":"י","z":"ז"}

def transliterate_to_hebrew(s: str) -> str:
    t = str(s).strip().lower()
    for pat, repl in TRANSLIT_RULES: t = re.sub(pat, repl, t)
    out = []
    for ch in t:
        if ch.isalpha(): out.append(TRANSLIT_SINGLE.get(ch, ""))
    return "".join(out)

@st.cache_data
def primes_first_n(n=1000) -> List[int]:
    limit = 90000
    sieve = [True] * (limit + 1); sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            step = p; start = p * p
            sieve[start:limit + 1:step] = [False] * (((limit - start) // step) + 1)
    ps = [i for i, ok in enumerate(sieve) if ok]
    return ps[:n]

PRIMES = primes_first_n(1000)
PRIME_INDEX = {i + 1: p for i, p in enumerate(PRIMES)}

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

def extract_names_from_row(row: pd.Series) -> List[Dict[str, str]]:
    names = []
    for col in row.index:
        val = row[col]
        if pd.isna(val): continue
        if isinstance(val, (int, float)): continue
        text = str(val).strip()
        if not text: continue
        if str(col).lower() == "aliases":
            parts = [p.strip() for p in re.split(r"[;,]", text) if p.strip()]
            for p in parts: names.append({"source": "aliases", "name": p})
        else:
            names.append({"source": str(col), "name": text})
    by_key = {}
    for it in names:
        key = _clean_text_ascii_letters(it["name"]) or it["name"].strip().lower()
        if key not in by_key: by_key[key] = {"name": it["name"], "sources": [it["source"]]}
        else:
            if it["source"] not in by_key[key]["sources"]:
                by_key[key]["sources"].append(it["source"])
    return [{"source": ", ".join(v["sources"]), "name": v["name"]} for v in by_key.values()]

def build_custom_team_df(team_name: str, aliases_text: str = "") -> List[str]:
    names = []
    if team_name and team_name.strip(): names.append(team_name.strip())
    if aliases_text and aliases_text.strip():
        for part in re.split(r"[;,\n]", str(aliases_text)):
            p = part.strip()
            if p: names.append(p)
    keyset, out = set(), []
    for n in names:
        key = _clean_text_ascii_letters(n) or n.lower()
        if key not in keyset:
            keyset.add(key); out.append(n)
    return out

def english_rows_from_names(names: List[str]) -> pd.DataFrame:
    rows = []
    for nm in names:
        sc = gematria_english(nm)
        rows.append({"source":"custom","name":nm, **sc})
    df = pd.DataFrame(rows)
    if not df.empty: df = df.sort_values(by=["name"]).reset_index(drop=True)
    return df

def hebrew_rows_from_names(names: List[str], hebrew_overrides: List[str], prefer_override: bool) -> pd.DataFrame:
    rows = []
    override_iter = iter(hebrew_overrides) if prefer_override and hebrew_overrides else None
    for nm in names:
        if override_iter:
            try:
                he = next(override_iter).strip()
                he = _hebrew_only(he) or transliterate_to_hebrew(nm)
            except StopIteration:
                he = transliterate_to_hebrew(nm)
        else:
            he = _hebrew_only(nm) or transliterate_to_hebrew(nm)
        scores = gematria_hebrew_all_systems(he)
        row = {"source":"custom","name":f"{nm} (ע)","hebrew_raw":he, **scores}
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty: df = df.sort_values(by=["name"]).reset_index(drop=True)
    return df

def name_table_from_csv_row(row: pd.Series, language_mode: str, hebrew_systems: List[str]) -> pd.DataFrame:
    items = extract_names_from_row(row)
    names = [it["name"] for it in items]
    if language_mode == "English":
        return english_rows_from_names(names)
    heb_names = []
    if "hebrew_name" in row.index and isinstance(row["hebrew_name"], str):
        heb_names.append(row["hebrew_name"])
    if "hebrew_aliases" in row.index and isinstance(row["hebrew_aliases"], str):
        heb_names += [p.strip() for p in re.split(r"[;,\n]", row["hebrew_aliases"]) if p.strip()]
    prefer = bool(heb_names)
    dfh = hebrew_rows_from_names(names, heb_names, prefer)
    keep = ["name","source","hebrew_raw"] + [f"hebrew_{s.lower()}" for s in hebrew_systems]
    return dfh[[c for c in keep if c in dfh.columns]]

def values_from_df(df: pd.DataFrame) -> Dict[str, Set[int]]:
    keys = ["ordinal","reduction","reverse_ordinal","reverse_reduction","hebrew_hechrechi","hebrew_gadol","hebrew_katan","hebrew_siduri"]
    v = {k:set() for k in keys}
    if df is None or df.empty: return v
    for _, r in df.iterrows():
        for k in keys:
            if k in r and pd.notna(r[k]):
                try: v[k].add(int(r[k]))
                except Exception: pass
    return v

def find_matches(home_df: pd.DataFrame, away_df: pd.DataFrame, date_nums, venue_df: pd.DataFrame | None = None) -> pd.DataFrame:
    home_vals = values_from_df(home_df); away_vals = values_from_df(away_df); venue_vals = values_from_df(venue_df)
    date_values = list(date_nums.values())
    date_set_raw = set(int(x) for x in date_values)
    date_set_zf  = set(int(str(x).replace("0","")) if int(x)!=0 else 0 for x in date_values)
    matches = []
    def record(match_type, detail, value, context):
        try: matches.append({"type": match_type, "detail": detail, "value": int(value), "context": context})
        except Exception: pass
    systems = [k for k in home_vals.keys() if home_vals[k] or away_vals[k] or venue_vals[k]]
    for sys in systems:
        inter = home_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter): record("home-away direct", sys, val, {"system": sys})
        inter_home = venue_vals[sys].intersection(home_vals[sys])
        for val in sorted(inter_home): record("venue-home direct", sys, val, {"system": sys})
        inter_away = venue_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter_away): record("venue-away direct", sys, val, {"system": sys})
    all_vals = set().union(*home_vals.values()).union(*away_vals.values()).union(*venue_vals.values())
    all_vals_zf = set(int(str(v).replace("0","")) if int(v)!=0 else 0 for v in all_vals)
    for val in sorted(all_vals):
        if (val in date_set_raw) or (_zero_free_int(val) in date_set_zf):
            record("value-date direct", "any system", val, {})
    from math import prod as _prod
    # prime paths
    PRIMES_LOCAL = {i+1:p for i,p in enumerate(PRIMES)}
    for val in sorted(all_vals):
        if val in PRIMES_LOCAL:
            pval = PRIMES_LOCAL[val]; ds = sum(int(d) for d in str(pval))
            if (pval in date_set_raw) or (_zero_free_int(pval) in date_set_zf):
                record("prime-index", f"nth prime where n={val}", pval, {"n": val, "prime": pval})
            if (ds in date_set_raw) or (_zero_free_int(ds) in date_set_zf):
                record("prime-digit-sum", f"sum(digits(prime(n))) where n={val}", ds, {"n": val, "prime": pval})
    for dv in sorted(date_set_raw):
        n = _zero_free_int(dv)
        if n in PRIMES_LOCAL:
            pval = PRIMES_LOCAL[n]; ds = sum(int(d) for d in str(pval))
            if (pval in all_vals) or (_zero_free_int(pval) in all_vals_zf):
                record("date->prime-index->value", "prime(date_value) equals team/venue value", pval, {"n": n, "prime": pval})
            if (ds in all_vals) or (_zero_free_int(ds) in all_vals_zf):
                record("date->prime-digit-sum->value", "sum(digits(prime(date_value))) equals team/venue value", ds, {"n": n, "prime": pval})
    return pd.DataFrame(matches)

import colorsys
def _int_to_hex_color(n: int) -> str:
    h = (n * 0.61803398875) % 1.0
    l = 0.38
    s = 0.70
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))

def _compute_highlights(matches_df: pd.DataFrame):
    systems = ["ordinal","reduction","reverse_ordinal","reverse_reduction","hebrew_hechrechi","hebrew_gadol","hebrew_katan","hebrew_siduri"]
    tables = ["home","away","venue","date"]
    highlights = {t: {"by_system": {s: {} for s in systems}, "any": set()} for t in tables}
    if matches_df is None or matches_df.empty: return highlights
    def add_any(tables_sel, val):
        for tb in tables_sel: highlights[tb]["any"].add(_zero_free_int(int(val)))
    def add_sys(tables_sel, sys_name, val, typ):
        for tb in tables_sel:
            d = highlights[tb]["by_system"][str(sys_name)]
            if typ not in d: d[typ] = set()
            d[typ].add(_zero_free_int(int(val)))
    for _, r in matches_df.iterrows():
        typ = r.get("type"); val = r.get("value")
        ctx = r.get("context", {}) if isinstance(r.get("context", {}), dict) else {}
        sys_name = ctx.get("system", None); n = ctx.get("n", None)
        if typ == "home-away direct" and sys_name is not None: add_sys(["home","away"], sys_name, val, typ)
        elif typ == "venue-home direct" and sys_name is not None: add_sys(["venue","home"], sys_name, val, typ)
        elif typ == "venue-away direct" and sys_name is not None: add_sys(["venue","away"], sys_name, val, typ)
        elif typ == "value-date direct": add_any(["home","away","venue","date"], val)
        elif typ == "prime-index":
            if n is not None: add_any(["home","away","venue"], n)
            if val is not None: add_any(["date"], val)
        elif typ == "prime-digit-sum":
            if n is not None: add_any(["home","away","venue"], n)
            if val is not None: add_any(["date"], val)
        elif typ == "date->prime-index->value":
            add_any(["home","away","venue"], val)
            if n is not None: add_any(["date"], n)
        elif typ == "date->prime-digit-sum->value":
            add_any(["home","away","venue"], val)
            if n is not None: add_any(["date"], n)
    return highlights

def _collect_values(highlights):
    vals = set()
    for tb, obj in highlights.items():
        vals |= set(obj.get("any", set()))
        for _, d in obj.get("by_system", {}).items():
            for s in d.values(): vals |= set(s)
    return set(_zero_free_int(v) for v in vals)

def _build_value_colors(highlights):
    values = sorted(_collect_values(highlights))
    return {v: _int_to_hex_color(v) for v in values}

def _style_for_value(v, sys_name, table_label, highlights, color_for, focus_set, enable_bg=True):
    try: vv = int(v)
    except Exception: return ""
    vvz = _zero_free_int(vv)
    bysys = highlights.get(table_label, {}).get("by_system", {}).get(sys_name, {})
    in_sys = any(vvz in s for s in bysys.values())
    if in_sys or vvz in highlights.get(table_label, {}).get("any", set()):
        bg = f"background-color:{color_for.get(vvz, '#4b5563')};color:white;font-weight:600" if enable_bg else ""
        border = "border:2px solid #ef4444" if (vvz in focus_set and vvz != 0) else ""
        join = ";" if (bg and border) else ""
        return f"{bg}{join}{border}" if (bg or border) else ""
    if vvz in focus_set and vvz != 0: return "border:2px solid #ef4444"
    return ""

def style_df_with_highlights(df: pd.DataFrame, table_label: str, highlights, color_for, focus_set, enable_bg=True, badges: dict | None = None):
    disp = df.drop(columns=["source","hebrew_raw"], errors="ignore").copy()
    systems = [c for c in ["ordinal","reduction","reverse_ordinal","reverse_reduction","hebrew_hechrechi","hebrew_gadol","hebrew_katan","hebrew_siduri"] if c in disp.columns]
    styler = disp.style
    for sys_name in systems:
        styler = styler.apply(lambda col: [_style_for_value(v, sys_name, table_label, highlights, color_for, focus_set, enable_bg) for v in col], subset=[sys_name])
    if badges:
        date_set = badges.get("date", set())
        prime_set = badges.get("prime", set())
        def _fmt_with_badges(v):
            try: iv = int(v)
            except Exception: return v
            vz = _zero_free_int(iv); marks = ""
            if vz in date_set: marks += " 📅"
            if vz in prime_set: marks += " 🔺"
            return f"{iv}{marks}"
        for sys_name in systems:
            styler = styler.format(_fmt_with_badges, subset=[sys_name])
    return styler


def style_date_df_with_highlights(df: pd.DataFrame, highlights, color_for, focus_set, enable_bg=True):
    disp = df.copy()
    # If no 'value' column present, just return a basic styler to avoid KeyError.
    if "value" not in disp.columns:
        return disp.style

    date_any = set(highlights.get("date", {}).get("any", set()))
    date_sys_union = set()
    for _, d in highlights.get("date", {}).get("by_system", {}).items():
        for s in d.values():
            date_sys_union |= set(s)
    allowed = date_any | date_sys_union

    def _style_date_cell(v):
        try:
            vv = int(v)
        except Exception:
            return ""
        vvz = int(str(vv).replace("0", "")) if vv != 0 else 0
        if vvz in allowed:
            if enable_bg:
                color = color_for.get(vvz, "#4b5563")
                return f"background-color:{color};color:white;font-weight:600"
        if vvz in focus_set and vvz != 0:
            return "border:2px solid #ef4444"
        return ""

    def _fmt_num(v):
        try:
            if pd.isna(v):
                return ""
            return f"{int(v)}"
        except Exception:
            return v

    styler = disp.style.apply(lambda col: [_style_date_cell(v) for v in col], subset=["value"])
    styler = styler.format(_fmt_num, subset=["value"])
    return styler


def _safe_int(x):
    try:
        if x is None: return None
        if isinstance(x, str) and not x.strip(): return None
        import math as _m
        if isinstance(x, float) and (_m.isnan(x) or _m.isinf(x)): return None
        return int(float(x))
    except Exception:
        return None

def build_prime_hits(matches_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=["prime","n","digit_sum","from"])
    for _, r in matches_df.iterrows():
        try:
            ctx = r.get("context", {}) if isinstance(r.get("context", {}), dict) else {}
            pval = ctx.get("prime", None); n = ctx.get("n", None)
            p_int = _safe_int(pval); n_int = _safe_int(n)
            if p_int is None: continue
            ds = digit_sum_once(int(p_int))
            rows.append({"prime": int(p_int), "n": int(n_int) if n_int is not None else None, "digit_sum": int(ds), "from": str(r.get("type"))})
        except Exception: continue
    if not rows: return pd.DataFrame(columns=["prime","n","digit_sum","from"])
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

def style_primes_df_with_highlights(df: pd.DataFrame, color_for, focus_set):
    disp = df.copy()
    def _style_num(v):
        try: vv = int(v)
        except Exception: return ""
        vvz = _zero_free_int(vv)
        bg = f"background-color:{color_for.get(vvz, '#4b5563')};color:white;font-weight:600" if vvz in color_for else ""
        border = "border:2px solid #ef4444" if (vvz in focus_set and vvz != 0) else ""
        join = ";" if (bg and border) else ""
        return f"{bg}{join}{border}"
    def _fmt_num(v):
        try:
            if pd.isna(v): return ""
            return f"{int(v)}"
        except Exception: return v
    styler = disp.style
    for col in ["prime","n","digit_sum"]:
        if col in disp.columns:
            styler = styler.apply(lambda c: [_style_num(v) for v in c], subset=[col])
            styler = styler.format(_fmt_num, subset=[col])
    return styler

def _norm_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).lower().strip()

def _build_team_lookup(df: pd.DataFrame):
    lookup = {}
    for idx, row in df.iterrows():
        candidates = set()
        for col in ["team_full","city","state","abbr","nickname","abbr_nickname"]:
            if col in df.columns and pd.notna(row.get(col, None)) and str(row[col]).strip():
                candidates.add(str(row[col]).strip())
        if "aliases" in df.columns and pd.notna(row.get("aliases", None)):
            for part in re.split(r"[;,]", str(row["aliases"])):
                p = part.strip()
                if p: candidates.add(p)
        if "city" in df.columns and "nickname" in df.columns and pd.notna(row.get("city")) and pd.notna(row.get("nickname")):
            candidates.add(f"{row['city']} {row['nickname']}")
        if "team_full" in df.columns and pd.notna(row.get("team_full", None)):
            candidates.add(str(row["team_full"]).strip())
        for cand in candidates:
            key = _norm_key(cand)
            if key and key not in lookup: lookup[key] = idx
    return lookup

def _resolve_team(text: str, df: pd.DataFrame, lookup):
    if not text or not str(text).strip(): return None
    key = _norm_key(text)
    if key in lookup: return lookup[key]
    from difflib import get_close_matches
    keys = list(lookup.keys())
    if keys:
        match = get_close_matches(key, keys, n=1, cutoff=0.6)
        if match: return lookup[match[0]]
    if "team_full" in df.columns:
        tkeys = [_norm_key(x) for x in df["team_full"].astype(str).tolist()]
        match = get_close_matches(key, tkeys, n=1, cutoff=0.6)
        if match: return df.index[tkeys.index(match[0])]
    return None

st.sidebar.header("Settings")
default_path = "nfl_teams_aliases_3_with_state.csv"
file = st.sidebar.file_uploader("Upload team CSV", type=["csv"], help="Any columns are allowed. At minimum include team_full.")
csv_path_info = st.sidebar.text_input("Or type a CSV path", value=default_path)
highlight_tables = st.sidebar.checkbox("Highlight matches in tables", value=True)
collapse_all = st.sidebar.checkbox("Collapse all sections", value=False)
focus_numbers_raw = st.sidebar.text_input("Focus number(s) (comma-separated, optional)", value="")
show_droot = st.sidebar.checkbox("Show Digital Root in Date Digit-Sums", value=False)
prime_filter_coordinated = st.sidebar.checkbox("Only show prime hits coordinated with date-matching team values", value=False)

def _parse_focus_set(s: str):
    vals = set()
    if not s: return vals
    for part in re.split(r"[\s,]+", s.strip()):
        if part.isdigit():
            n = int(part)
            if n > 0: vals.add(n)
    return set(int(str(v).replace("0","")) if int(v)!=0 else 0 for v in vals)

focus_set = _parse_focus_set(focus_numbers_raw)

teams_df = None
try:
    if file is not None: teams_df = load_csv(file)
    else: teams_df = load_csv(csv_path_info)
except Exception as e:
    st.sidebar.error(f"CSV load error: {e}")

st.title("🏈 Gematria — NFL CSV or ANY Team (English & Hebrew)")
st.caption("Badges: 📅 = Date match, 🔺 = Prime-linked. Toggle Hebrew to compute gematria on Hebrew spellings (transliteration or CSV-provided).")

if teams_df is None:
    st.warning("Please upload a valid CSV or provide a correct path in the sidebar.")
    st.stop()

lang_mode = st.radio("Language / Script", ["English", "Hebrew"], horizontal=True, index=0)
heb_systems = ["Hechrechi","Gadol","Katan","Siduri"]
heb_selected = st.multiselect("Hebrew systems to show", heb_systems, default=["Hechrechi","Katan"]) if lang_mode=="Hebrew" else []

source = st.radio("Team source", ["NFL CSV", "Custom typed"], horizontal=True, index=0)

home_df = pd.DataFrame(); away_df = pd.DataFrame()

if source == "NFL CSV":
    colA, colB = st.columns(2)
    selection_mode = st.radio("Team selection", ["Dropdowns", "Type NFL names"], horizontal=True, index=0)
    if selection_mode == "Dropdowns":
        with colA:
            home_team = st.selectbox("Home team", options=sorted(teams_df["team_full"].astype(str).unique()))
        with colB:
            away_team = st.selectbox("Away team", options=sorted(teams_df["team_full"].astype(str).unique()))
        home_row = teams_df[teams_df["team_full"].astype(str).str.lower() == str(home_team).lower()].iloc[0]
        away_row = teams_df[teams_df["team_full"].astype(str).str.lower() == str(away_team).lower()].iloc[0]
    else:
        lookup = _build_team_lookup(teams_df)
        with colA:
            home_text = st.text_input("Home team (NFL—type any known form)", value="", placeholder="e.g., Arizona Cardinals / ARI / Cardinals / Phoenix")
        with colB:
            away_text = st.text_input("Away team (NFL—type any known form)", value="", placeholder="e.g., Seattle Seahawks / SEA / Seahawks / Seattle")
        h_idx = _resolve_team(home_text, teams_df, lookup)
        a_idx = _resolve_team(away_text, teams_df, lookup)
        home_row = teams_df.loc[h_idx] if h_idx is not None else None
        away_row = teams_df.loc[a_idx] if a_idx is not None else None
        if home_row is None: st.info("Type a valid NFL home team (alias/abbr/city/nickname).")
        else: st.caption(f"Resolved **Home** → **{home_row['team_full']}**")
        if away_row is None: st.info("Type a valid NFL away team (alias/abbr/city/nickname).")
        else: st.caption(f"Resolved **Away** → **{away_row['team_full']}**")

    def _safe_table(row):
        if row is None: return pd.DataFrame(columns=["name"])
        if lang_mode == "English":
            return name_table_from_csv_row(row, "English", [])
        else:
            return name_table_from_csv_row(row, "Hebrew", heb_selected or ["Hechrechi","Katan"])

    home_df = _safe_table(home_row)
    away_df = _safe_table(away_row)

else:
    st.markdown("**Custom teams (any level — NCAA, HS, etc.)**")
    colA, colB = st.columns(2)
    with colA:
        home_text = st.text_input("Home team name", value="", placeholder="e.g., Alabama Crimson Tide")
        home_aliases = st.text_area("Home aliases / variants (optional)", value="", height=80, placeholder="e.g., Alabama; Crimson Tide; Bama; UA")
        home_hebrew = st.text_area("Home Hebrew names (optional, one per line; overrides transliteration)", value="", height=80, placeholder="e.g., אלבמה\nאלבמה טייד")
    with colB:
        away_text = st.text_input("Away team name", value="", placeholder="e.g., Georgia Bulldogs")
        away_aliases = st.text_area("Away aliases / variants (optional)", value="", height=80, placeholder="e.g., Georgia; Bulldogs; UGA; Dawgs")
        away_hebrew = st.text_area("Away Hebrew names (optional, one per line; overrides transliteration)", value="", height=80, placeholder="e.g., ג׳ורג׳יה\nבולדוגס")

    names_home = build_custom_team_df(home_text, home_aliases)
    names_away = build_custom_team_df(away_text, away_aliases)

    if lang_mode == "English":
        home_df = english_rows_from_names(names_home)
        away_df = english_rows_from_names(names_away)
    else:
        home_df = hebrew_rows_from_names(names_home, [ln for ln in home_hebrew.splitlines() if ln.strip()], prefer_override=True if home_hebrew.strip() else False)
        away_df = hebrew_rows_from_names(names_away, [ln for ln in away_hebrew.splitlines() if ln.strip()], prefer_override=True if away_hebrew.strip() else False)
        keep = ["name","source","hebrew_raw"] + [f"hebrew_{s.lower()}" for s in (heb_selected or ["Hechrechi","Katan"])]
        home_df = home_df[[c for c in keep if c in home_df.columns]]
        away_df = away_df[[c for c in keep if c in away_df.columns]]

game_date = st.date_input("Game date", value=date.today())
home_qb = st.text_input("Home QB (optional)", value="", placeholder="e.g., Bryce Young / ברייס יאנג")
away_qb = st.text_input("Away QB (optional)", value="", placeholder="e.g., Jalen Hurts / ג'יילן הרטס")
venue = st.text_input("Venue (City / Stadium)", value="", placeholder="e.g., Lambeau Field, Green Bay, WI / אצטדיון")

def _append_qb(df, qb_name: str):
    qb = qb_name.strip()
    if not qb: return df
    if lang_mode == "English":
        scores = gematria_english(qb)
        row = {"source":"QB","name":qb, **scores}
    else:
        he = _hebrew_only(qb) or transliterate_to_hebrew(qb)
        scores = gematria_hebrew_all_systems(he)
        row = {"source":"QB","name":f"{qb} (ע)","hebrew_raw":he, **scores}
        keep = ["source","name","hebrew_raw"] + [f"hebrew_{s.lower()}" for s in (heb_selected or ["Hechrechi","Katan"])]
        row = {k:v for k,v in row.items() if k in keep}
    if "name" in df.columns and not df["name"].astype(str).str.lower().eq(str(row["name"]).lower()).any():
        return pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

home_df = _append_qb(home_df, home_qb)
away_df = _append_qb(away_df, away_qb)

venue_df = None
if venue.strip():
    if lang_mode == "English":
        row = {"source":"venue","name":venue.strip(), **gematria_english(venue.strip())}
    else:
        he = _hebrew_only(venue) or transliterate_to_hebrew(venue)
        sc = gematria_hebrew_all_systems(he)
        row = {"source":"venue","name":f"{venue.strip()} (ע)","hebrew_raw":he, **sc}
        keep = ["source","name","hebrew_raw"] + [f"hebrew_{s.lower()}" for s in (heb_selected or ["Hechrechi","Katan"])]
        row = {k:v for k,v in row.items() if k in keep}
    venue_df = pd.DataFrame([row])

date_vals = date_numbers(game_date)
matches_df = find_matches(home_df, away_df, date_vals, venue_df=venue_df)
if isinstance(matches_df, pd.DataFrame) and 'type' in matches_df.columns:
    matches_df = matches_df[matches_df['type'] != 'home-away direct'].reset_index(drop=True)

hl = _compute_highlights(matches_df)
colors_map = _build_value_colors(hl)
primes_df = build_prime_hits(matches_df)

def compute_badge_sets(matches_df: pd.DataFrame):
    date_set, prime_set = set(), set()
    if matches_df is None or matches_df.empty: return date_set, prime_set
    for _, r in matches_df.iterrows():
        typ = r.get("type"); v = r.get("value")
        ctx = r.get("context", {}) if isinstance(r.get("context", {}), dict) else {}
        n = ctx.get("n", None)
        if typ == "value-date direct" and v is not None:
            date_set.add(_zero_free_int(int(v)))
        if typ in ("prime-index","prime-digit-sum") and n is not None:
            prime_set.add(_zero_free_int(int(n)))
        if typ in ("date->prime-index->value","date->prime-digit-sum->value") and v is not None:
            prime_set.add(_zero_free_int(int(v)))
    return date_set, prime_set

date_badges, prime_badges = compute_badge_sets(matches_df)
badges = {'date': date_badges, 'prime': prime_badges}

st.subheader("Team Values")
st.caption("Legend: 📅 = matches a Date Number,  🔺 = involved in a Prime-based match")
c1, c2 = st.columns(2)
with c1:
    with st.expander("Home team", expanded=not collapse_all):
        if highlight_tables or focus_set:
            st.table(style_df_with_highlights(home_df, "home", hl, colors_map, focus_set, enable_bg=highlight_tables, badges=badges))
        else:
            st.dataframe(home_df.drop(columns=["source","hebrew_raw"], errors="ignore"), use_container_width=True)
with c2:
    with st.expander("Away team", expanded=not collapse_all):
        if highlight_tables or focus_set:
            st.table(style_df_with_highlights(away_df, "away", hl, colors_map, focus_set, enable_bg=highlight_tables, badges=badges))
        else:
            st.dataframe(away_df.drop(columns=["source","hebrew_raw"], errors="ignore"), use_container_width=True)

if venue_df is not None and not venue_df.empty:
    with st.expander("Venue gematria", expanded=not collapse_all):
        if highlight_tables or focus_set:
            st.table(style_df_with_highlights(venue_df, "venue", hl, colors_map, focus_set, enable_bg=highlight_tables, badges=badges))
        else:
            st.dataframe(venue_df.drop(columns=["source","hebrew_raw"], errors="ignore"), use_container_width=True)

st.subheader("Date Numbers")
date_df = pd.DataFrame({"formula": list(date_vals.keys()), "value": list(date_vals.values())})
with st.expander("Date Numbers", expanded=not collapse_all):
    st.table(style_date_df_with_highlights(date_df, hl, colors_map, focus_set, enable_bg=highlight_tables))

st.subheader("Date Numbers — Digit Sums")
rows_ds = []
for k, v in date_vals.items():
    try:
        iv = int(v)
        ds = digit_sum_once(iv)
        dr = 1 + ((iv - 1) % 9) if iv > 0 else 0
        rows_ds.append({"formula": k, "value": iv, "Digit Sum": ds, "Digital Root": dr})
    except Exception: pass
date_ds_df = pd.DataFrame(rows_ds, columns=['formula','base','digit_sum','digital_root'])
if not show_droot and "Digital Root" in date_ds_df.columns:
    date_ds_df = date_ds_df.drop(columns=["Digital Root"])

with st.expander("Date Numbers — Digit Sums", expanded=not collapse_all):
    if not date_ds_df.empty and set(['formula','digit_sum']).issubset(date_ds_df.columns):
        ds_view = date_ds_df[['formula','digit_sum']].rename(columns={'digit_sum':'value'})
        st.markdown("**Digit Sum**")
        st.table(style_date_df_with_highlights(ds_view, hl, colors_map, focus_set, enable_bg=highlight_tables))
    else:
        st.caption("No date values computed.")
    if show_droot and (not date_ds_df.empty) and set(['formula','digital_root']).issubset(date_ds_df.columns):
        dr_view = date_ds_df[['formula','digital_root']].rename(columns={'digital_root':'value'})
        st.markdown("**Digital Root**")
        st.table(style_date_df_with_highlights(dr_view, hl, colors_map, focus_set, enable_bg=highlight_tables))
    

st.subheader("Prime Hits")
try:
    _pr_df = primes_df
    if prime_filter_coordinated:
        _pr_df = filter_primes_by_date_coord(primes_df, matches_df, date_vals)
    if _pr_df is None or _pr_df.empty:
        st.caption("No prime-related matches." if not prime_filter_coordinated else "No coordinated prime hits (team value also matches a Date number).")
    else:
        with st.expander("Prime Hits", expanded=not collapse_all):
            st.table(style_primes_df_with_highlights(_pr_df, colors_map, focus_set))
except Exception as e:
    st.error(f"Prime Hits rendering error: {e}")

def build_match_summary(matches_df: pd.DataFrame, color_for: dict) -> pd.DataFrame:
    if matches_df is None or matches_df.empty:
        return pd.DataFrame(columns=["number","color","categories"])
    rows = {}
    def add_val(vz, cat):
        if vz not in rows:
            rows[vz] = {"number": vz, "color": color_for.get(vz, "#4b5563"), "categories": set()}
        rows[vz]["categories"].add(cat)
    for _, r in matches_df.iterrows():
        vz = _zero_free_int(int(r.get("value")))
        add_val(vz, r.get("type"))
    out = []
    for vz, r in rows.items():
        out.append({"Number": vz, "Color": r["color"], "Categories": ", ".join(sorted(r["categories"]))})
    return pd.DataFrame(out).sort_values(by="Number").reset_index(drop=True)

st.subheader("Grouped by Number (Color)")
summary_df = build_match_summary(matches_df, colors_map)
if summary_df is None or summary_df.empty:
    st.caption("No matches to summarize.")
else:
    with st.expander("Grouped by Number (Color)", expanded=not collapse_all):
        st.table(summary_df.style.apply(lambda c: [f"background-color:{v};" if isinstance(v,str) and v.startswith('#') else "" for v in c], subset=["Color"]))

def _prettify_matches(matches_df: pd.DataFrame) -> pd.DataFrame:
    if matches_df is None or matches_df.empty: return matches_df
    rows = []
    for _, r in matches_df.iterrows():
        ctx = r.get("context", {})
        sysn = ctx.get("system", None)
        n = ctx.get("n", None); prime = ctx.get("prime", None)
        row = {
            "Match Type": r.get("type"),
            "Matched Number": int(r.get("value")) if pd.notna(r.get("value")) else r.get("value"),
            "System": sysn,
            "Team/Venue Value (n)": n,
            "Prime # (if applicable)": prime,
        }
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

st.subheader("Matches")
if matches_df.empty:
    st.info("No matches found with the current inputs.")
else:
    with st.expander("Matches", expanded=not collapse_all):
        st.table(_prettify_matches(matches_df))
