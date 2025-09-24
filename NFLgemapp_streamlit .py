
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Set, Tuple

import pandas as pd

# ---------------------
# CSV CONFIG
# ---------------------
DEFAULT_CSV_PATH = r"/mnt/data/Untitled spreadsheet - nfl_teams_aliases_core6.csv"


# ---------------------
# Gematria maps
# ---------------------
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


# ---------------------
# Prime helpers
# ---------------------
def primes_first_n(n=1000) -> List[int]:
    # Good enough sieve for the first 1000 primes
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


# ---------------------
# Date number formulas (per user's screenshot)
# ---------------------
def date_numbers(dt: date) -> Dict[str, int]:
    m, d, yyyy = dt.month, dt.day, dt.year
    yy = yyyy % 100

    # Sums
    nums = {
        "(m)+(d)+(20)+(yy)": m + d + 20 + yy,
        "(m)+(d)+y-digit-sum": m + d + sum(int(x) for x in str(yyyy)),
        "m+d+y-digit-sum (all digits)": sum(int(x) for x in f"{m}{d}{yyyy}"),
        "(m)+(d)+(yy)": m + d + yy,
        "m+d+yy-digit-sum (all digits)": sum(int(x) for x in f"{m}{d}{yy}"),
        "day_of_year": int(dt.strftime("%j")),
        "days_left_in_year": (366 if (date(dt.year, 12, 31).timetuple().tm_yday == 366) else 365) - int(dt.strftime("%j")),
        "(m)+(d)": m + d,
        "m+d+(20)+(yy)": m + d + 20 + yy,
        "(m)+(d)+y-last-two-digits": m + d + (yy // 10) + (yy % 10),
        "m+d+(yy)": m + d + yy,
    }

    # Products (omit zeros)
    digits_all = [int(c) for c in f"{m}{d}{yyyy}" if c != "0"]
    digits_no_century = [int(c) for c in f"{m}{d}{yy}" if c != "0"]
    nums["product_all_digits(m,d,20,yy)"] = math.prod(digits_all) if digits_all else 0
    nums["product_all_digits(m,d,yy)"] = math.prod(digits_no_century) if digits_no_century else 0
    return nums


# ---------------------
# CSV alias loading
# ---------------------
def load_team_aliases(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expecting columns: city, nickname, abbr, team_full, aliases
    # aliases are separated by ';'
    return df


def aliases_for_team(team_query: str, teams_df: pd.DataFrame) -> List[str]:
    # Try exact on team_full first
    row = teams_df[teams_df["team_full"].str.lower() == team_query.lower()]
    if row.empty:
        # Try contains
        row = teams_df[teams_df["team_full"].str.lower().str.contains(team_query.lower())]
    if row.empty:
        return [team_query]

    row = row.iloc[0]
    aliases = [a.strip() for a in str(row["aliases"]).split(";") if a.strip()]
    # Deduplicate while preserving order
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


# ---------------------
# Matching logic
# ---------------------
def find_matches(home_df: pd.DataFrame, away_df: pd.DataFrame, date_nums: Dict[str, int]) -> pd.DataFrame:
    def values_from(df: pd.DataFrame) -> Dict[str, Set[int]]:
        v = {"ordinal": set(), "reduction": set(), "reverse_ordinal": set(), "reverse_reduction": set()}
        for _, r in df.iterrows():
            for k in v.keys():
                v[k].add(int(r[k]))
        return v

    home_vals = values_from(home_df)
    away_vals = values_from(away_df)
    date_values_set: Set[int] = set(date_nums.values())

    matches = []

    def record(match_type, detail, value, context):
        matches.append({"type": match_type, "detail": detail, "value": value, "context": context})

    # 1) Direct matches between home/away in same system
    for sys in home_vals.keys():
        inter = home_vals[sys].intersection(away_vals[sys])
        for val in sorted(inter):
            record("home-away direct", sys, val, {"system": sys})

    # 2) Team value equals a date number
    all_team_vals = set().union(*home_vals.values()).union(*away_vals.values())
    for val in sorted(all_team_vals):
        if val in date_values_set:
            record("team-date direct", "any system", val, {})

    # 3) Prime index match: team value = n, date value = prime(n)
    for val in sorted(all_team_vals):
        if val in PRIME_INDEX:
            p = PRIME_INDEX[val]
            if p in date_values_set:
                record("prime-index", f"nth prime where n=team_value={val}", p, {"n": val, "prime": p})

    # 4) Prime digit-sum match: sum(digits(prime(n))) equals a date value
    for val in sorted(all_team_vals):
        if val in PRIME_INDEX:
            p = PRIME_INDEX[val]
            s = digit_sum_once(p)
            if s in date_values_set:
                record("prime-digit-sum", f"sum(digits(prime(n))) where n=team_value={val}", s, {"n": val, "prime": p})

    # Reverse direction: date value -> prime/date transforms that hit team values
    for dv in sorted(date_values_set):
        if dv in PRIME_INDEX:
            p = PRIME_INDEX[dv]
            if p in all_team_vals:
                record("date->prime-index->team", "prime(date_value) equals team value", p, {"n": dv, "prime": p})
            s = digit_sum_once(p)
            if s in all_team_vals:
                record("date->prime-digit-sum->team", "sum(digits(prime(date_value))) equals team value", s, {"n": dv, "prime": p})

    return pd.DataFrame(matches)


# ---------------------
# Convenience runner
# ---------------------
def run_matchup(home_team: str, away_team: str, game_date: date, csv_path: str = DEFAULT_CSV_PATH):
    teams_df = load_team_aliases(csv_path)
    home_aliases = aliases_for_team(home_team, teams_df)
    away_aliases = aliases_for_team(away_team, teams_df)
    home_table = build_name_table(home_aliases)
    away_table = build_name_table(away_aliases)
    date_nums = date_numbers(game_date)
    matches = find_matches(home_table, away_table, date_nums)
    return home_table, away_table, pd.DataFrame({"formula": list(date_nums.keys()), "value": list(date_nums.values())}), matches
