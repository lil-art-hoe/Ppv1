# app.py
import streamlit as st
from datetime import date
import math
import pandas as pd

# =========================
# Hide Streamlit branding/footer
# =========================
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def digits(n): return [int(ch) for ch in str(n)]
def digits_no_zero(n): return [int(ch) for ch in str(n) if ch != '0']
def normalize_nozeros(n):
    s = ''.join(ch for ch in str(n) if ch != '0')
    return int(s) if s else None
def digit_sum(n): return sum(int(ch) for ch in str(n))
def zero_insensitive_equal(x, y): return x == y or x == normalize_nozeros(y)

def style_duplicates(df):
    """Add a Duplicate flag and sort using only columns that exist."""
    if df is None or df.empty or "Match" not in df.columns:
        return df
    df = df.copy()
    counts = df["Match"].value_counts()
    df["Duplicate"] = df["Match"].map(lambda x: "⚠️ Duplicate" if counts[x] > 1 else "✅ Unique")

    sort_by = ["Duplicate", "Match"]
    for col in ["Rule", "Gematria Field", "Source"]:
        if col in df.columns:
            sort_by.append(col)
    for col in ["Date Value", "Matched Date", "Matched G/J", "Detail"]:
        if col in df.columns:
            sort_by.append(col)

    ascending = [False] + [True] * (len(sort_by) - 1)
    return df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

# =========================
# Gematria
# =========================
def ordinal_value(name): return sum((ord(c.upper())-64) for c in name if c.isalpha())
def reduction_value(name):
    def lr(ch): return ((ord(ch.upper())-65) % 9)+1
    return sum(lr(c) for c in name if c.isalpha())
def reverse_ordinal_value(name): return sum((27-(ord(c.upper())-64)) for c in name if c.isalpha())
def reverse_reduction_value(name):
    def rlr(ch):
        v = 27 - (ord(ch.upper()) - 64)
        return ((v - 1) % 9) + 1
    return sum(rlr(ch) for ch in name if ch.isalpha())

def gematria_values(name):
    return {
        "Ordinal": ordinal_value(name),
        "Reduction": reduction_value(name),
        "Reverse Ordinal": reverse_ordinal_value(name),
        "Reverse Reduction": reverse_reduction_value(name),
    }

# =========================
# Date values (exact 10)
# =========================
def date_values(d):
    m, day, y = d.month, d.day, d.year
    yy = y % 100
    d_digits, yy_digits, y_digits = digits_no_zero(day), digits_no_zero(yy), digits_no_zero(y)

    # 4) days left in year; 5) day of year
    v4 = (date(y, 12, 31) - d).days
    v5 = (d - date(y, 1, 1)).days + 1

    v1  = m + sum(d_digits) + sum(y_digits)
    v2  = m + day + yy
    v3  = m + sum(d_digits) + sum(yy_digits)
    v6  = m + day
    v7  = m + day + sum(yy_digits)
    v8  = m + sum(d_digits) + yy

    v9  = m
    for g in (d_digits + yy_digits): v9 *= g
    v10 = m
    for g in (d_digits + y_digits):  v10 *= g

    return [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10]

# =========================
# Primes
# =========================
def generate_first_n_primes(n):
    primes, candidate = [], 2
    while len(primes) < n:
        is_prime = True
        r = int(math.sqrt(candidate))
        for p in primes:
            if p > r: break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1 if candidate == 2 else 2  # skip evens after 2
    return primes

# =========================
# NEW: Date Digit Sums table
# =========================
def date_digit_sums_table(dvals):
    rows = []
    for i, dv in enumerate(dvals, 1):
        ds = digit_sum(dv)
        z = normalize_nozeros(dv)
        z_ds = digit_sum(z) if z is not None else None
        rows.append({
            "V": f"V{i}",
            "Date Value": dv,
            "Digit Sum": ds,
            "Zero-insensitive": z if (z is not None and z != dv) else "",
            "Digit Sum (zero-insensitive)": z_ds if (z is not None and z != dv) else "",
            "Match": ds  # to reuse duplicate-highlighting
        })
    return pd.DataFrame(rows)

# =========================
# Match functions
# =========================
def pairwise_gd(subject_name, jersey, dvals):
    """Strict equality: Gematria/Jersey ↔ Date (no zero-insensitive)."""
    gvals = gematria_values(subject_name)
    rows = []
    for label, gv in gvals.items():
        for i, dv in enumerate(dvals, 1):
            if gv == dv:
                rows.append({"Rule":"G/J ↔ Date (strict)", "Source":str(label), "Value":gv,
                             "Date Value":f"V{i}={dv}", "Match":gv})
    for i, dv in enumerate(dvals, 1):
        if jersey == dv:
            rows.append({"Rule":"G/J ↔ Date (strict)", "Source":"Jersey", "Value":jersey,
                         "Date Value":f"V{i}={dv}", "Match":jersey})
    return pd.DataFrame(rows)

def pairwise_gd_zero_insensitive(subject_name, dvals):
    """Gematria ↔ Date (zero-insensitive on Date side)."""
    gvals = gematria_values(subject_name)
    rows = []
    for label, gv in gvals.items():
        for i, dv in enumerate(dvals, 1):
            if zero_insensitive_equal(gv, dv):
                note = "Exact" if gv == dv else f"Zero-insensitive (normalize {dv}→{normalize_nozeros(dv)})"
                rows.append({
                    "Rule": "Gematria ↔ Date (zero-insensitive)",
                    "Gematria Field": label,
                    "Gematria Value": gv,
                    "Date Value": f"V{i}={dv}",
                    "Match": gv,
                    "Detail": note
                })
    return pd.DataFrame(rows)

def pairwise_section2_jersey_prime_matches_date(jersey, primes, dvals):
    """
    SECTION 2 — Jersey-only rule:
      Use the jersey number n as a prime INDEX (n → p_n).
      Trigger a match ONLY if p_n equals a Date value (zero-insensitive).
      (No gematria, no digit-sum.)
    """
    rows = []
    if isinstance(jersey, int) and 1 <= jersey <= len(primes):
        p = primes[jersey - 1]
        hits = [f"V{i}={dv}" for i, dv in enumerate(dvals, 1) if zero_insensitive_equal(p, dv)]
        if hits:
            rows.append({
                "Rule": "Jersey index → prime value equals Date",
                "Source": "Jersey",
                "Jersey #": jersey,
                "Prime p_n": p,
                "Matched Date": ", ".join(hits),
                "Match": p
            })
    return pd.DataFrame(rows)

def pairwise_dp(dvals, primes):
    """Context-only: Date value is prime."""
    idx = {p: i + 1 for i, p in enumerate(primes)}
    rows = []
    for i, dv in enumerate(dvals, 1):
        if dv in idx:
            rows.append({"Rule":"Date ↔ Prime", "Detail":f"V{i}={dv} is prime (#{idx[dv]})", "Match":dv})
    return pd.DataFrame(rows)

def three_way(subject_name, jersey, dvals, primes):
    """3-Way: G/J ↔ Date ↔ Prime (zero-insensitive for Date comparisons)."""
    gvals = gematria_values(subject_name)
    idx = {p: i + 1 for i, p in enumerate(primes)}
    rows = []
    sources = list(gvals.items()) + [("Jersey", jersey)]
    gj_vals = set(gvals.values()) | {jersey}

    def add(rule, lab, n, i, dv, detail, m):
        rows.append({"Rule": rule, "Source": f"{lab}({n})",
                     "Date Value": f"V{i}={dv}", "Prime Detail": detail, "Match": m})

    for lab, n in sources:
        # A) Same number is prime and matches Date
        if n in idx:
            for i, dv in enumerate(dvals, 1):
                if zero_insensitive_equal(n, dv):
                    add("A) Same prime", lab, n, i, dv, f"{n} is prime (#{idx[n]})", n)
        # B) n as index → prime value matches Date
        if isinstance(n, int) and 1 <= n <= len(primes):
            p = primes[n - 1]
            for i, dv in enumerate(dvals, 1):
                if zero_insensitive_equal(p, dv):
                    add("B) Index→prime", lab, n, i, dv, f"{n}th prime = {p}", p)
            # C) digit-sum(p_n) equals a G/J value; Date tied via n or p_n
            s = digit_sum(p)
            if s in gj_vals:
                for i, dv in enumerate(dvals, 1):
                    if zero_insensitive_equal(n, dv) or zero_insensitive_equal(p, dv):
                        via = "n" if zero_insensitive_equal(n, dv) else "p_n"
                        add("C) Index→digit-sum", lab, n, i, dv,
                            f"{n}th prime = {p}, digit-sum = {s} (G/J match; Date via {via})", s)
    return pd.DataFrame(rows)

# =========================
# Processing helper (one subject)
# =========================
def process_subject(label, subject_name, jersey, dvals, primes):
    st.markdown(f"### {label}: **{subject_name}** (Jersey: {jersey})")

    st.caption("Gematria")
    gvals = gematria_values(subject_name)
    st.write(gvals)

    st.caption("1-Way Matches: G/J ↔ Date (strict)")
    df_gd = style_duplicates(pairwise_gd(subject_name, jersey, dvals))
    st.dataframe(df_gd, use_container_width=True) if (df_gd is not None and not df_gd.empty) else st.info("None.")

    st.caption("Gematria ↔ Date (Zero-insensitive)")
    df_gd_zero = style_duplicates(pairwise_gd_zero_insensitive(subject_name, dvals))
    st.dataframe(df_gd_zero, use_container_width=True) if (df_gd_zero is not None and not df_gd_zero.empty) else st.info("None.")

    st.caption("2-Way: Jersey → Prime equals Date")
    df_sec2 = style_duplicates(pairwise_section2_jersey_prime_matches_date(jersey, primes, dvals))
    st.dataframe(df_sec2, use_container_width=True) if (df_sec2 is not None and not df_sec2.empty) else st.info("None.")

    st.caption("3-Way: G/J ↔ Date ↔ Prime")
    df_3w = style_duplicates(three_way(subject_name, jersey, dvals, primes))
    st.dataframe(df_3w, use_container_width=True) if (df_3w is not None and not df_3w.empty) else st.info("None.")

# =========================
# UI — Global date + optional CSV
# =========================
st.title("Numerology — Home & Away (CSV or Manual)")

date_input = st.date_input("Date", value=date.today())

st.divider()
st.subheader("Optional: Upload teams CSV (must include a 'team_full' column)")
uploaded = st.file_uploader("Upload CSV", type=["csv"], help="NBA/MLB/NHL files you made will work out of the box.")
teams_df = None
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        if "team_full" not in df.columns:
            st.error("CSV load error: CSV must include at least 'team_full'. Other columns are optional.")
        else:
            teams_df = df
            with st.expander("Preview uploaded teams", expanded=False):
                st.dataframe(teams_df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

st.divider()
cols = st.columns(2)

# ---------- HOME INPUT ----------
with cols[0]:
    st.subheader("Home")
    method_home = st.radio("Input method (Home)", ["Manual text", "From CSV"], horizontal=True)
    if method_home == "From CSV":
        if teams_df is not None:
            home_name = st.selectbox("Select Home Team", teams_df["team_full"], key="home_select")
        else:
            st.warning("No CSV loaded. Switching to Manual.")
            method_home = "Manual text"
            home_name = st.text_input("Home name (manual)", key="home_manual")
    else:
        home_name = st.text_input("Home name (manual)", key="home_manual")
    jersey_home = st.number_input("Home jersey number (optional; used for jersey→prime rule)", min_value=0, step=1, value=0, key="home_jersey")

# ---------- AWAY INPUT ----------
with cols[1]:
    st.subheader("Away")
    method_away = st.radio("Input method (Away)", ["Manual text", "From CSV"], horizontal=True)
    if method_away == "From CSV":
        if teams_df is not None:
            away_name = st.selectbox("Select Away Team", teams_df["team_full"], key="away_select")
        else:
            st.warning("No CSV loaded. Switching to Manual.")
            method_away = "Manual text"
            away_name = st.text_input("Away name (manual)", key="away_manual")
    else:
        away_name = st.text_input("Away name (manual)", key="away_manual")
    jersey_away = st.number_input("Away jersey number (optional; used for jersey→prime rule)", min_value=0, step=1, value=0, key="away_jersey")

st.divider()

# =========================
# Calculate
# =========================
if st.button("Calculate"):
    if not str(home_name).strip() or not str(away_name).strip():
        st.error("Please enter names for both Home and Away.")
    else:
        primes = generate_first_n_primes(1000)
        dvals = date_values(date_input)

        st.subheader("Date Values (V1..V10)")
        st.write({f"V{i+1}": v for i, v in enumerate(dvals)})

        st.subheader("Date Digit Sums")
        df_dd = style_duplicates(date_digit_sums_table(dvals))
        if df_dd is not None and not df_dd.empty:
            st.dataframe(df_dd, use_container_width=True)
        else:
            st.info("No date values found.")

        st.divider()
        with st.expander(f"Home Results — {home_name}", expanded=True):
            process_subject("HOME", home_name.strip(), int(jersey_home), dvals, primes)

        with st.expander(f"Away Results — {away_name}", expanded=True):
            process_subject("AWAY", away_name.strip(), int(jersey_away), dvals, primes)

        st.divider()
        st.caption("Note: Section 2 uses only the jersey→prime index rule against Date values (zero-insensitive). Gematria↔Date (zero-insensitive) also shown separately.")
