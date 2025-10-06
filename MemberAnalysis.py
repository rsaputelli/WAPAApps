
import io
import pandas as pd
import streamlit as st
import altair as alt

# ============================
# Constants
# ============================
RAW_MEMBER_TYPE_COL = "Member/Non-Member - Member Type"
RAW_MEMBERSHIP_COL  = "Member/Non-Member - Membership"

ID_COL        = "Member/Non-Member - Website ID"
LAST_DUES_COL = "Member/Non-Member - Date Last Dues Transaction"
EXP_COL       = "Member/Non-Member - Date Membership Expires"

TRANSACTION_HINT_COLS = {"Dues - Amount", "Dues - Date Processed", "Dues - Date Submitted"}

HEADER_MAP = {
    "Member Type - Member Type": RAW_MEMBER_TYPE_COL,
    "Dues - Membership": RAW_MEMBERSHIP_COL,
    "Member/Non-Member - Date Membership Expires": EXP_COL,
    "Member/Non-Member - Date Last Dues Transaction": LAST_DUES_COL,
    "Member/Non-Member - Website ID": ID_COL,
    "Member/Non-Member - Email": "Member/Non-Member - Email",
    "Member/Non-Member - First Name": "Member/Non-Member - First Name",
    "Member/Non-Member - Last Name": "Member/Non-Member - Last Name",
}

# ============================
# Core summary (same outputs)
# ============================
def compute_summary_from_raw(df: pd.DataFrame):
    if RAW_MEMBER_TYPE_COL not in df.columns or RAW_MEMBERSHIP_COL not in df.columns:
        raise ValueError(
            f"Required columns missing. Expected: '{RAW_MEMBER_TYPE_COL}' and '{RAW_MEMBERSHIP_COL}'. "
            f"Got: {list(df.columns)}"
        )

    member_type_totals = df[RAW_MEMBER_TYPE_COL].value_counts(dropna=False).sort_index()

    membership_breakdown = (
        df[[RAW_MEMBER_TYPE_COL, RAW_MEMBERSHIP_COL]]
        .value_counts(dropna=False)
        .rename("count")
        .reset_index()
        .sort_values([RAW_MEMBER_TYPE_COL, RAW_MEMBERSHIP_COL])
    )
    return member_type_totals, membership_breakdown

# ============================
# Preprocessing helpers
# ============================
def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: c.strip() for c in df.columns}).rename(columns=HEADER_MAP)

def _detect_report_type(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    has_trans_cols = TRANSACTION_HINT_COLS.issubset(cols)
    has_dupes = (ID_COL in df.columns) and df[ID_COL].duplicated().any()
    return "transaction" if (has_trans_cols or has_dupes) else "member"

def _dedup_transaction_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in (LAST_DUES_COL, EXP_COL):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    is_student = (
        df.get(RAW_MEMBER_TYPE_COL, pd.Series(dtype=object))
          .astype(str).str.strip().str.lower().eq("student")
    )
    df["__is_student"] = is_student
    sort_cols, asc = ["__is_student"], [True]
    if LAST_DUES_COL in df.columns:
        sort_cols.append(LAST_DUES_COL); asc.append(False)
    if EXP_COL in df.columns:
        sort_cols.append(EXP_COL); asc.append(False)
    if ID_COL in df.columns:
        sort_cols.append(ID_COL); asc.append(True)
    df = df.sort_values(by=sort_cols, ascending=asc)
    out = df.drop_duplicates(subset=[ID_COL], keep="first").drop(columns="__is_student", errors="ignore")
    return out

def preprocess_input(df_raw: pd.DataFrame, mode: str = "auto"):
    df = _normalize_headers(df_raw)

    # Ensure required columns exist even if original headers vary
    if RAW_MEMBER_TYPE_COL not in df.columns and "Member Type - Member Type" in df.columns:
        df = df.rename(columns={"Member Type - Member Type": RAW_MEMBER_TYPE_COL})
    if RAW_MEMBERSHIP_COL not in df.columns and "Dues - Membership" in df.columns:
        df = df.rename(columns={"Dues - Membership": RAW_MEMBERSHIP_COL})

    detected = _detect_report_type(df) if mode == "auto" else mode
    info = {"detected_mode": detected}

    if detected == "transaction":
        df_clean = _dedup_transaction_frame(df)
        info.update({
            "rows_in": len(df),
            "rows_out": len(df_clean),
            "unique_ids_in": (df[ID_COL].nunique() if ID_COL in df.columns else None),
            "unique_ids_out": (df_clean[ID_COL].nunique() if ID_COL in df_clean.columns else None),
            "dedup_applied": True,
        })
    else:
        df_clean = df
        info.update({
            "rows_in": len(df),
            "rows_out": len(df_clean),
            "unique_ids_in": (df[ID_COL].nunique() if ID_COL in df.columns else None),
            "unique_ids_out": (df_clean[ID_COL].nunique() if ID_COL in df_clean.columns else None),
            "dedup_applied": False,
        })

    # Guarantee required columns for summary
    for needed in (RAW_MEMBER_TYPE_COL, RAW_MEMBERSHIP_COL):
        if needed not in df_clean.columns:
            df_clean[needed] = ""

    return df_clean, info

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Member Analysis", layout="wide")
st.title("Member Analysis (Report-Agnostic)")

uploaded = st.file_uploader("Choose CSV export", type=["csv"])
if uploaded is None:
    st.info("Upload a YM CSV: either 'Today & After' (transaction-level) or 'As-of' (member-level).")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded)
except UnicodeDecodeError:
    df_raw = pd.read_csv(uploaded, encoding="latin-1")

st.caption(f"Raw columns ({len(df_raw.columns)}): {list(df_raw.columns)}")

mode = st.radio(
    "Report type",
    options=["Auto","Transaction-level","Member-level"],
    index=0,
    horizontal=True
)
mode_map = {"Auto":"auto", "Transaction-level":"transaction", "Member-level":"member"}
df_clean, info = preprocess_input(df_raw, mode=mode_map[mode])

with st.expander("Input detection & dedup summary", expanded=False):
    st.json(info)
    st.dataframe(df_clean.head(20))

member_type_totals, membership_breakdown = compute_summary_from_raw(df_clean)

st.subheader("Member Type Totals")
totals_df = member_type_totals.rename_axis("Member Type").reset_index(name="count")
st.dataframe(totals_df, use_container_width=True)

# Bar chart (Altair)
chart = (
    alt.Chart(totals_df)
    .mark_bar()
    .encode(x=alt.X("Member Type:N", sort=None), y="count:Q", tooltip=["Member Type","count"])
    .properties(height=300)
)
st.altair_chart(chart, use_container_width=True)

st.subheader("Membership Breakdown")
st.dataframe(membership_breakdown, use_container_width=True)

csv_buf = io.StringIO()
df_clean.to_csv(csv_buf, index=False)
st.download_button(
    "Download cleaned member-level CSV",
    data=csv_buf.getvalue(),
    file_name="member_level_clean.csv",
    mime="text/csv"
)
