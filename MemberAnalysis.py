
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
# Excel builder for legacy MoM output
# ============================
def build_legacy_summary_xlsx(totals_df: pd.DataFrame, membership_breakdown: pd.DataFrame) -> bytes:
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Member Type Totals"
    legacy_totals = totals_df.rename(columns={"Member Type":"index", "count":"Member/Non-Member - Member Type"})
    for r in dataframe_to_rows(legacy_totals, index=False, header=True):
        ws1.append(r)
    ws2 = wb.create_sheet("Membership Breakdown")
    legacy_breakdown = membership_breakdown.rename(columns={"count":"Count"})
    for r in dataframe_to_rows(legacy_breakdown, index=False, header=True):
        ws2.append(r)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    wb.save(tmp.name)
    tmp.seek(0)
    data = tmp.read()
    tmp.close()
    return data

def xlsx_from_frames(frames: dict) -> bytes:
    from openpyxl import Workbook
    from openpyxl.utils.dataframe import dataframe_to_rows
    wb = Workbook()
    names = list(frames.keys())
    ws = wb.active
    ws.title = names[0]
    for r in dataframe_to_rows(frames[names[0]], index=False, header=True):
        ws.append(r)
    for name in names[1:]:
        ws2 = wb.create_sheet(name)
        for r in dataframe_to_rows(frames[name], index=False, header=True):
            ws2.append(r)
    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    wb.save(tmp.name)
    tmp.seek(0)
    data = tmp.read()
    tmp.close()
    return data

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="Member Analysis", layout="wide")
st.title("Member Analysis (Report-Agnostic)")

tab1, tab2 = st.tabs(["Generate Monthly Member Summary", "Analyze Month-Over-Month Changes"])

# ---------------- Tab 1: Summary ----------------
with tab1:
    uploaded = st.file_uploader("Choose CSV export", type=["csv"], key="single")
    if uploaded is None:
        st.info("Upload a YM CSV: either 'Today & After' (transaction-level) or 'As-of' (member-level).")
    else:
        try:
            df_raw = pd.read_csv(uploaded)
        except UnicodeDecodeError:
            df_raw = pd.read_csv(uploaded, encoding="latin-1")

        st.caption(f"Raw columns ({len(df_raw.columns)}): {list(df_raw.columns)}")

        mode = st.radio(
            "Report type",
            options=["Auto","Transaction-level","Member-level"],
            index=0,
            horizontal=True,
            key="mode_single"
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

        chart = (
            alt.Chart(totals_df)
            .mark_bar()
            .encode(x=alt.X("Member Type:N", sort=None), y="count:Q", tooltip=["Member Type","count"])
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

        st.subheader("Membership Breakdown")
        st.dataframe(membership_breakdown, use_container_width=True)

        # Legacy export
        legacy_bytes = build_legacy_summary_xlsx(totals_df, membership_breakdown)
        st.download_button(
            "Download Member_Type_Summary.xlsx (legacy format)",
            data=legacy_bytes,
            file_name="Member_Type_Summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Cleaned CSV
        csv_buf = io.StringIO()
        df_clean.to_csv(csv_buf, index=False)
        st.download_button(
            "Download cleaned member-level CSV",
            data=csv_buf.getvalue(),
            file_name="member_level_clean.csv",
            mime="text/csv"
        )

# ---------------- Tab 2: Compare ----------------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        f_left = st.file_uploader("Left report (e.g., Last Month)", type=["csv"], key="left")
        mode_left = st.radio("Left report type", ["Auto","Transaction-level","Member-level"], index=0, horizontal=True, key="mode_left")
    with c2:
        f_right = st.file_uploader("Right report (e.g., This Month)", type=["csv"], key="right")
        mode_right = st.radio("Right report type", ["Auto","Transaction-level","Member-level"], index=0, horizontal=True, key="mode_right")

    if f_left and f_right:
        try:
            df_left_raw = pd.read_csv(f_left)
        except UnicodeDecodeError:
            df_left_raw = pd.read_csv(f_left, encoding="latin-1")
        try:
            df_right_raw = pd.read_csv(f_right)
        except UnicodeDecodeError:
            df_right_raw = pd.read_csv(f_right, encoding="latin-1")

        m = {"Auto":"auto","Transaction-level":"transaction","Member-level":"member"}
        df_left, info_left = preprocess_input(df_left_raw, mode=m[mode_left])
        df_right, info_right = preprocess_input(df_right_raw, mode=m[mode_right])

        st.markdown("#### Detection & Dedup Summaries")
        colA, colB = st.columns(2)
        with colA:
            st.json(info_left)
        with colB:
            st.json(info_right)

        # Compare by Website ID
        left_ids  = set(df_left[ID_COL].dropna().astype(str)) if ID_COL in df_left.columns else set()
        right_ids = set(df_right[ID_COL].dropna().astype(str)) if ID_COL in df_right.columns else set()
        in_both   = sorted(left_ids & right_ids)
        only_left = sorted(left_ids - right_ids)
        only_right= sorted(right_ids - left_ids)

        df_both = df_left[df_left[ID_COL].astype(str).isin(in_both)].copy()
        df_only_left = df_left[df_left[ID_COL].astype(str).isin(only_left)].copy()
        df_only_right = df_right[df_right[ID_COL].astype(str).isin(only_right)].copy()

        st.markdown("### Results")
        st.write({
            "Overlap (members in both)": len(in_both),
            "Only in Left": len(only_left),
            "Only in Right": len(only_right),
        })

        st.markdown("#### Overlap (Both)")
        cols_needed = [ID_COL, "Member/Non-Member - First Name", "Member/Non-Member - Last Name",
                       "Member/Non-Member - Email", RAW_MEMBER_TYPE_COL, EXP_COL, LAST_DUES_COL]
        cols_present = [c for c in cols_needed if c in df_both.columns]
        st.dataframe(df_both[cols_present].sort_values(by=ID_COL), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Only in Left")
            cols_present_l = [c for c in cols_needed if c in df_only_left.columns]
            st.dataframe(df_only_left[cols_present_l].sort_values(by=ID_COL), use_container_width=True)
        with c4:
            st.markdown("#### Only in Right")
            cols_present_r = [c for c in cols_needed if c in df_only_right.columns]
            st.dataframe(df_only_right[cols_present_r].sort_values(by=ID_COL), use_container_width=True)

        frames = {
            "Overlap (Both)": df_both[cols_present] if cols_present else df_both,
            "Only in Left": df_only_left[cols_present_l] if cols_present_l else df_only_left,
            "Only in Right": df_only_right[cols_present_r] if cols_present_r else df_only_right,
        }
        xlsx_bytes = xlsx_from_frames(frames)
        st.download_button(
            "Download comparison workbook (.xlsx)",
            data=xlsx_bytes,
            file_name="Report_Comparison.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("Upload two files to run the comparison.")
