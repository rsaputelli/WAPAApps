import io
import re
from datetime import datetime
from typing import Tuple, Dict, List

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# ---------------------------
# Helpers
# ---------------------------

RUN_DATE_STR = datetime.today().strftime("%Y-%m-%d")

RAW_MEMBER_TYPE_COL = "Member/Non-Member - Member Type"
RAW_MEMBERSHIP_COL   = "Member/Non-Member - Membership"

def compute_summary_from_raw(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      member_type_totals: DataFrame with columns [Member Type, Count]
      membership_breakdown: DataFrame with columns [Member Type, Membership, Count]
    """
    # Clean column names (robust to light changes)
    cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=cols)

    # Basic guards
    for need in [RAW_MEMBER_TYPE_COL, RAW_MEMBERSHIP_COL]:
        if need not in df.columns:
            raise ValueError(f"Expected column not found: {need}")

    # Member Type totals
    member_type_totals = (
        df[RAW_MEMBER_TYPE_COL]
        .value_counts()
        .rename_axis("Member Type")
        .reset_index(name="Count")
        .sort_values("Member Type", kind="stable")
    )

    # Membership breakdown within Member Type
    membership_breakdown = (
        df.groupby([RAW_MEMBER_TYPE_COL, RAW_MEMBERSHIP_COL])
          .size()
          .reset_index(name="Count")
          .rename(columns={
              RAW_MEMBER_TYPE_COL: "Member Type",
              RAW_MEMBERSHIP_COL: "Membership"
          })
          .sort_values(["Member Type", "Membership"], kind="stable")
    )

    return member_type_totals, membership_breakdown


def make_charts_from_summary(
    member_type_totals: pd.DataFrame,
    membership_breakdown: pd.DataFrame,
    title_prefix: str = "WAPA"
) -> Tuple[plt.Figure, plt.Figure]:
    """Builds the 2 charts and returns matplotlib Figures."""

    # 1) Bar chart – totals by Member Type (with numbers above bars)
    counts = member_type_totals.set_index("Member Type")["Count"]
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    counts.plot(kind="bar", ax=ax1)
    ax1.set_title(f"{title_prefix} - Total Members by Member Type\nRun Date: {RUN_DATE_STR}")
    ax1.set_xlabel("Member Type")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)
    for p in ax1.patches:
        ax1.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha="center", va="bottom", fontsize=9, xytext=(0, 3),
            textcoords="offset points"
        )
    fig1.tight_layout()

    # 2) Stacked bar with legend counts (clean legend, no per-segment labels)
    pivot = (
        membership_breakdown
        .pivot(index="Member Type", columns="Membership", values="Count")
        .fillna(0)
        .astype(int)
        .sort_index()
    )

    # Build legend with totals across all Member Types
    membership_totals = pivot.sum(axis=0).to_dict()
    legend_labels = [f"{m} ({membership_totals.get(m, 0)})" for m in pivot.columns]

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    pivot.plot(kind="bar", stacked=True, ax=ax2)
    ax2.set_title(f"{title_prefix} - Membership Breakdown by Member Type\nRun Date: {RUN_DATE_STR}")
    ax2.set_xlabel("Member Type")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=45)
    ax2.legend(legend_labels, title="Membership", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig2.tight_layout()

    return fig1, fig2


def build_summary_excel(member_type_totals: pd.DataFrame,
                        membership_breakdown: pd.DataFrame) -> bytes:
    """Return a bytes Excel file with two sheets: Member Type Totals, Membership Breakdown."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        member_type_totals.to_excel(writer, index=False, sheet_name="Member Type Totals")
        membership_breakdown.to_excel(writer, index=False, sheet_name="Membership Breakdown")

        wb = writer.book
        # Optional header format
        header_fmt = wb.add_format({"bold": True})

        for sheet_name, df in [
            ("Member Type Totals", member_type_totals),
            ("Membership Breakdown", membership_breakdown),
        ]:
            ws = writer.sheets[sheet_name]
            # Bold header row
            ws.set_row(0, None, header_fmt)

            # Set column widths without re-reading the file
            for i, col in enumerate(df.columns):
                # Heuristic: header width vs first 100 values
                samples = df[col].astype(str).head(100).tolist()
                max_len = max([len(str(col))] + [len(s) for s in samples])
                ws.set_column(i, i, min(max_len + 2, 40))  # pad a bit, cap at 40

    buf.seek(0)
    return buf.read()



MONTH_RE = re.compile(r"(\d{4})[-_]?(\d{2})")  # prefer YYYY-MM or YYYYMM in filename


def infer_month_from_filename(filename: str) -> str:
    """
    Tries to find 'YYYY-MM' from the filename. Falls back to today's month.
    Returns 'YYYY-MM'.
    """
    m = MONTH_RE.search(filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return datetime.today().strftime("%Y-%m")


def aggregate_summaries_to_master(files: List) -> bytes:
    """
    Accepts a list of uploaded summary Excel files (with "Member Type Totals" sheet),
    builds:
      - Time Series (Member Type): rows = member types, columns = months
      - MoM Δ
      - %Δ
      - Trend charts per member type
    Returns: master workbook as bytes.
    """
    # Load each file's "Member Type Totals"
    month_to_series: Dict[str, pd.Series] = {}
    all_member_types = set()

    for f in files:
        month_key = infer_month_from_filename(f.name)
        df = pd.read_excel(f, sheet_name="Member Type Totals")
        if not {"Member Type", "Count"}.issubset(df.columns):
            raise ValueError(f"{f.name} missing required columns in 'Member Type Totals'.")
        s = df.set_index("Member Type")["Count"].astype(int)
        month_to_series[month_key] = s
        all_member_types |= set(s.index)

    # Build time series table
    months_sorted = sorted(month_to_series.keys())
    ts = pd.DataFrame(index=sorted(all_member_types), columns=months_sorted).fillna(0).astype(int)
    for m, s in month_to_series.items():
        ts.loc[s.index, m] = s

    # MoM Δ and %Δ
    mom = ts.diff(axis=1).fillna(0).astype(int)
    pct = ts.pct_change(axis=1).replace([pd.NA, pd.NaT], 0.0).fillna(0.0) * 100.0

    # Write to Excel with charts using xlsxwriter
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        ts.to_excel(writer, sheet_name="Time Series (Member Type)")
        mom.to_excel(writer, sheet_name="MoM Δ")
        pct.to_excel(writer, sheet_name="%Δ")

        wb = writer.book
        ws_ts = writer.sheets["Time Series (Member Type)"]

        # Add a trends worksheet with charts
        ws_charts = wb.add_worksheet("Trend Charts")
        title_fmt = wb.add_format({"bold": True, "font_size": 14})
        ws_charts.write(0, 0, f"WAPA – Member Type Trends (Run: {RUN_DATE_STR})", title_fmt)

        # Create a line chart per member type, arranged in a grid
        # Data range: ts sheet. Row offsets (+1 header), col offsets (+1 index col)
        start_row = 2
        charts_per_col = 3
        chart_width = 9  # columns
        chart_height = 15  # rows

        for idx, mtype in enumerate(ts.index):
            chart = wb.add_chart({"type": "line"})
            # Category labels are the months (header row)
            chart.add_series({
                "name": str(mtype),
                "categories": ["Time Series (Member Type)", 0, 1, 0, len(months_sorted)],
                "values":     ["Time Series (Member Type)", idx + 1, 1, idx + 1, len(months_sorted)],
            })
            chart.set_title({"name": str(mtype)})
            chart.set_legend({"none": True})
            chart.set_y_axis({"name": "Count"})
            chart.set_x_axis({"name": "Month"})

            grid_row = idx % charts_per_col
            grid_col = idx // charts_per_col

            ws_charts.insert_chart(
                start_row + grid_row * chart_height,
                grid_col * chart_width,
                chart,
                {"x_scale": 1.1, "y_scale": 1.1}
            )

        # Autofit columns lightly for the 3 sheets
        for sheet in ["Time Series (Member Type)", "MoM Δ", "%Δ"]:
            ws = writer.sheets[sheet]
            df_sheet = {"Time Series (Member Type)": ts, "MoM Δ": mom, "%Δ": pct}[sheet]
            for i in range(len(df_sheet.columns) + 1):
                ws.set_column(i, i, 16)

    out.seek(0)
    return out.read()


# ---------------------------
# UI
# ---------------------------

st.set_page_config(page_title="WAPA – Member Summary & Trends", layout="wide")

# --- Header with logo on the left and title on the right ---
LOGO_PATH = Path(__file__).parent / "logo.png"

left, right = st.columns([1, 8])

with left:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=460)  # adjust as desired
    else:
        st.write("")  # spacer if logo missing

with right:
    st.markdown(
        f"""
        <div style="padding-top:6px;">
          <h1 style="margin-bottom:0;">WAPA – Member Type Summary & Trend Builder</h1>
          <p style="color:#666; margin-top:4px;">Run Date: {RUN_DATE_STR}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

tab1, tab2 = st.tabs(["1) Create Monthly Summary from Raw Export", "2) Build Master Workbook from Summaries"])

with tab1:
    st.subheader("Upload a fresh raw member export (CSV)")
    raw_file = st.file_uploader("Choose CSV export", type=["csv"])
    if raw_file is not None:
        try:
            df_raw = pd.read_csv(raw_file)
        except UnicodeDecodeError:
            # Fallback if CSV is UTF-16/Windows-1252, etc.
            raw_file.seek(0)
            df_raw = pd.read_csv(raw_file, encoding="latin-1")

        st.write("Preview:", df_raw.head())

        member_type_totals, membership_breakdown = compute_summary_from_raw(df_raw)

        # Show dataframes
        st.markdown("**Member Type Totals**")
        st.dataframe(member_type_totals, use_container_width=True)

        st.markdown("**Membership Breakdown**")
        st.dataframe(membership_breakdown, use_container_width=True)

        # Charts
        fig1, fig2 = make_charts_from_summary(member_type_totals, membership_breakdown, title_prefix="WAPA")
        st.pyplot(fig1, use_container_width=True)
        st.pyplot(fig2, use_container_width=True)
        
        # Chart download buttons
        import io
        buf1 = io.BytesIO()
        fig1.savefig(buf1, format="png", bbox_inches="tight")
        buf1.seek(0)
        st.download_button(
            "Download Total Members Chart (PNG)",
            data=buf1,
            file_name=f"WAPA_Total_Members_{RUN_DATE_STR}.png",
            mime="image/png"
        )
        
        buf2 = io.BytesIO()
        fig2.savefig(buf2, format="png", bbox_inches="tight")
        buf2.seek(0)
        st.download_button(
            "Download Membership Breakdown Chart (PNG)",
            data=buf2,
            file_name=f"WAPA_Membership_Breakdown_{RUN_DATE_STR}.png",
            mime="image/png"
        )
        
        # Download Excel
        out_bytes = build_summary_excel(member_type_totals, membership_breakdown)
        out_name = f"Member_Type_Summary_{RUN_DATE_STR}.xlsx"
        st.download_button(
            "Download Summary Excel",
            data=out_bytes,
            file_name=out_name,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


with tab2:
    st.subheader("Upload multiple monthly summary Excel files")
    st.caption("Tip: name files like `Member_Type_Summary_YYYY-MM.xlsx` so months are auto-detected.")
    multi_files = st.file_uploader(
        "Choose one or more summary workbooks",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True
    )
    if multi_files:
        try:
            master_bytes = aggregate_summaries_to_master(multi_files)
            master_name = f"WAPA_Member_Type_Master_{RUN_DATE_STR}.xlsx"
            st.success("Master workbook created.")
            st.download_button(
                "Download Master Workbook",
                data=master_bytes,
                file_name=master_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Also show a quick line chart for the top 5 member types by last-month size
            # so you get an immediate visual in the app
            # Build the same TS to plot quickly:
            # (Reloading is cheap given small sizes)
            month_to_series = {}
            all_types = set()
            for f in multi_files:
                mk = infer_month_from_filename(f.name)
                df = pd.read_excel(f, sheet_name="Member Type Totals")
                s = df.set_index("Member Type")["Count"].astype(int)
                month_to_series[mk] = s
                all_types |= set(s.index)
            months_sorted = sorted(month_to_series.keys())
            ts = pd.DataFrame(index=sorted(all_types), columns=months_sorted).fillna(0).astype(int)
            for m, s in month_to_series.items():
                ts.loc[s.index, m] = s

            if not ts.empty:
                st.markdown("**Quick trend preview (top 5 by latest month)**")
                last_m = months_sorted[-1]
                top5 = ts.sort_values(last_m, ascending=False).head(5)

                fig3, ax3 = plt.subplots(figsize=(10, 5))
                for mtype, row in top5.iterrows():
                    ax3.plot(months_sorted, row.values, marker="o", label=mtype)
                ax3.set_title(f"WAPA – Top 5 Member Types Trend (Run: {RUN_DATE_STR})")
                ax3.set_xlabel("Month")
                ax3.set_ylabel("Count")
                ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                fig3.tight_layout()
                st.pyplot(fig3, use_container_width=True)

        except Exception as e:
            st.error(str(e))
