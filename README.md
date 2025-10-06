# WAPA – Member Summary, Comparison & Trends

## Overview
This Streamlit app generates monthly membership summaries and analyzes changes over time using YourMembership (YM) exports.

It includes three major capabilities:
1. **Generate Monthly Summary** — from a raw YM export.
2. **Month-over-Month Comparison** — compare two monthly summaries.
3. **Longitudinal Trends** — build a running master workbook for trend tracking.

---

## Part 1. Generate Monthly Summary
1. In **YourMembership**, run the “Today and After” or “As of [Date]” export.
2. Upload the resulting CSV (or Excel) file to **Tab 1 – Generate Monthly Summary**.
3. The app:
   - Cleans duplicates (if detected).
   - Displays preview tables for Member Type and Membership Breakdown.
   - Generates side-by-side bar charts.
4. Download options:
   - **Summary Excel file**: `YYYY.MM.DD_Member_Type_Summary.xlsx`
   - **Charts as PNGs**

---

## Part 2. Month-over-Month Comparison
1. In **Tab 2 – Month-over-Month Comparison**, upload:
   - Left report (e.g., prior month)
   - Right report (e.g., current month)
   - Both may be CSV or XLSX summaries.
2. The app automatically:
   - Detects date from filenames and labels charts accordingly.
   - Computes Δ and %Δ by Member Type.
   - Displays comparison charts (Counts + Change).
   - Exports a combined workbook `YYYY.MM.DD_Summary_Comparison.xlsx` including labeled charts.

💡 **Download Buttons:** Each chart can be downloaded as a high-resolution PNG for presentations.

---

## Part 3. Longitudinal Trend Analysis
1. In **Tab 3 – Build Master Workbook from Summaries**:
   - Upload multiple `Member_Type_Summary` workbooks (one per month).
2. The app merges all into a time-series by Member Type.
3. It produces:
   - A running **Trend** workbook (`YYYY.MM.DD_Member_Trends.xlsx`)
   - Line charts per Member Type (cleaner, labeled axes, no data-point clutter)
   - Option to **download the trend chart as PNG**

---

## Notes
- Accepts both **CSV** and **XLSX**.
- Chart PNGs match the on-screen visuals.
- Deduplication toggle available to clean raw YM exports if needed.
- Run reports consistently near the first of each month for accuracy.
- Save monthly summaries in Dropbox for continuity.
