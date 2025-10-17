# wapa_recon_app.py
# Streamlit app for WAPA PayPal ↔ YM ↔ Bank reconciliation
# Features:
# - JE Lines grouped by deposit (deposit_gid), DEBITs first
# - Split Debit / Credit columns in export
# - PayPal Fees posted as positive debits (robust to missing cols)
# - Membership deferrals using YM Column Z + mid-month rule
# - 12/24-month deferral logic
# - DISCOUNTS ignored via YM Payment Description / Item Description
# - VAT → 4314 (detects YM "Payment Allocation (Year-to-Date) - Item Description" = Tax/VAT; excluded from deferral)
# - PAC donations → 2202 (YM GL/text + PayPal-only Item Title when not matched to YM)
# - PayPal-only Invoice Payments → 99999 (placeholder, flagged for manual coding)
# - YM Payment Date normalization & period filter (prevents next-month bleed)
# - Excel: all money columns formatted as currency (2 decimals)

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="WAPA Recon (JE grouped + Deferrals + PAC + VAT)", layout="wide")

# ------------------------- Config -------------------------
BANK_GL = "1002 · TD Bank Checking x6455"
AUTO_BALANCE = False  # set True to add a 9999 balancing line per deposit

# Expense routing for fees by item title
FEE_ACCT_MEMBERSHIP = "5104 · Membership Expenses:5104 · Dues Expense"
FEE_ACCT_CONFERENCE = "5316 · Fall Conference Expenses:5316 · Registration"
FEE_ACCT_OTHER      = "5012 · Administrative Expenses:5012 · Bank Charges/Credit Card Fees"

# Revenue (current-year) by membership type
REV_DUES_BY_TYPE = {
    "fellow":       "4101 · Fellow Membership",
    "member":       "4102 · Member Membership",
    "affiliate":    "4103 · Affiliate Membership",
    "organization": "4104 · Organizational Membership",
    "student":      "4105 · Student Membership",
    "sustaining":   "4106 · Sustaining Membership",
    "hardship":     "4107 · Hardship Membership",
}
REV_MEMBERSHIP_DEFAULT = "4100 · Membership Dues"

# Deferred (liability) account names by membership type
DEFER_210_BY_TYPE = {
    "fellow":       "2101 · Deferred Dues - Fellow 2026",
    "member":       "2102 · Deferred Dues - Member 2026",
    "affiliate":    "2103 · Deferred Dues - Affiliate 2026",
    "organization": "2104 · Defer Dues - Organization 2026",
    "student":      "2105 · Deferred Dues - Student 2026",
    "sustaining":   "2106 · Deferred Dues - Sustaining 2026",
    "hardship":     "2107 · Deferred Dues - Hardship 2026",
}
DEFER_212_BY_TYPE = {
    "fellow":       "2125 · Deferred Dues - Fellow 2027",
    "member":       "2126 · Deferred Dues - Member 2027",
    "affiliate":    "2127 · Deferred Dues - Affiliate 2027",
    "organization": "2128 · Defer Dues - Organization 2027",
    "student":      "2129 · Deferred Dues - Student 2027",
    "sustaining":   "2130 · Deferred Dues - Sustaining 2027",
    "hardship":     "2131 · Deferred Dues - Hardship 2027",
}
DEF_MEMBERSHIP_DEFAULT_NEXT   = "2100 · Deferred Dues (Next FY)"
DEF_MEMBERSHIP_DEFAULT_FOLLOW = "2100 · Deferred Dues (Following FY)"

PAC_LIABILITY          = "2202 · Due to WAPA PAC"
VAT_OFFSET_INCOME      = "4314 · Offset of Credit Card Trans Fees"
UNMAPPED_REVIEW_ACCT   = "99999 · Needs Coding (Review)"

# ------------------------- Helpers -------------------------
def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def read_csv_robust(file) -> pd.DataFrame:
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file, engine="python")

def normalize_cols(df):
    df = df.copy()
    df.columns = [re.sub(r'\s+', ' ', str(c)).strip() for c in df.columns]
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    return df

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        for cand in candidates:
            if cand.lower() in col.lower():
                return col
    return None

def to_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.\-\+]", "", regex=True)
        .replace({"": np.nan, "-": np.nan})
        .astype(float)
    )

def choose_fee_account_from_item_title(item_title: str) -> str:
    t = norm_text(item_title)
    if t in {norm_text("Membership Dues"), norm_text("Membership+Dues")}:
        return FEE_ACCT_MEMBERSHIP
    if t in {norm_text("Online Store Order"), norm_text("Online+Store+Order")}:
        return FEE_ACCT_CONFERENCE
    return FEE_ACCT_OTHER

def effective_receipt_month(d: pd.Timestamp) -> pd.Timestamp:
    if pd.isna(d):
        return pd.NaT
    if d.day <= 15:
        return d.replace(day=1)
    y = d.year + (1 if d.month == 12 else 0)
    m = 1 if d.month == 12 else d.month + 1
    return pd.Timestamp(year=y, month=m, day=1)

def months_left_in_year(start_month: pd.Timestamp) -> int:
    if pd.isna(start_month):
        return 0
    return 12 - start_month.month + 1

def infer_member_type(mem_val: str) -> str:
    s = norm_text(mem_val)
    for k in ["fellow","member","affiliate","organization","student","sustaining","hardship"]:
        if k in s:
            return k
    return "member"

def is_two_year(mem_val: str) -> bool:
    s = str(mem_val or "").lower()
    return ("2-year" in s) or ("2year" in s) or ("two year" in s) or ("24" in s and "month" in s)

def is_discount_text(s: str) -> bool:
    return "discount" in str(s or "").lower()

def is_vat_text(s: str) -> bool:
    d_raw = str(s or "").strip()
    d = d_raw.lower()
    nt = re.sub(r"[^a-z0-9]", "", d)
    if d_raw.upper() in {"TAX/VAT", "TAX / VAT", "TAX VAT"} or nt in {"taxvat","vattax"}:
        return True
    return ("tax" in d) or ("vat" in d) or ("cc fee offset" in d) or ("credit card fee offset" in d) or ("processing fee offset" in d)

def is_pac_text(s: str) -> bool:
    s = str(s or "").lower()
    return ("pac" in s) or ("political action" in s) or ("pac donation" in s) or ("due to wapa pac" in s)

def is_pac_line(item_desc: str, gl_code: str, payment_desc: str) -> bool:
    g = str(gl_code or "").lower()
    if "2202" in g:
        return True
    return is_pac_text(item_desc) or is_pac_text(payment_desc) or ("pac" in g)

# ------------------------- UI -------------------------
st.title("WAPA PayPal ↔ YM ↔ Bank — JE grouped + Deferrals + PAC + VAT")

st.markdown("""
Upload CSVs (any column order is OK; headers auto-detected):

- **PayPal**: Type, Gross, Fee, Net, Transaction ID, Item Title, Source/Description (if present)
- **YM Export**: Invoice/Reference, Item Description (N), GL Code (O), Allocation (Q),
                **Member/Non-Member - Date Last Dues Transaction** (Z), **Payment Description**, Membership,
                **Payment Allocation (YTD) - Item Description**, **Payment Date** (any grouping)
- **Bank (TD)**: Date, Description, Deposit/Credit/Amount
""")

pp_file = st.file_uploader("PayPal CSV", type=["csv"], key="pp")
ym_file = st.file_uploader("YM Export CSV", type=["csv"], key="ym")
bank_file = st.file_uploader("Bank/TD CSV", type=["csv"], key="bank")

# --- Recon period controls ---
today = pd.Timestamp.today().normalize()
default_month_start = pd.Timestamp(year=today.year, month=today.month, day=1)

recon_anchor = st.date_input(
    "Reconcile month (pick any day IN the month)",
    value=default_month_start.date()
)
lead_bleed_days  = st.number_input("Include days BEFORE month start (front bleed)", min_value=0, max_value=31, value=0, step=1)
trail_bleed_days = st.number_input("Include days AFTER month end (back bleed)",   min_value=0, max_value=31, value=0, step=1)

run_btn = st.button("Run Reconciliation")

if run_btn:
    if not (pp_file and ym_file and bank_file):
        st.error("Please upload all three files.")
        st.stop()

    # --- Load & normalize ---
    pp = normalize_cols(read_csv_robust(pp_file))
    ym = normalize_cols(read_csv_robust(ym_file))
    bank = normalize_cols(read_csv_robust(bank_file))

    # --- PayPal columns ---
    pp_type_col       = find_col(pp, ["type"])
    pp_gross_col      = find_col(pp, ["gross"])
    pp_fee_col        = find_col(pp, ["fee", "fees"])
    pp_net_col        = find_col(pp, ["net"])
    pp_txn_col        = find_col(pp, ["transaction_id", "transactionid", "txn_id"])
    pp_date_col       = find_col(pp, ["date", "date_time", "time", "transaction_initiation_date"])
    pp_item_title_col = find_col(pp, ["item_title", "item_name", "title", "product_name"])
    pp_src_col        = find_col(pp, ["source", "note", "description", "memo", "name"])

    for col in [pp_gross_col, pp_fee_col, pp_net_col]:
        if col and pp[col].dtype == object:
            pp[col] = to_float(pp[col])

    pp["_parsed_date"] = pd.to_datetime(pp[pp_date_col], errors="coerce") if pp_date_col else pd.NaT
    pp["_type_norm"] = pp[pp_type_col].astype(str).str.strip().str.lower() if pp_type_col else ""

    withdrawal_terms = ["general withdrawal", "general witdrawal", "withdrawal", "bank deposit"]
    pp["_is_withdrawal"] = pp["_type_norm"].apply(lambda x: any(term in x for term in withdrawal_terms))

    # Normalize PayPal withdrawal signs to positive
    for col in [pp_gross_col, pp_fee_col, pp_net_col]:
        if col and col in pp.columns:
            mask = (pp["_is_withdrawal"]) & (pp[col] < 0)
            pp.loc[mask, col] = -pp.loc[mask, col]

    # Grouping deposits by withdrawal rows
    pp = pp.reset_index().rename(columns={"index": "_orig_idx"})
    pp = pp.sort_values(by=["_parsed_date", "_orig_idx"]).reset_index(drop=True)
    pp["_dep_gid"] = np.nan
    withdrawal_positions = pp.index[pp["_is_withdrawal"]].tolist()
    for k, idx_k in enumerate(withdrawal_positions):
        prev_idx = withdrawal_positions[k-1] if k > 0 else -1
        tx_mask = (pp.index > prev_idx) & (pp.index < idx_k) & (~pp["_is_withdrawal"])
        pp.loc[tx_mask, "_dep_gid"] = k
    pp.loc[pp.index.isin(withdrawal_positions), "_dep_gid"] = range(len(withdrawal_positions))

    withdrawals = pp.loc[pp["_is_withdrawal"]].copy()

    # --- YM columns ---
    ym_ref_col        = find_col(ym, ["invoice_-_reference_number", "invoice_reference_number", "reference_number", "invoice"])
    ym_item_desc_col  = find_col(ym, ["item_descriptions", "item_description", "item", "item_name", "name"])   # N (legacy)
    ym_gl_code_col    = find_col(ym, ["gl_codes", "gl_code", "gl", "account", "account_code"])                # O
    ym_alloc_col      = find_col(ym, ["allocation", "allocated_amount", "amount", "line_total"])              # Q
    ym_dues_rcpt_col  = find_col(ym, ["member/non-member_-_date_last_dues_transaction", "date_last_dues_transaction", "dues_paid_date", "paid_date"])
    ym_membership_col = find_col(ym, ["membership", "membership_type"])                                       # AC
    ym_pay_desc_col   = find_col(ym, ["payment_description", "payment_descriptions", "payment_desc", "ym_payment_description"])

    # Allocation Item Description (for VAT flagging)
    ym_alloc_item_desc_col = find_col(ym, [
        "payment_allocation_(year-to-date)_-_item_description",
        "payment_allocation_-_item_description",
        "payment_allocation_item_description",
        "payment_allocation_ytd_item_description",
        "payment_allocation_item_desc",
        "allocation_item_description",
    ])

    # NEW: normalize YM Payment Date (covers multiple export layouts)
    ym_pay_date_col = find_col(ym, [
        "payment_allocation_(year-to-date)_-_payment_date",
        "payment_allocation_-_payment_date",
        "payment_allocation_payment_date",
        "invoice_-_payment_date",
        "invoice_payment_date",
        "payment_date"
    ])
    ym["_ym_pay_date"] = pd.to_datetime(ym[ym_pay_date_col], errors="coerce") if ym_pay_date_col else pd.NaT

    if ym_alloc_col and ym[ym_alloc_col].dtype == object:
        ym[ym_alloc_col] = to_float(ym[ym_alloc_col])

    ym["_dues_rcpt"]  = pd.to_datetime(ym[ym_dues_rcpt_col], errors="coerce") if ym_dues_rcpt_col else pd.NaT
    ym["_eff_month"]  = ym["_dues_rcpt"].apply(effective_receipt_month)

    # --- Build recon month & window from controls ---
    anchor = pd.to_datetime(recon_anchor)
    recon_start = anchor.replace(day=1).normalize()
    recon_end   = (recon_start + pd.offsets.MonthEnd(1))
    window_start = recon_start - pd.Timedelta(days=int(lead_bleed_days))
    window_end   = recon_end   + pd.Timedelta(days=int(trail_bleed_days))

    # --- YM: keep only rows whose Payment Date falls inside the window
    if ym_pay_date_col:
        ym = ym.loc[ym["_ym_pay_date"].between(window_start, window_end)].copy()

    # --- PayPal: keep only CHILD transactions that:
    #     (a) belong to a deposit whose WITHDRAWAL DATE is inside the selected month
    #     (b) have a txn date inside the [window_start, window_end] window
    pp["_wd_date"] = np.where(pp["_is_withdrawal"], pp[pp_date_col], np.nan)
    wd_date_map = pd.to_datetime(pp.loc[pp["_is_withdrawal"], [ "_dep_gid", pp_date_col ]].set_index("_dep_gid")[pp_date_col], errors="coerce").to_dict()
    pp["_wd_date"] = pp["_dep_gid"].map(wd_date_map)
    pp["_wd_in_selected_month"] = pd.to_datetime(pp["_wd_date"], errors="coerce").between(recon_start, recon_end)

    pp["_child_in_window"] = (
        (~pp["_is_withdrawal"]) &
        (pp["_dep_gid"].notna()) &
        (pp["_wd_in_selected_month"]) &
        (pd.to_datetime(pp["_parsed_date"], errors="coerce").between(window_start, window_end))
    )

    transactions = pp.loc[pp["_child_in_window"]].copy()


    # --- Link PP transactions to YM by TransactionID ↔ Reference
    transactions["_pp_txn_key"] = transactions[pp_txn_col].astype(str).str.strip() if pp_txn_col else ""
    ym["_ym_ref_key"] = ym[ym_ref_col].astype(str).str.strip() if ym_ref_col else ""

    # Build pp↔ym join frame (include the Allocation Item Description for VAT detection)
    join_cols = [c for c in [
        ym_ref_col, "_ym_ref_key", ym_item_desc_col, ym_gl_code_col, ym_alloc_col,
        "_dues_rcpt", "_eff_month", ym_membership_col, ym_pay_desc_col, ym_alloc_item_desc_col
    ] if c]
    ppym = transactions.merge(
        ym[join_cols],
        left_on="_pp_txn_key",
        right_on="_ym_ref_key",
        how="left",
        suffixes=("", "_ym"),
    )

    # TRUE matches only: intersection of PP Txn IDs and YM Ref IDs
    pp_keys = set(transactions["_pp_txn_key"].dropna().astype(str))
    ym_keys = set(ym["_ym_ref_key"].dropna().astype(str))
    matched_txns = pp_keys & ym_keys

    # --- Aggregations per deposit ---
    tx_sums = (
        pp.loc[(pp["_dep_gid"].notna()) & (~pp["_is_withdrawal"])]
        .groupby("_dep_gid")
        .agg(
            tx_count=("net", "size" if pp_net_col else "size"),
            tx_gross_sum=(pp_gross_col, "sum"),
            tx_fee_sum=(pp_fee_col, "sum"),
            tx_net_sum=(pp_net_col, "sum"),
        )
    )

    wd = withdrawals.set_index("_dep_gid")[[pp_date_col, pp_gross_col, pp_fee_col, pp_net_col]].rename(
        columns={
            pp_date_col: "deposit_date",
            pp_gross_col: "withdrawal_gross",
            pp_fee_col: "withdrawal_fee",
            pp_net_col: "withdrawal_net",
        }
    )
    # --- Map PayPal withdrawals to Bank deposits to anchor month by bank posting date ---

    # 1) Identify bank columns and normalize amount
    bank_date_col   = find_col(bank, ["date", "posting_date", "transaction_date"])
    bank_desc_col   = find_col(bank, ["description", "memo", "details"])
    bank_credit_col = find_col(bank, ["deposit", "credit", "amount"])
    if bank_credit_col and bank[bank_credit_col].dtype == object:
        bank[bank_credit_col] = to_float(bank[bank_credit_col])
    bank["_bank_date"] = pd.to_datetime(bank[bank_date_col], errors="coerce") if bank_date_col else pd.NaT

    # 2) Keep bank rows in the selected RECON WINDOW (front/back bleed included)
    bank_in_window = bank.loc[
        bank["_bank_date"].between(window_start, window_end, inclusive="both")
    ].copy()

    # 3) Build a quick lookup of bank deposits by rounded amount -> list of (date, idx)
    #    (round to 2 decimals to match WD net; ignore non-positive/NaN)
    bank_in_window["_amt2"] = bank_in_window[bank_credit_col].round(2) if bank_credit_col else np.nan
    bank_match_index = {}
    if bank_credit_col:
        for i, r in bank_in_window.dropna(subset=["_amt2"]).iterrows():
            a = float(r["_amt2"])
            if a <= 0:
                continue
            bank_match_index.setdefault(a, []).append((r["_bank_date"], i))

    # 4) For each PayPal withdrawal, try to find a bank posting with the SAME amount
    #    within a small posting window (e.g., ±3 days). If found, use that bank date
    #    to decide whether this WD is "in the selected month."
    withdrawals = withdrawals.copy()
    withdrawals["_wd_bank_post_date"] = pd.NaT

    POST_DAYS_TOL = 3  # days between PP withdrawal and bank post allowed
    for i, r in withdrawals.iterrows():
        wd_gid = r["_dep_gid"]
        wd_amt = float(r.get(pp_net_col, 0) or 0)
        if wd_amt <= 0 or pd.isna(wd_gid):
            continue
        key = round(wd_amt, 2)
        if key in bank_match_index:
            wd_dt = pd.to_datetime(r.get(pp_date_col), errors="coerce")
            # choose the closest bank date by absolute delta (<= POST_DAYS_TOL)
            candidates = []
            for bdt, _idx in bank_match_index[key]:
                if pd.isna(wd_dt) or pd.isna(bdt):
                    continue
                delta = abs((bdt.normalize() - wd_dt.normalize()).days)
                if delta <= POST_DAYS_TOL:
                    candidates.append((delta, bdt))
            if candidates:
                candidates.sort(key=lambda x: x[0])
                withdrawals.loc[i, "_wd_bank_post_date"] = candidates[0][1]

    # 5) Month flag: TRUE if **either** the PP withdrawal date is in the selected month
    #    OR the matched bank posting date is in the selected month.
    wd_flags = []
    for i, r in withdrawals.iterrows():
        wd_dt  = pd.to_datetime(r.get(pp_date_col), errors="coerce")
        bnk_dt = pd.to_datetime(r.get("_wd_bank_post_date"), errors="coerce")
        in_month_by_pp  = pd.notna(wd_dt)  and wd_dt.between(recon_start, recon_end, inclusive="both")
        in_month_by_bank= pd.notna(bnk_dt) and bnk_dt.between(recon_start, recon_end, inclusive="both")
        wd_flags.append(in_month_by_pp or in_month_by_bank)
    withdrawals["_wd_in_selected_month"] = wd_flags

    # 6) Carry the bank-post date + flag into wd table (used by deposit_summary and JE)
    wd["_wd_bank_post_date"] = withdrawals.set_index("_dep_gid")["_wd_bank_post_date"]
    wd["_wd_in_selected_month"] = withdrawals.set_index("_dep_gid")["_wd_in_selected_month"]

    deposit_summary = wd.join(tx_sums, how="left")
    deposit_summary["calc_net"] = deposit_summary["tx_gross_sum"].fillna(0) - deposit_summary["tx_fee_sum"].fillna(0)
    deposit_summary["variance_vs_withdrawal"] = deposit_summary["withdrawal_net"].fillna(0) - deposit_summary["calc_net"]

    # --- Build Deferral Schedule
    deferral_rows = []
    if not ppym.empty and ym_alloc_col:
        for _, r in ppym.iterrows():
            alloc = r.get(ym_alloc_col, np.nan)
            if pd.isna(alloc) or float(alloc) == 0:
                continue

            item_desc = str(r.get(ym_item_desc_col, "") or "")
            pay_desc  = str(r.get(ym_pay_desc_col, "") or "")
            gl_code   = str(r.get(ym_gl_code_col, "") or "")
            mem_str   = r.get(ym_membership_col, "")
            alloc_item_desc = str(r.get(ym_alloc_item_desc_col, "") or "")

            if is_discount_text(pay_desc) or is_discount_text(item_desc):
                continue
            if is_pac_line(item_desc, gl_code, pay_desc):
                continue
            _vat_gate = (
                is_vat_text(item_desc) or is_vat_text(pay_desc) or
                is_vat_text(alloc_item_desc) or ("4314" in str(gl_code).lower())
            )
            if _vat_gate:
                continue

            is_membership_dues = (isinstance(mem_str, str) and mem_str.strip() != "") and ("dues" in item_desc.lower())
            if not is_membership_dues and (isinstance(mem_str, str) and mem_str.strip() != "") and gl_code.strip().startswith("410"):
                is_membership_dues = True

            if not is_membership_dues:
                continue

            eff_month = pd.to_datetime(r.get("_eff_month", pd.NaT), errors="coerce")
            if pd.isna(eff_month):
                # last resorts if needed
                eff_month = pd.to_datetime(r.get("_dues_rcpt", pd.NaT), errors="coerce")
            if pd.isna(eff_month):
                eff_month = pd.NaT

            mt = infer_member_type(mem_str)
            months_current = months_left_in_year(eff_month)
            total_months = 24 if is_two_year(mem_str) else 12

            cur_mo   = min(months_current, total_months)
            next_mo  = min(12, max(0, total_months - cur_mo))
            follow_mo= max(0, total_months - cur_mo - next_mo)

            per = float(alloc) / float(total_months)
            amt_cur    = round(per * cur_mo, 2)
            amt_next   = round(per * next_mo, 2)
            amt_follow = round(float(alloc) - amt_cur - amt_next, 2)

            deferral_rows.append({
                "deposit_gid": r.get("_dep_gid", np.nan),
                "TransactionID": r.get("_ym_ref_key", ""),
                "Membership": mem_str,
                "Item Description": item_desc,
                "Payment Description": pay_desc,
                "Receipt Date (col Z)": r.get("_dues_rcpt", pd.NaT),
                "Effective Month": eff_month,
                "Term Months": total_months,
                "Months Current CY": cur_mo,
                "Months Next (2026)": next_mo,
                "Months Following (2027)": follow_mo,
                "Recognize Current (→ 410x)": amt_cur,
                "Defer 2026 (→ 210x)": amt_next,
                "Defer 2027 (→ 212x)": amt_follow,
                "Rev Account (410x)": REV_DUES_BY_TYPE.get(mt, REV_MEMBERSHIP_DEFAULT),
                "Defer 2026 Acct (210x)": DEFER_210_BY_TYPE.get(mt, DEF_MEMBERSHIP_DEFAULT_NEXT),
                "Defer 2027 Acct (212x)": DEFER_212_BY_TYPE.get(mt, DEF_MEMBERSHIP_DEFAULT_FOLLOW),
            })

    deferral_df = pd.DataFrame(deferral_rows)

    # --- JE Lines (grouped by deposit) ---
    je_rows = []

    # 1) DR Bank per deposit
    for _, row in deposit_summary.reset_index().rename(columns={"_dep_gid":"deposit_gid"}).iterrows():
        dep_gid = int(row["deposit_gid"])
        # Prefer bank post date if available
        dep_date = (
            row["_wd_bank_post_date"]
            if "_wd_bank_post_date" in row and pd.notna(row["_wd_bank_post_date"])
            else row["deposit_date"]
)        wd_net = float(row.get("withdrawal_net", 0) or 0)
        if wd_net != 0:
            je_rows.append({
                "deposit_gid": dep_gid,
                "date": pd.to_datetime(dep_date, errors="coerce").date() if pd.notna(dep_date) else None,
                "line_type": "DEBIT",
                "account": BANK_GL,
                "description": "PayPal Withdrawal (bank deposit)",
                "amount": round(wd_net, 2),
                "source": "Withdrawal",
            })

    # 2) DR Fees by account per deposit (positive)
    if (
        "pp_item_title_col" in locals() and "pp_fee_col" in locals() and
        pp_item_title_col and pp_fee_col and
        (pp_fee_col in pp.columns) and (pp_item_title_col in pp.columns)
    ):
        fee_tx = pp.loc[
            (~pp["_is_withdrawal"]) &
            (pp["_dep_gid"].notna()) &
            (pp[pp_fee_col].notna()) &
            (pp["_child_in_window"])
        ].copy()

        fee_tx["_fee_account"] = fee_tx[pp_item_title_col].apply(choose_fee_account_from_item_title)
        fee_tx["_fee_amt_pos"] = fee_tx[pp_fee_col].abs()

        fee_alloc = (
            fee_tx.groupby(["_dep_gid", "_fee_account"])["_fee_amt_pos"]
            .sum()
            .reset_index()
        )

        for _, r in fee_alloc.iterrows():
            amt = float(r["_fee_amt_pos"] or 0.0)
            if amt == 0:
                continue
            je_rows.append({
                "deposit_gid": int(r["_dep_gid"]),
                "date": None,
                "line_type": "DEBIT",
                "account": r["_fee_account"],
                "description": "PayPal Fees by Item Title",
                "amount": round(amt, 2),
                "source": "PayPal Fees",
            })

    # 3) CREDIT side per deposit using YM allocations (+ deferral + PAC + VAT + discounts)
    def_by_ref = {}
    if not deferral_df.empty:
        for _, r in deferral_df.iterrows():
            key = str(r.get("TransactionID",""))
            def_by_ref[key] = {
                "recognize": float(r.get("Recognize Current (→ 410x)", 0) or 0),
                "defer_210": float(r.get("Defer 2026 (→ 210x)", 0) or 0),
                "defer_212": float(r.get("Defer 2027 (→ 212x)", 0) or 0),
                "rev_acct": r.get("Rev Account (410x)") or REV_MEMBERSHIP_DEFAULT,
                "acct_210": r.get("Defer 2026 Acct (210x)") or DEF_MEMBERSHIP_DEFAULT_NEXT,
                "acct_212": r.get("Defer 2027 Acct (212x)") or DEF_MEMBERSHIP_DEFAULT_FOLLOW,
            }

    if not ppym.empty and ym_alloc_col:
        for dep_gid, grp in ppym.groupby("_dep_gid"):
            dep_gid = int(dep_gid) if not pd.isna(dep_gid) else None

            pac_sum = 0.0
            vat_sum = 0.0
            mem_recognize = 0.0
            mem_defer_210 = 0.0
            mem_defer_212 = 0.0
            mem_rev_acct = None
            mem_acct_210 = None
            mem_acct_212 = None
            other_rev_by_acct = {}

            for _, r in grp.dropna(subset=[ym_alloc_col]).iterrows():
                alloc     = float(r.get(ym_alloc_col, 0) or 0)
                if alloc == 0:
                    continue
                item_desc = str(r.get(ym_item_desc_col, "") or "")
                pay_desc  = str(r.get(ym_pay_desc_col, "") or "")
                gl_code   = str(r.get(ym_gl_code_col, "") or "")
                ref_key   = str(r.get("_ym_ref_key",""))
                alloc_item_desc = str(r.get(ym_alloc_item_desc_col, "") or "")

                if is_discount_text(pay_desc) or is_discount_text(item_desc):
                    continue

                if is_pac_line(item_desc, gl_code, pay_desc):
                    pac_sum += alloc
                    continue

                if (
                    is_vat_text(item_desc) or is_vat_text(pay_desc) or
                    is_vat_text(alloc_item_desc) or ("4314" in str(gl_code).lower())
                ):
                    vat_sum += alloc
                    continue

                if ref_key in def_by_ref:
                    parts = def_by_ref[ref_key]
                    mem_recognize += parts["recognize"]
                    mem_defer_210 += parts["defer_210"]
                    mem_defer_212 += parts["defer_212"]
                    mem_rev_acct = parts["rev_acct"]
                    mem_acct_210 = parts["acct_210"]
                    mem_acct_212 = parts["acct_212"]
                else:
                    acct = str(gl_code).strip()
                    if acct.lower() in {"", "nan", "none"}:
                        # simple fallback: bucket for review
                        acct = "UNMAPPED · Review"
                    other_rev_by_acct[acct] = other_rev_by_acct.get(acct, 0.0) + alloc

            if dep_gid is None:
                continue

            if pac_sum != 0:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": PAC_LIABILITY,
                    "description": "PAC Donations (liability)",
                    "amount": round(pac_sum, 2),
                    "source": "YM Allocations → PAC",
                })

            if vat_sum != 0:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": VAT_OFFSET_INCOME,
                    "description": "Tax/VAT / CC Fee Offset",
                    "amount": round(vat_sum, 2),
                    "source": "YM Allocations",
                })

            if mem_recognize != 0:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": mem_rev_acct or REV_MEMBERSHIP_DEFAULT,
                    "description": "Membership Dues (current-year recognized)",
                    "amount": round(mem_recognize, 2),
                    "source": "Membership Deferral",
                })

            if mem_defer_210 != 0:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": mem_acct_210 or DEF_MEMBERSHIP_DEFAULT_NEXT,
                    "description": "Membership Dues (deferred to 2026)",
                    "amount": round(mem_defer_210, 2),
                    "source": "Membership Deferral",
                })

            if mem_defer_212 != 0:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": mem_acct_212 or DEF_MEMBERSHIP_DEFAULT_FOLLOW,
                    "description": "Membership Dues (deferred to 2027)",
                    "amount": round(mem_defer_212, 2),
                    "source": "Membership Deferral",
                })

            for acct, amt in other_rev_by_acct.items():
                if amt == 0:
                    continue
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": acct,
                    "description": "YM Allocation (non-membership)",
                    "amount": round(amt, 2),
                    "source": "YM Allocations",
                })

    # 4) PAC donations that appear only in PayPal (not in YM) — normalize Item Title (Col O)
    item_title_col = pp_item_title_col
    if pp_txn_col and pp_txn_col in pp.columns:
        pp["_pp_txn_key"] = pp[pp_txn_col].astype(str).str.strip()
    else:
        pp["_pp_txn_key"] = ""

    pac_mask = (
        (~pp["_is_withdrawal"]) &
        (pp["_dep_gid"].notna()) &
        (pp["_child_in_window"])      # <-- changed here
    )
    if matched_txns:
        pac_mask &= (~pp["_pp_txn_key"].isin(list(matched_txns)))


    if item_title_col:
        pp["_pp_item_title_norm"] = (
            pp[item_title_col]
            .astype(str)
            .fillna("")
            .str.replace(r"[+_]", " ", regex=True)
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        pp["_pp_item_title_norm"] = ""

    pac_re = r"(?:\bwapa\s+pac\b|\bpolitical\s+action\b|\bpac\b)"
    pp["_pp_pac_only"] = pac_mask & pp["_pp_item_title_norm"].str.contains(pac_re, regex=True, na=False)
    pac_only = pp.loc[pp["_pp_pac_only"]].copy()

    if not pac_only.empty and (pp_gross_col in pac_only.columns):
        pac_add = (
            pac_only.groupby("_dep_gid")[pp_gross_col]
            .sum()
            .reset_index()
            .rename(columns={pp_gross_col: "pac_amt"})
        )
        for _, r in pac_add.iterrows():
            amt = float(r["pac_amt"] or 0)
            if amt == 0:
                continue
            je_rows.append({
                "deposit_gid": int(r["_dep_gid"]),
                "date": None,
                "line_type": "CREDIT",
                "account": PAC_LIABILITY,
                "description": "PAC Donation (PayPal Item Title)",
                "amount": round(amt, 2),
                "source": "PayPal PAC (Item Title only)",
            })

    with st.expander("PAC-only rows detected from PayPal Item Title"):
        preview_cols = ["_dep_gid", pp_date_col, pp_txn_col, item_title_col, pp_gross_col]
        preview_cols = [c for c in preview_cols if c in pac_only.columns]
        st.dataframe(pac_only[preview_cols].sort_values(["_dep_gid"]).head(500))

    # 5) PayPal-only "Payment for Invoice No. ####" → 99999 placeholder
    inv_mask = (~pp["_is_withdrawal"]) & (pp["_dep_gid"].notna()) & (pp["_child_in_window"])
    if matched_txns:
        inv_mask &= (~pp["_pp_txn_key"].isin(list(matched_txns)))

    if item_title_col:
        pp["_pp_item_title_norm_inv"] = (
            pp[item_title_col]
            .astype(str).fillna("")
            .str.replace(r"[+_]", " ", regex=True)
            .str.replace(r"[^\w\s\.\#]", " ", regex=True)
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    else:
        pp["_pp_item_title_norm_inv"] = ""

    inv_re = r"payment\s+for\s+invoice\s+no\.?\s*[#/]?\s*(\d{5,})"
    pp["_pp_invoice_payment_only"] = inv_mask & pp["_pp_item_title_norm_inv"].str.contains(inv_re, regex=True, na=False)
    inv_only = pp.loc[pp["_pp_invoice_payment_only"]].copy()

    if not inv_only.empty and (pp_gross_col in inv_only.columns):
        inv_only["_invoice_no"] = inv_only["_pp_item_title_norm_inv"].str.extract(inv_re, expand=False)
        inv_add = (
            inv_only.groupby("_dep_gid")[pp_gross_col]
            .sum()
            .reset_index()
            .rename(columns={pp_gross_col: "inv_amt"})
        )
        for _, r in inv_add.iterrows():
            amt = float(r["inv_amt"] or 0)
            if amt == 0:
                continue
            je_rows.append({
                "deposit_gid": int(r["_dep_gid"]),
                "date": None,
                "line_type": "CREDIT",
                "account": UNMAPPED_REVIEW_ACCT,
                "description": "PayPal Invoice Payment (unmatched) — see detail",
                "amount": round(amt, 2),
                "source": "PayPal Invoice Payment (Item Title only)",
            })

    with st.expander("PayPal-only Invoice Payments detected (to 99999)"):
        cols = ["_dep_gid", pp_date_col, pp_txn_col, item_title_col, "_invoice_no", pp_gross_col]
        cols = [c for c in cols if c in inv_only.columns]
        if cols:
            st.dataframe(inv_only[cols].sort_values(["_dep_gid"]).head(500))

    # Collect out-of-period refunds for review (excluded from JE)
    if pp_type_col:
        type_is_refund = pp[pp_type_col].astype(str).str.lower().str.contains("refund|chargeback", na=False)
    else:
        type_is_refund = False
    amt_is_refund = False
    if pp_gross_col in pp.columns:
        amt_is_refund = (pp[pp_gross_col] < 0)
    elif pp_net_col in pp.columns:
        amt_is_refund = (pp[pp_net_col] < 0)
    pp["_is_refund"] = type_is_refund | amt_is_refund

    oop_refunds = pp.loc[
        (~pp["_is_withdrawal"]) &
        (pp["_dep_gid"].notna()) &
        (~pp["_child_in_window"]) &   # <-- replaced here
        (pp["_is_refund"])
    ].copy()


    je_df = pd.DataFrame(je_rows)

    # Optional auto-balance per deposit
    if AUTO_BALANCE and not je_df.empty:
        RECON_VARIANCE_ACCT = "9999 · Recon Variance (Review)"
        for dep_gid, sub in je_df.groupby("deposit_gid"):
            deb = float(sub.loc[sub["line_type"]=="DEBIT","amount"].sum() or 0.0)
            cred = float(sub.loc[sub["line_type"]=="CREDIT","amount"].sum() or 0.0)
            diff = round(deb - cred, 2)
            if abs(diff) >= 0.01:
                je_df.loc[len(je_df)] = {
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": ("CREDIT" if diff > 0 else "DEBIT"),
                    "account": RECON_VARIANCE_ACCT,
                    "description": "Auto balance to zero (review)",
                    "amount": abs(diff),
                    "source": "Auto Balance",
                }

    # Sort for readability
    if not je_df.empty:
        je_df["line_order"] = je_df["line_type"].map({"DEBIT": 0, "CREDIT": 1}).fillna(2)
        je_df = je_df.sort_values(["deposit_gid","line_order","account"]).drop(columns=["line_order"])

    # Presentation split
    je_view = je_df.copy()
    if not je_view.empty:
        je_view["Debit"]  = np.where(je_view["line_type"]=="DEBIT",  je_view["amount"].round(2), np.nan)
        je_view["Credit"] = np.where(je_view["line_type"]=="CREDIT", je_view["amount"].round(2), np.nan)
    else:
        je_view["Debit"] = np.nan
        je_view["Credit"] = np.nan

    je_cols = ["deposit_gid", "date", "account", "description", "Debit", "Credit", "source"]
    je_cols = [c for c in je_cols if c in je_view.columns]
    je_out = je_view[je_cols] if je_cols else je_view

    # Balance check
    if not je_out.empty:
        balance_df = (
            je_out.groupby("deposit_gid")
            .agg(
                Debits = ("Debit",  lambda s: round(float(np.nansum(s)), 2)),
                Credits= ("Credit", lambda s: round(float(np.nansum(s)), 2)),
            )
            .reset_index()
        )
        balance_df["Diff"] = (balance_df["Debits"] - balance_df["Credits"]).round(2)
    else:
        balance_df = pd.DataFrame(columns=["deposit_gid","Debits","Credits","Diff"])

    # Deposit Summary output
    dep_out = deposit_summary.reset_index().rename(columns={"_dep_gid": "deposit_gid"})[[
        "deposit_gid", "deposit_date", "withdrawal_gross", "withdrawal_fee", "withdrawal_net",
        "tx_count", "tx_gross_sum", "tx_fee_sum", "tx_net_sum", "calc_net", "variance_vs_withdrawal"
    ]].rename(columns={
        "deposit_date": "Deposit Date",
        "withdrawal_gross": "Withdrawal Gross",
        "withdrawal_fee": "Withdrawal Fee",
        "withdrawal_net": "Withdrawal Net",
        "tx_count": "# PayPal Txns",
        "tx_gross_sum": "Sum Gross (Txns)",
        "tx_fee_sum": "Sum Fees (Txns)",
        "tx_net_sum": "Sum Net (Txns)",
        "calc_net": "Calc Net (Gross-Fees)",
        "variance_vs_withdrawal": "Variance vs Withdrawal Net"
    })

    # YM Detail (joined)
    _ym_detail = ppym.rename(columns={
        "_pp_txn_key": "TransactionID",
        ym_item_desc_col: "Item Descriptions",
        ym_gl_code_col: "GL Codes",
        ym_alloc_col: "Allocation",
        "_dues_rcpt": "Dues Receipt Date (col Z)",
        "_eff_month": "Effective Receipt Month",
        ym_membership_col: "Membership",
        ym_pay_desc_col: "Payment Description",
        ym_alloc_item_desc_col: "Allocation Item Desc (YTD)",
    })
    _ym_cols = ["TransactionID","Item Descriptions","GL Codes","Allocation","_dep_gid",
                "Dues Receipt Date (col Z)","Effective Receipt Month","Membership","Payment Description","Allocation Item Desc (YTD)"]
    _ym_cols = [c for c in _ym_cols if c in _ym_detail.columns]
    _ym_detail = _ym_detail[_ym_cols].rename(columns={"_dep_gid": "deposit_gid"})

    # --- Excel build with currency formatting on money columns ---
    def moneyish(colname: str) -> bool:
        if not isinstance(colname, str):
            return False
        name = colname.lower()
        hints = [
            "amount", "debit", "credit", "debits", "credits", "diff",
            "gross", "fee", "net", "allocation", "recognize", "defer",
            "sum", "calc", "variance"
        ]
        return any(h in name for h in hints)

    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        # Write sheets
        dep_out.to_excel(writer, sheet_name="Deposit Summary", index=False)
        balance_df.to_excel(writer, sheet_name="JE Balance Check", index=False)
        je_out.to_excel(writer, sheet_name="JE Lines (Grouped by Deposit)", index=False)
        _ym_detail.to_excel(writer, sheet_name="YM Detail (joined)", index=False)
        if not deferral_df.empty:
            deferral_df.to_excel(writer, sheet_name="Deferral Schedule", index=False)
        if not oop_refunds.empty:
            cols = [c for c in ["deposit_gid","_parsed_date", pp_txn_col, pp_item_title_col, pp_src_col, pp_gross_col, pp_fee_col, pp_net_col] if c in oop_refunds.columns]
            oop_refunds.rename(columns={"_parsed_date":"Transaction Date"}).to_excel(writer, sheet_name="Out-of-Period Refunds (Review)", index=False, columns=cols)

        # Apply currency format to "money-like" columns on every sheet
        wb = writer.book
        cur = wb.add_format({"num_format": "$#,##0.00"})

        def format_money_cols(df, sheet_name):
            ws = writer.sheets[sheet_name]
            # set reasonable column widths
            ws.set_column(0, len(df.columns)-1, 16)
            for i, col in enumerate(df.columns):
                if moneyish(str(col)) or (df[col].dtype.kind in {"f","i"} and str(col).lower() not in {"deposit_gid","# paypal txns","term months","months current cy","months next (2026)","months following (2027)"}):
                    ws.set_column(i, i, 16, cur)

        format_money_cols(dep_out, "Deposit Summary")
        format_money_cols(balance_df, "JE Balance Check")
        format_money_cols(je_out, "JE Lines (Grouped by Deposit)")
        format_money_cols(_ym_detail, "YM Detail (joined)")
        if not deferral_df.empty:
            format_money_cols(deferral_df, "Deferral Schedule")
        if not oop_refunds.empty:
            format_money_cols(oop_refunds.rename(columns={"_parsed_date":"Transaction Date"}), "Out-of-Period Refunds (Review)")

    st.success("Reconciliation complete.")
    st.dataframe(balance_df)

    st.download_button(
        label="Download Excel Workbook",
        data=out_buf.getvalue(),
        file_name="WAPA_Recon_JE_Grouped_Deferrals_PAC_VAT.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    with st.expander("Preview: JE Lines (first 200 rows)"):
        st.dataframe(je_out.head(200))

    if not deferral_df.empty:
        with st.expander("Preview: Deferral Schedule (first 200 rows)"):
            st.dataframe(deferral_df.head(200))
