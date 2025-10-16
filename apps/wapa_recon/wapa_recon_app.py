# wapa_recon_app.py
# Streamlit app for WAPA PayPal↔YM↔Bank reconciliation with fee routing and deferral schedule
# Author: ChatGPT (based on requirements from WAPA)
#
# Features
# - Exact-amount PayPal withdrawal ↔ Bank deposit matching (±7 days, prefers "PAYPAL" description).
# - Fee routing by PayPal Item Title:
#     "Membership+Dues" -> 5104 · Membership Expenses:5104 · Dues Expense
#     "Online+Store+Order" -> 5316 · Fall Conference Expenses:5316 · Registration
#     All others -> 5012 · Administrative Expenses:5012 · Bank Charges/Credit Card Fees
# - Deferral scaffold (FYE=12/31; mid-month rule):
#     Only defer when YM membership column is populated AND YM item description contains "dues".
#     Term inference: if membership mentions "2-year" then 24 months, else 12.
# - Bank debit GL: 1002 · TD Bank Checking x6455
# - Outputs an Excel with tabs: Deposit Summary, JE (fees routed), YM Detail, Deferral Schedule, Unmatched tabs, Bank (all)
#
# Notes
# - This app expects CSV inputs. Column detection is robust/fuzzy, but recommended headers are documented in the UI.
# - After you provide the final membership revenue GL and exact deferred liability GL mapping,
#   we can switch JE credits from YM GLs to revenue/deferred buckets.
#
# Run:
#   streamlit run wapa_recon_app.py

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title="WAPA PayPal ↔ YM ↔ Bank Recon", layout="wide")

# ------------------------- Config -------------------------
FYE_MONTH, FYE_DAY = 12, 31
BANK_GL = "1002 · TD Bank Checking x6455"

FEE_ACCT_MEMBERSHIP = "5104 · Membership Expenses:5104 · Dues Expense"
FEE_ACCT_CONFERENCE = "5316 · Fall Conference Expenses:5316 · Registration"
FEE_ACCT_OTHER      = "5012 · Administrative Expenses:5012 · Bank Charges/Credit Card Fees"

TERM_MAP_BY_MEMBERSHIP_COL_AC = {"2year": 24}

# ------------------------- Helpers -------------------------
def norm_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def read_csv_robust(file) -> pd.DataFrame:
    # file is a file-like object from Streamlit uploader
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
            if cand in col:
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

def service_start(dues_paid_date: pd.Timestamp) -> pd.Timestamp:
    # Mid-month rule: <=15 count the month; >15 start next month
    if pd.isna(dues_paid_date):
        return pd.NaT
    if dues_paid_date.day <= 15:
        return dues_paid_date.replace(day=1)
    y = dues_paid_date.year + (1 if dues_paid_date.month == 12 else 0)
    m = 1 if dues_paid_date.month == 12 else dues_paid_date.month + 1
    return pd.Timestamp(year=y, month=m, day=1)

def month_span(beg: pd.Timestamp, months: int):
    if pd.isna(beg):
        return []
    y, m = beg.year, beg.month
    out = []
    for _ in range(months):
        out.append((y, m))
        if m == 12:
            y += 1
            m = 1
        else:
            m += 1
    return out

def split_by_calendar_year(months_list):
    buckets = {"current": 0, "next": 0, "following": 0}
    if not months_list:
        return buckets
    first_year = months_list[0][0]
    for (y, m) in months_list:
        if y == first_year:
            buckets["current"] += 1
        elif y == first_year + 1:
            buckets["next"] += 1
        else:
            buckets["following"] += 1
    return buckets

def allocate_amount_by_buckets(amount: float, buckets: dict) -> dict:
    total_months = sum(buckets.values())
    if total_months == 0 or pd.isna(amount):
        return {"current": 0.0, "next": 0.0, "following": 0.0}
    shares = {k: (amount * buckets[k] / total_months) for k in buckets}
    rounded = {k: round(v, 2) for k, v in shares.items()}
    resid = round(amount - sum(rounded.values()), 2)
    target = max(rounded, key=lambda k: (rounded[k], k == "current"))
    rounded[target] = round(rounded[target] + resid, 2)
    return rounded

def infer_term_months_from_ac(mem_val: str) -> int:
    s = norm_text(mem_val)
    for key, months in TERM_MAP_BY_MEMBERSHIP_COL_AC.items():
        if key in s:
            return months
    return 12

def infer_member_type(mem_val: str) -> str:
    s = norm_text(mem_val)
    for k in ["member","fellow","affiliate","organization","student","sustaining","hardship"]:
        if k in s:
            return k
    return "member"

# ------------------------- UI -------------------------
st.title("WAPA PayPal ↔ YM ↔ Bank Reconciliation")

st.markdown("""
**Upload CSVs** (any column order is OK; headers are auto-detected):
- **PayPal**: includes Type (E), Gross (H), Fee (I), Net (J), TransactionID (M), Item Title (O)
- **YM Export**: includes Invoice Ref (I), Item Description (N), GL Code (O), Allocation (Q), Dues Paid Date, Membership (AC)
- **Bank (TD)**: includes Date (A), Description (E), Deposit Amount (G) [a.k.a. Debit in your sheet]

> Deferral rule: A YM line is considered **membership dues** only if **Membership (AC) is populated AND Item Description (N) contains "dues"**.
""")

pp_file = st.file_uploader("PayPal CSV", type=["csv"], key="pp")
ym_file = st.file_uploader("YM Export CSV", type=["csv"], key="ym")
bank_file = st.file_uploader("Bank/TD CSV", type=["csv"], key="bank")

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
    pp_type_col = find_col(pp, ["type"])
    pp_gross_col = find_col(pp, ["gross"])
    pp_fee_col = find_col(pp, ["fee", "fees"])
    pp_net_col = find_col(pp, ["net"])
    pp_txn_col = find_col(pp, ["transaction_id", "transactionid", "txn_id"])
    pp_date_col = find_col(pp, ["date", "date_time", "time", "transaction_initiation_date"])
    pp_item_title_col = find_col(pp, ["item_title", "item_name", "title", "product_name"])

    for col in [pp_gross_col, pp_fee_col, pp_net_col]:
        if col and pp[col].dtype == object:
            pp[col] = to_float(pp[col])

    pp["_parsed_date"] = pd.to_datetime(pp[pp_date_col], errors="coerce") if pp_date_col else pd.NaT
    pp["_type_norm"] = pp[pp_type_col].astype(str).str.strip().str.lower() if pp_type_col else ""

    withdrawal_terms = ["general withdrawal", "general witdrawal", "withdrawal", "bank deposit"]
    pp["_is_withdrawal"] = pp["_type_norm"].apply(lambda x: any(term in x for term in withdrawal_terms))

    # Normalize PayPal withdrawal signs to positive
    for col in [pp_gross_col, pp_fee_col, pp_net_col]:
        if col in pp.columns:
            mask = (pp["_is_withdrawal"]) & (pp[col] < 0)
            pp.loc[mask, col] = -pp.loc[mask, col]

    # Grouping by withdrawals
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
    transactions = pp.loc[(~pp["_is_withdrawal"]) & (~pp["_dep_gid"].isna())].copy()

    # --- YM columns ---
    ym_ref_col = find_col(ym, ["invoice_-_reference_number", "invoice_reference_number", "reference_number", "invoice"])
    ym_item_desc_col = find_col(ym, ["item_descriptions", "item_description", "item", "item_name", "name"])  # Col N
    ym_gl_code_col = find_col(ym, ["gl_codes", "gl_code", "gl", "account", "account_code"])  # Col O
    ym_alloc_col = find_col(ym, ["allocation", "allocated_amount", "amount", "line_total"])  # Col Q
    ym_dues_paid_col = find_col(ym, ["dues_paid_date", "paid_date", "membership_paid_date"])
    ym_membership_col = find_col(ym, ["membership", "membership_type"])  # Col AC

    if ym_alloc_col and ym[ym_alloc_col].dtype == object:
        ym[ym_alloc_col] = to_float(ym[ym_alloc_col])

    ym["_dues_paid"] = pd.to_datetime(ym[ym_dues_paid_col], errors="coerce") if ym_dues_paid_col else pd.NaT

    transactions["_pp_txn_key"] = transactions[pp_txn_col].astype(str).str.strip() if pp_txn_col else ""
    ym["_ym_ref_key"] = ym[ym_ref_col].astype(str).str.strip() if ym_ref_col else ""

    ppym = transactions.merge(
        ym[[c for c in [ym_ref_col, "_ym_ref_key", ym_item_desc_col, ym_gl_code_col, ym_alloc_col, "_dues_paid", ym_membership_col] if c]],
        left_on="_pp_txn_key",
        right_on="_ym_ref_key",
        how="left",
        suffixes=("", "_ym"),
    )

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

    deposit_summary = wd.join(tx_sums, how="left")
    deposit_summary["calc_net"] = deposit_summary["tx_gross_sum"].fillna(0) - deposit_summary["tx_fee_sum"].fillna(0)
    deposit_summary["variance_vs_withdrawal"] = deposit_summary["withdrawal_net"].fillna(0) - deposit_summary["calc_net"]

    # --- Bank matching (EXACT cents; ±7 days) ---
    bank_date_col = find_col(bank, ["date_posted", "date"])
    bank_desc_col = find_col(bank, ["description", "memo", "details", "desc"])

    cand_amt_cols = []
    for name in ["deposit", "credit", "amount", "debit"]:
        c = find_col(bank, [name])
        if c and c not in cand_amt_cols:
            cand_amt_cols.append(c)

    for c in cand_amt_cols:
        if bank[c].dtype == object:
            bank[c] = to_float(bank[c])

    if bank_date_col is None:
        bank_date_col = bank.columns[0]
    bank["_date"] = pd.to_datetime(bank[bank_date_col], errors="coerce")
    bank["_desc"] = bank[bank_desc_col].astype(str) if bank_desc_col else ""

    bank["_deposit_amt"] = np.nan
    if "credit" in cand_amt_cols:
        bank["_deposit_amt"] = bank["credit"]
    if bank["_deposit_amt"].isna().all() and "deposit" in cand_amt_cols:
        bank["_deposit_amt"] = bank["deposit"]
    if bank["_deposit_amt"].isna().all() and "amount" in cand_amt_cols:
        bank["_deposit_amt"] = bank["amount"].where(bank["amount"] > 0)
    if bank["_deposit_amt"].isna().all() and "debit" in cand_amt_cols:
        bank["_deposit_amt"] = bank["debit"]

    bank["_deposit_amt_abs"] = bank["_deposit_amt"].abs()
    bank["_deposit_cents"] = np.rint(bank["_deposit_amt_abs"] * 100).astype("Int64")
    bank["_is_paypal_desc"] = bank["_desc"].str.upper().str.contains("PAYPAL", na=False)
    bank = bank.sort_values(by=["_date"]).reset_index(drop=True)
    bank["_matched"] = False

    withdrawals_df = deposit_summary.reset_index().rename(columns={"_dep_gid": "deposit_gid"})
    withdrawals_df["Deposit Date"] = pd.to_datetime(withdrawals_df["deposit_date"], errors="coerce")
    withdrawals_df["Withdrawal Net"] = withdrawals_df["withdrawal_net"].astype(float)
    withdrawals_df["_withdrawal_cents"] = np.rint(withdrawals_df["Withdrawal Net"].round(2) * 100).astype("Int64")

    match_rows = []
    for _, wrow in withdrawals_df.iterrows():
        pay_date = wrow["Deposit Date"]
        pay_cents = wrow["_withdrawal_cents"]
        if pd.isna(pay_date) or pd.isna(pay_cents):
            match_rows.append({
                "deposit_gid": wrow["deposit_gid"],
                "Matched Bank Date": None,
                "Matched Bank Amount": None,
                "Bank Description": "NO MATCH (missing date/amount)"
            })
            continue

        window_mask = bank["_date"].between(pay_date - timedelta(days=7), pay_date + timedelta(days=7))
        amt_mask = bank["_deposit_cents"] == pay_cents  # exact cents
        candidates = bank.loc[window_mask & amt_mask & (~bank["_matched"])].copy()

        if candidates.empty:
            candidates = bank.loc[amt_mask & (~bank["_matched"])].copy()

        if not candidates.empty:
            candidates["has_paypal"] = candidates["_is_paypal_desc"].astype(int)
            candidates["date_dist"] = (candidates["_date"] - pay_date).abs()
            candidates = candidates.sort_values(by=["has_paypal", "date_dist"], ascending=[False, True])
            best = candidates.iloc[0]
            bank_idx = best.name
            bank.loc[bank_idx, "_matched"] = True

            match_rows.append({
                "deposit_gid": wrow["deposit_gid"],
                "Matched Bank Date": best["_date"],
                "Matched Bank Amount": float(best["_deposit_amt"]) if not pd.isna(best["_deposit_amt"]) else float(best["_deposit_amt_abs"]),
                "Bank Description": str(best["_desc"]),
            })
        else:
            match_rows.append({
                "deposit_gid": wrow["deposit_gid"],
                "Matched Bank Date": None,
                "Matched Bank Amount": None,
                "Bank Description": "NO MATCH FOUND"
            })

    matches_df = pd.DataFrame(match_rows)
    deposit_with_bank = withdrawals_df.merge(matches_df, on="deposit_gid", how="left")

    # --- PayPal fee routing ---
    fee_lines = []
    if pp_item_title_col and pp_fee_col:
        fee_tx = pp.loc[(~pp["_is_withdrawal"]) & (pp["_dep_gid"].notna()) & (pp[pp_fee_col].notna())].copy()
        fee_tx["_fee_account"] = fee_tx[pp_item_title_col].apply(choose_fee_account_from_item_title)
        fee_alloc = fee_tx.groupby(["_dep_gid", "_fee_account"])[pp_fee_col].sum().reset_index()
        for _, r in fee_alloc.iterrows():
            fee_lines.append({
                "deposit_gid": int(r["_dep_gid"]),
                "account": r["_fee_account"],
                "amount": float(r[pp_fee_col]),
                "source": "PayPal Fees by Item Title"
            })
    fee_df = pd.DataFrame(fee_lines)

    # --- Deferral schedule: ONLY when Membership col present AND Item Description contains "dues" ---
    deferral_rows = []
    if not ppym.empty and ym_alloc_col:
        for _, r in ppym.iterrows():
            alloc = r.get(ym_alloc_col, np.nan)
            if pd.isna(alloc) or alloc == 0:
                continue
            dues_paid = r.get("_dues_paid", pd.NaT)
            mem_str = r.get(ym_membership_col, "")
            item_desc = str(r.get(ym_item_desc_col, "") or "")

            # new rule: both membership present AND item description indicates dues
            is_membership_dues = (isinstance(mem_str, str) and mem_str.strip() != "") and ("dues" in item_desc.lower())

            if not is_membership_dues:
                continue

            start = service_start(dues_paid)
            term_months = 24 if "2-year" in str(mem_str).lower() or "2year" in str(mem_str).lower() else 12
            months_list = month_span(start, term_months)
            buckets = split_by_calendar_year(months_list)
            amounts = allocate_amount_by_buckets(alloc, buckets)

            mem_type = infer_member_type(mem_str)
            next_year = start.year + 1 if start is not pd.NaT else np.nan
            following_year = start.year + 2 if start is not pd.NaT else np.nan

            deferral_rows.append({
                "deposit_gid": r.get("_dep_gid", np.nan),
                "TransactionID": r.get("_pp_txn_key", ""),
                "Membership": mem_str,
                "Item Description": item_desc,
                "Member Type": mem_type,
                "Dues Paid Date": dues_paid,
                "Service Start": start,
                "Term Months": term_months,
                "Allocation": alloc,
                "Months Current FY": buckets["current"],
                "Months Next FY": buckets["next"],
                "Months Following FY": buckets["following"],
                "Amount Current FY (recognize)": amounts["current"],
                "Amount Next FY (defer)": amounts["next"],
                "Amount Following FY (defer)": amounts["following"],
                "Next FY (calendar year)": next_year,
                "Following FY (calendar year)": following_year,
            })

    deferral_df = pd.DataFrame(deferral_rows)

    # --- JE Lines ---
    je_rows = []
    # Bank debit per deposit
    for _, row in deposit_summary.reset_index().rename(columns={"_dep_gid":"deposit_gid"}).iterrows():
        dep_gid = int(row["deposit_gid"])
        dep_date = row["deposit_date"]
        wd_net = float(row.get("withdrawal_net", 0) or 0)
        je_rows.append({
            "deposit_gid": dep_gid,
            "date": pd.to_datetime(dep_date, errors="coerce").date() if pd.notna(dep_date) else None,
            "line_type": "DEBIT",
            "account": BANK_GL,
            "description": "PayPal Withdrawal (bank deposit)",
            "amount": wd_net,
            "source": "Withdrawal",
        })

    # Fee debits per mapped account per deposit
    for _, fr in fee_df.iterrows():
        je_rows.append({
            "deposit_gid": int(fr["deposit_gid"]),
            "date": None,
            "line_type": "DEBIT",
            "account": fr["account"],
            "description": "PayPal Fees by Item Title (not deferred)",
            "amount": float(fr["amount"]),
            "source": "PayPal Fees",
        })

    # Credits from YM allocations (no deferral swap yet)
    if not ppym.empty and ym_alloc_col:
        credits = (
            ppym.dropna(subset=[ym_alloc_col])
            .groupby(["_dep_gid", ym_gl_code_col], dropna=False)[ym_alloc_col]
            .sum()
            .reset_index()
            .rename(columns={ym_gl_code_col: "GL Code", ym_alloc_col: "Amount"})
        )
        for _, cr in credits.iterrows():
            je_rows.append({
                "deposit_gid": int(cr["_dep_gid"]),
                "date": None,
                "line_type": "CREDIT",
                "account": str(cr["GL Code"]),
                "description": "YM Allocation (no deferral swap yet)",
                "amount": float(cr["Amount"]),
                "source": "YM Allocations",
            })

    je_df = pd.DataFrame(je_rows)

    # --- Unmatched tabs ---
    unmatched_paypal = transactions.loc[~transactions["_pp_txn_key"].isin(ym["_ym_ref_key"] if "_ym_ref_key" in ym.columns else [])].copy()
    unmatched_paypal = unmatched_paypal[[pp_date_col, pp_txn_col, pp_gross_col, pp_fee_col, pp_net_col, "_dep_gid"]].rename(columns={
        pp_date_col: "date",
        pp_txn_col: "TransactionID",
        pp_gross_col: "gross",
        pp_fee_col: "fee",
        pp_net_col: "net",
        "_dep_gid": "deposit_gid"
    })

    unmatched_ym = ym.loc[~ym["_ym_ref_key"].isin(transactions["_pp_txn_key"])].copy() if "_ym_ref_key" in ym.columns else pd.DataFrame()
    if not unmatched_ym.empty:
        subset_cols = [c for c in [ym_ref_col, ym_item_desc_col, ym_gl_code_col, ym_alloc_col, ym_dues_paid_col, ym_membership_col] if c in unmatched_ym.columns]
        unmatched_ym = unmatched_ym[subset_cols]
        unmatched_ym = unmatched_ym.rename(columns={
            ym_ref_col: "Invoice - Reference Number",
            ym_item_desc_col: "Item Descriptions",
            ym_gl_code_col: "GL Codes",
            ym_alloc_col: "Allocation",
            ym_dues_paid_col: "Dues Paid Date" if ym_dues_paid_col else "Dues Paid Date (missing)",
            ym_membership_col: "Membership" if ym_membership_col else "Membership (missing)",
        })

    unmatched_bank = bank.loc[(bank["_deposit_cents"].notna()) & (bank["_matched"] == False)][["_date", "_desc", "_deposit_amt"]].rename(columns={
        "_date": "Bank Date",
        "_desc": "Bank Description",
        "_deposit_amt": "Bank Amount"
    })

    # --- Deposit Summary output ---
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

    dep_out = dep_out.merge(
        deposit_with_bank[["deposit_gid", "Matched Bank Date", "Matched Bank Amount", "Bank Description"]],
        on="deposit_gid",
        how="left"
    )

    # --- Build Excel and download ---
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        dep_out.to_excel(writer, sheet_name="Deposit Summary - Bank", index=False)
        je_df.to_excel(writer, sheet_name="JE Lines - Fees Routed", index=False)
        ppym.rename(columns={
            "_pp_txn_key": "TransactionID",
            ym_item_desc_col: "Item Descriptions",
            ym_gl_code_col: "GL Codes",
            ym_alloc_col: "Allocation",
            "_dues_paid": "Dues Paid Date",
            ym_membership_col: "Membership",
        })[[c for c in ["TransactionID","Item Descriptions","GL Codes","Allocation","_dep_gid","Dues Paid Date","Membership"] if c in
             ["TransactionID","Item Descriptions","GL Codes","Allocation","_dep_gid","Dues Paid Date","Membership"]]].rename(columns={"_dep_gid":"deposit_gid"}).to_excel(writer, sheet_name="YM Detail dues+member", index=False)
        if not deferral_df.empty:
            deferral_df.to_excel(writer, sheet_name="Deferral Schedule", index=False)
        unmatched_paypal.to_excel(writer, sheet_name="Unmatched - PayPal", index=False)
        if not unmatched_ym.empty:
            unmatched_ym.to_excel(writer, sheet_name="Unmatched - YM", index=False)
        bank.to_excel(writer, sheet_name="Bank Transactions all", index=False)
        unmatched_bank.to_excel(writer, sheet_name="Unmatched - Bank", index=False)

    st.success("Reconciliation complete.")
    st.dataframe(dep_out.head(50))

    st.download_button(
        label="Download Excel Workbook",
        data=out_buf.getvalue(),
        file_name="WAPA_Recon_WITH_DeferralScaffold.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    with st.expander("Preview: JE Lines (first 200 rows)"):
        st.dataframe(je_df.head(200))

    with st.expander("Preview: Deferral Schedule (first 200 rows)"):
        st.dataframe(deferral_df.head(200))
