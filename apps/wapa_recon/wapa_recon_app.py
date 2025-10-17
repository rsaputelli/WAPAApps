
# wapa_recon_app_v5c.py
# Streamlit app for WAPA PayPal ↔ YM ↔ Bank reconciliation
# JE Lines grouped by deposit with Deferrals + PAC liability (v5c)
#
# Run:
#   streamlit run wapa_recon_app_v5c.py

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta

st.set_page_config(page_title="WAPA Recon (JE grouped + Deferrals + PAC)", layout="wide")

# ------------------------- Config -------------------------
BANK_GL = "1002 · TD Bank Checking x6455"

# Expense routing for fees by item title (unchanged)
FEE_ACCT_MEMBERSHIP = "5104 · Membership Expenses:5104 · Dues Expense"
FEE_ACCT_CONFERENCE = "5316 · Fall Conference Expenses:5316 · Registration"
FEE_ACCT_OTHER      = "5012 · Administrative Expenses:5012 · Bank Charges/Credit Card Fees"

# Revenue/Deferred defaults (can be refined to 4101–4107 ↔ 2101–2107 if mapping is provided)
REV_MEMBERSHIP_DEFAULT = "4100 · Membership Dues"
DEF_MEMBERSHIP_DEFAULT = "2100 · Deferred Dues"
PAC_LIABILITY          = "2202 · Due to WAPA PAC"
VAT_OFFSET_INCOME      = "4314 · Fall Conference CC Fee Offset"

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

def service_start_midmonth_rule(dues_paid_date: pd.Timestamp) -> pd.Timestamp:
    # <= 15th counts current month; >15 start next month
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
    # bias residue to "current"
    rounded["current"] = round(rounded["current"] + resid, 2)
    return rounded

# ------------------------- UI -------------------------
st.title("WAPA PayPal ↔ YM ↔ Bank — JE grouped by deposit + Deferrals + PAC")

st.markdown("""
Upload CSVs (any column order is OK; headers auto-detected):

- **PayPal**: Type, Gross, Fee, Net, Transaction ID, Item Title
- **YM Export**: Invoice/Reference, Item Description, GL Code, Allocation, Dues Paid Date, Membership
- **Bank (TD)**: Date, Description, Deposit/Credit/Amount

Deferral rule: A YM line is **membership dues** only if **Membership is populated AND Item Description contains "dues"**.  
Recognition: **Current-year portion** is credited to **Revenue**; **Next/Following-year portions** are credited to **Deferred Dues**.  
PAC Donations: credited to **2202 · Due to WAPA PAC**.
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
    transactions = pp.loc[(~pp["_is_withdrawal"]) & (~pp["_dep_gid"].isna())].copy()

    # --- YM columns ---
    ym_ref_col = find_col(ym, ["invoice_-_reference_number", "invoice_reference_number", "reference_number", "invoice"])
    ym_item_desc_col = find_col(ym, ["item_descriptions", "item_description", "item", "item_name", "name"])  # N
    ym_gl_code_col = find_col(ym, ["gl_codes", "gl_code", "gl", "account", "account_code"])                 # O
    ym_alloc_col = find_col(ym, ["allocation", "allocated_amount", "amount", "line_total"])                 # Q
    ym_dues_paid_col = find_col(ym, ["dues_paid_date", "paid_date", "membership_paid_date"])
    ym_membership_col = find_col(ym, ["membership", "membership_type"])  # AC

    if ym_alloc_col and ym[ym_alloc_col].dtype == object:
        ym[ym_alloc_col] = to_float(ym[ym_alloc_col])

    ym["_dues_paid"] = pd.to_datetime(ym[ym_dues_paid_col], errors="coerce") if ym_dues_paid_col else pd.NaT

    # Link PP transactions to YM by TransactionID ↔ Reference
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

    # --- Build Deferral Schedule for membership dues lines ---
    deferral_rows = []
    if not ppym.empty and ym_alloc_col:
        for _, r in ppym.iterrows():
            alloc = r.get(ym_alloc_col, np.nan)
            if pd.isna(alloc) or alloc == 0:
                continue
            dues_paid = r.get("_dues_paid", pd.NaT)
            mem_str = r.get(ym_membership_col, "")
            item_desc = str(r.get(ym_item_desc_col, "") or "")

            is_membership_dues = (isinstance(mem_str, str) and mem_str.strip() != "") and ("dues" in item_desc.lower())
            if not is_membership_dues:
                continue

            start = service_start_midmonth_rule(dues_paid)
            # Term inference: if membership mentions "2-year" or "2year" then 24, else 12.
            term_months = 24 if ("2-year" in str(mem_str).lower() or "2year" in str(mem_str).lower()) else 12
            months_list = month_span(start, term_months)
            buckets = split_by_calendar_year(months_list)
            amounts = allocate_amount_by_buckets(alloc, buckets)

            deferral_rows.append({
                "deposit_gid": r.get("_dep_gid", np.nan),
                "TransactionID": r.get("_pp_txn_key", ""),
                "Membership": mem_str,
                "Item Description": item_desc,
                "Dues Paid Date": dues_paid,
                "Service Start": start,
                "Term Months": term_months,
                "Allocation": alloc,
                "Recognize (Current FY)": amounts["current"],
                "Defer (Next FY)": amounts["next"],
                "Defer (Following FY)": amounts["following"],
                "Revenue Account": REV_MEMBERSHIP_DEFAULT,
                "Deferred Account": DEF_MEMBERSHIP_DEFAULT,
            })

    deferral_df = pd.DataFrame(deferral_rows)

    # Quick PAC detector: in YM Item Description or GL Code text
    def is_pac_line(desc: str, gl_code: str) -> bool:
        d = (desc or "")
        g = (gl_code or "")
        return ("pac" in d.lower()) or ("pac" in g.lower()) or ("2202" in g)

    def is_discount(desc: str) -> bool:
        if desc is None:
            return False
        return "discount" in str(desc).lower()

    def is_vat(desc: str) -> bool:
        if desc is None:
            return False
        return "tax" in str(desc).lower() or "vat" in str(desc).lower()

    # --- JE Lines (grouped by deposit) ---
    je_rows = []

    # 1) DR Bank per deposit
    for _, row in deposit_summary.reset_index().rename(columns={"_dep_gid":"deposit_gid"}).iterrows():
        dep_gid = int(row["deposit_gid"])
        dep_date = row["deposit_date"]
        wd_net = float(row.get("withdrawal_net", 0) or 0)
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

    # 2) DR Fees by account per deposit
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
            })
    for fr in fee_lines:
        if fr["amount"] != 0:
            je_rows.append({
                "deposit_gid": int(fr["deposit_gid"]),
                "date": None,
                "line_type": "DEBIT",
                "account": fr["account"],
                "description": "PayPal Fees by Item Title",
                "amount": round(fr["amount"], 2),
                "source": "PayPal Fees",
            })

    # 3) CREDIT side per deposit using YM allocations, with deferral + PAC + VAT + discounts rules
    # Build a convenience lookup from deferral_df for membership dues portions by TransactionID
    def_by_txn = {}
    if not deferral_df.empty:
        for _, r in deferral_df.iterrows():
            def_by_txn[str(r.get("TransactionID",""))] = {
                "recognize": float(r.get("Recognize (Current FY)", 0) or 0),
                "defer_next": float(r.get("Defer (Next FY)", 0) or 0),
                "defer_follow": float(r.get("Defer (Following FY)", 0) or 0),
                "rev_acct": r.get("Revenue Account") or REV_MEMBERSHIP_DEFAULT,
                "def_acct": r.get("Deferred Account") or DEF_MEMBERSHIP_DEFAULT,
                "deposit_gid": r.get("deposit_gid", np.nan),
            }

    # Aggregate YM allocations (excluding discounts); derive PAC, VAT, membership dues, other revenue
    if not ppym.empty and ym_alloc_col:
        for dep_gid, grp in ppym.groupby("_dep_gid"):
            dep_gid = int(dep_gid) if not pd.isna(dep_gid) else None
            # Sums
            pac_sum = 0.0
            vat_sum = 0.0
            other_rev_by_acct = {}
            mem_recognize = 0.0
            mem_defer = 0.0
            mem_rev_acct = REV_MEMBERSHIP_DEFAULT
            mem_def_acct = DEF_MEMBERSHIP_DEFAULT

            for _, r in grp.dropna(subset=[ym_alloc_col]).iterrows():
                alloc = float(r.get(ym_alloc_col, 0) or 0)
                if alloc == 0:
                    continue
                item_desc = str(r.get(ym_item_desc_col, "") or "")
                gl_code = str(r.get(ym_gl_code_col, "") or "")
                ref_key = str(r.get("_ym_ref_key",""))

                if is_discount(item_desc):
                    continue

                if is_pac_line(item_desc, gl_code):
                    pac_sum += alloc
                    continue

                if is_vat(item_desc):
                    vat_sum += alloc
                    continue

                # Membership dues check via deferral lookup
                if ref_key in def_by_txn:
                    parts = def_by_txn[ref_key]
                    mem_recognize += parts["recognize"]
                    mem_defer += (parts["defer_next"] + parts["defer_follow"])
                    mem_rev_acct = parts["rev_acct"]
                    mem_def_acct = parts["def_acct"]
                else:
                    # Non-membership → credit GL Code as-is (YM revenue mapping)
                    acct = gl_code if gl_code else "UNMAPPED · Review"
                    other_rev_by_acct[acct] = other_rev_by_acct.get(acct, 0.0) + alloc

            # Emit credits for this deposit
            # PAC Liability
            if pac_sum != 0 and dep_gid is not None:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": PAC_LIABILITY,
                    "description": "PAC Donations (liability)",
                    "amount": round(pac_sum, 2),
                    "source": "YM Allocations → PAC",
                })

            # VAT Offset Income
            if vat_sum != 0 and dep_gid is not None:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": VAT_OFFSET_INCOME,
                    "description": "Tax/VAT offset",
                    "amount": round(vat_sum, 2),
                    "source": "YM Allocations → VAT",
                })

            # Membership: immediate revenue (current FY)
            if mem_recognize != 0 and dep_gid is not None:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": mem_rev_acct,
                    "description": "Membership Dues (current FY recognized)",
                    "amount": round(mem_recognize, 2),
                    "source": "Membership Deferral",
                })

            # Membership: deferred portions (next + following FY) → liability
            if mem_defer != 0 and dep_gid is not None:
                je_rows.append({
                    "deposit_gid": dep_gid,
                    "date": None,
                    "line_type": "CREDIT",
                    "account": mem_def_acct,
                    "description": "Membership Dues (deferred to future FY)",
                    "amount": round(mem_defer, 2),
                    "source": "Membership Deferral",
                })

            # Other revenue by YM GL code
            for acct, amt in other_rev_by_acct.items():
                if amt == 0 or dep_gid is None:
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

    je_df = pd.DataFrame(je_rows)

    # Balance check per deposit
    checks = []
    for dep_gid, sub in je_df.groupby("deposit_gid"):
        debits = round(float(sub.loc[sub["line_type"]=="DEBIT","amount"].sum() or 0), 2)
        credits = round(float(sub.loc[sub["line_type"]=="CREDIT","amount"].sum() or 0), 2)
        diff = round(debits - credits, 2)
        checks.append({"deposit_gid": dep_gid, "Debits": debits, "Credits": credits, "Diff": diff})
    balance_df = pd.DataFrame(checks)

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

    # --- Build Excel and download ---
    out_buf = io.BytesIO()
    with pd.ExcelWriter(out_buf, engine="xlsxwriter") as writer:
        dep_out.to_excel(writer, sheet_name="Deposit Summary", index=False)
        balance_df.to_excel(writer, sheet_name="JE Balance Check", index=False)
        je_df.to_excel(writer, sheet_name="JE Lines (Grouped by Deposit)", index=False)

        # Detail tabs for review
        ppym.rename(columns={
            "_pp_txn_key": "TransactionID",
            ym_item_desc_col: "Item Descriptions",
            ym_gl_code_col: "GL Codes",
            ym_alloc_col: "Allocation",
            "_dues_paid": "Dues Paid Date",
            ym_membership_col: "Membership",
        })[[c for c in ["TransactionID","Item Descriptions","GL Codes","Allocation","_dep_gid","Dues Paid Date","Membership"] if c in
             ["TransactionID","Item Descriptions","GL Codes","Allocation","_dep_gid","Dues Paid Date","Membership"]]].rename(columns={"_dep_gid":"deposit_gid"}).to_excel(writer, sheet_name="YM Detail (joined)", index=False)
        if not deferral_df.empty:
            deferral_df.to_excel(writer, sheet_name="Deferral Schedule", index=False)

    st.success("Reconciliation complete.")
    st.dataframe(balance_df)

    st.download_button(
        label="Download Excel Workbook",
        data=out_buf.getvalue(),
        file_name="WAPA_Recon_JE_Grouped_Deferrals_PAC.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    with st.expander("Preview: JE Lines (first 200 rows)"):
        st.dataframe(je_df.head(200))

    if not deferral_df.empty:
        with st.expander("Preview: Deferral Schedule (first 200 rows)"):
            st.dataframe(deferral_df.head(200))
