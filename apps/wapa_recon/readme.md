WAPA Reconciliation – Output Workbook Guide

This workbook is produced by the Streamlit app and summarizes PayPal withdrawals against YM (YourMembership) detail, with journal entries and checks. Tabs that may be optional (written only when data exists) are noted.

1) Deposit Summary

Per–withdrawal roll-up that ties PayPal withdrawals to the underlying transactions.

Columns

deposit_gid — Internal group ID for the withdrawal batch.

PayPal Withdrawal Date — The date of the PayPal withdrawal/payout.

Bank Deposit Date — Date the deposit posted to TD Bank (used for bank-match timing).

Withdrawal Gross — Total gross of the withdrawal per PayPal.

Withdrawal Fee — Total fees deducted in the withdrawal per PayPal.

Withdrawal Net — Net amount received after fees per PayPal.

# PayPal Txns — Count of individual PayPal transactions in this withdrawal.

Sum Gross (Txns) — Sum of gross across all member transactions mapped to this withdrawal.

Sum Fees (Txns) — Sum of fees across those transactions.

Sum Net (Txns) — Sum of net across those transactions.

Calc Net (Gross-Fees) — Recomputed Sum Gross (Txns) - Sum Fees (Txns) for a quick math check.

Variance vs Withdrawal Net — Calc Net – Withdrawal Net.

Use this to spot mismatches (e.g., out-of-period refunds, missing items, timing differences).

Typical uses

Confirm each withdrawal’s net matches the recomputed transaction-level net.

Choose which bank date to recognize for the JE (usually Bank Deposit Date).

2) JE Lines (Grouped by Deposit)

Journal lines grouped by each deposit. Designed to drop straight into your JE tool.

Columns

deposit_gid — Links JE lines back to the Deposit Summary row.

date — Posting date for the line (project standard; typically the bank deposit date).

account — Account name/number (e.g., 1002 · TD Bank Checking x6455, revenue, fees, PAC liability).

description — Line memo/description (e.g., “PayPal Fees by Item Title”, “YM Allocation”).

Debit — Debit amount (blank if credit).

Credit — Credit amount (blank if debit).

source — Origin or classification hint (e.g., PayPal Fees, YM Allocations, Bank Deposit).

Notes

The first line per group is typically DR Bank for the withdrawal net.

Fees are posted to the designated expense account.

PAC donations (when present) credit the liability account you specified (2202 · Due to WAPA PAC).

3) Consolidated JE (Single Entry)

A single consolidated multi-line entry that totals accounts across all deposits in the run.

Columns

account — Account name/number.

Debit — Total debits for that account across the run.

Credit — Total credits for that account across the run.

Memo — Fixed text: “Consolidated JE — single multi-line entry”.

Notes

Use this if you prefer one JE for the period instead of per-deposit entries.

4) JE Balance Check

Quick proof that JE debits equal credits at the deposit_gid level.

Columns

deposit_gid — Deposit group ID.

Debits — Sum of debits for that group.

Credits — Sum of credits for that group.

Diff — Debits - Credits (should be 0.00 for a balanced group).

Tips

Filter for non-zero Diff to find problems fast.

5) YM Detail (joined)

Transaction-level YM detail aligned to PayPal transactions (post-join). Helpful for audit and GL mapping.

Columns

deposit_gid — Withdrawal group ID the transaction landed in.

TransactionID — The joined key linking PayPal to YM (often the invoice/reference).

Item Descriptions — YM line/item description (e.g., “Fellow: 2 Year”, “Fall Conference Reg”).

GL Codes — YM GL code per line (used for mapping revenue/liability).

Allocation — Dollar allocation to this line from YM (YTD or current as applicable).

Dues Receipt Date (col Z) — The dues receipt date from YM export (used for deferral logic).

Effective Receipt Month — Month the receipt is considered effective (e.g., 15th cutoff rules).

Payment Description — YM payment description string (often mirrors item or categorization).

Allocation Item Desc (YTD) — YM’s YTD allocation label when provided.

Category — YM category label (e.g., Membership, Store, Donation/PAC).

Notes

This is the canonical place to verify membership vs non-membership, 2-year terms, etc.

Deferral logic inspects Membership/Category, Item Descriptions, and Dues Receipt Date.

6) Deferral Schedule (optional; written when deferrable memberships exist)

Calculated allocation of current vs future periods for multi-month/2-year memberships.

Columns

deposit_gid — Deposit the membership payment rolled into.

TransactionID — Transaction key (from YM).

Membership — Normalized membership type.

Item Description — Original YM item label for the membership.

Payment Description — YM payment descriptor.

Receipt Date (col Z) — YM dues receipt date.

Effective Month — Month from which recognition starts.

Term Months — Total membership term length in months (e.g., 24 for 2-year).

Months Current CY — Months recognized in current fiscal year.

Months Next (2026) — Months to defer into the next fiscal year (label shows the target year).

Months Following (2027) — Months to defer into the following fiscal year (label shows the target year).

Recognize Current (→ 410x) — Amount recognized to dues revenue now.

Defer 2026 (→ 212x) — Amount to the next-year deferral liability.

Defer 2027 (→ 210x) — Amount to the following-year deferral liability.

Rev Account (410x) — Revenue account resolved from GL/code + membership type.

Defer 2026 Acct (212x) — Liability account for next-year deferral.

Defer 2027 Acct (210x) — Liability account for following-year deferral.

7) Refunds (optional)

In-period refunds extracted from PayPal detail and/or YM — for manual review and JE adjustments if needed.

Typical columns

deposit_gid — Deposit group the refund was associated with (if any).

Transaction Date — Parsed transaction date for the refund.

TransactionID — PayPal/YM reference.

Item Title / Item Descriptions — Refunded item label.

Source — Data source (PayPal/YM).

Gross / Fee / Net — Refund amounts by component.

8) Out-of-Period Refunds (Review) (optional)

Refunds that fall outside the current reporting window. Useful to explain variances.

Columns

Same structure as Refunds, but limited to out-of-period items.

Transaction Date will clearly land outside the run’s date range.

How the tabs relate

YM Detail (joined) → feeds Deposit Summary (roll-up) and provides the allocation/GL context.

Deposit Summary → drives JE Lines (Grouped by Deposit) (DR Bank, CR revenue/fees/PAC).

JE Balance Check → validates each deposit’s JE lines balance to zero.

Consolidated JE (Single Entry) → alternative aggregated posting option.

Deferral Schedule → posts recognition vs liabilities for multi-period memberships.

Refunds / Out-of-Period Refunds → explain variances and prevent silent mismatches.
