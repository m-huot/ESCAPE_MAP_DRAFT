#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Set

import pandas as pd


# --- Regex: keep only Spike substitutions like "S:N331T" or "S:R158G"
RE_SPIKE_SUB = re.compile(r"^S:([A-Z\*])(\d+)([A-Z\*])$")


def parse_spike_substitutions(cell: Optional[str]) -> Set[str]:
    """
    Parse a cell like "S:N331T,S:R158G,ORF1a:T100I" and return a set
    of Spike substitution tokens WITHOUT the 'S:' prefix, e.g. {'N331T','R158G'}.
    We return a set to avoid counting duplicates from the same row more than once.
    """
    if not isinstance(cell, str) or not cell.strip():
        return set()

    # Split on commas or whitespace
    tokens = [t.strip() for t in re.split(r"[,\s]+", cell) if t.strip()]
    muts = set()
    for tok in tokens:
        m = RE_SPIKE_SUB.match(tok)
        if not m:
            continue
        from_aa, pos, to_aa = m.group(1), m.group(2), m.group(3)
        muts.add(f"{from_aa}{pos}{to_aa}")
    return muts


def update_first_date(
    current_first: Optional[pd.Timestamp], candidate: Optional[pd.Timestamp]
) -> Optional[pd.Timestamp]:
    """
    Keep the earliest (minimum) valid date. If either is None/NaT, return the other.
    """
    if pd.isna(current_first) or current_first is None:
        return candidate
    if pd.isna(candidate) or candidate is None:
        return current_first
    return min(current_first, candidate)


def build_mutation_stats(
    df: pd.DataFrame, date_col: str = "date", subs_col: str = "aaSubstitutions"
) -> Dict[str, Tuple[int, Optional[str]]]:
    """
    Build dictionary: mutation -> (count, first_date_iso)
    - count: number of rows (variants) in which the mutation appears at least once
    - first_date_iso: earliest ISO date string (YYYY-MM-DD) seen for that mutation, or None
    """
    # Ensure columns exist
    for col in (date_col, subs_col):
        if col not in df.columns:
            df[col] = pd.NA

    # Normalize date column to pandas Timestamps (coerce invalid to NaT)
    dates = pd.to_datetime(df[date_col], errors="coerce")

    counts: Dict[str, int] = {}
    first_dates: Dict[str, Optional[pd.Timestamp]] = {}

    print(f"Processing {len(df)} rows...")

    for i, row in df.iterrows():
        muts = parse_spike_substitutions(row.get(subs_col))
        if not muts:
            continue

        row_date = dates.iat[i]  # Timestamp or NaT
        for mut in muts:
            counts[mut] = counts.get(mut, 0) + 1
            first_dates[mut] = update_first_date(first_dates.get(mut), row_date)

    # Convert to final dict with ISO date strings (or None)
    result: Dict[str, Tuple[int, Optional[str]]] = {}
    for mut, cnt in counts.items():
        ts = first_dates.get(mut)
        date_str = None
        if ts is not None and not pd.isna(ts):
            # Keep only date part in ISO format
            date_str = ts.date().isoformat()
        result[mut] = (cnt, date_str)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate Spike substitutions from a TSV and report counts and first-seen dates."
    )
    parser.add_argument(
        "--tsv",
        default="ns_metadata.tsv",
        help="Input TSV file (must contain 'aaSubstitutions' column).",
    )
    parser.add_argument(
        "--date-col", default="date", help="Name of the date column (default: date)."
    )
    parser.add_argument(
        "--subs-col",
        default="aaSubstitutions",
        help="Name of the substitutions column (default: aaSubstitutions).",
    )
    parser.add_argument(
        "--out-csv",
        default="extracted_mutations_summary.csv",
        help="Optional path to write a CSV summary (mutation,count,first_date).",
    )
    args = parser.parse_args()

    # Read TSV as strings; let pandas handle NA tokens
    df = pd.read_csv(
        args.tsv, sep="\t", dtype=str, keep_default_na=True, na_values=["", "NA", "NaN"]
    )

    stats = build_mutation_stats(df, date_col=args.date_col, subs_col=args.subs_col)

    # Pretty print (sorted by position numerically when possible)
    def sort_key(m):
        m_re = re.match(r"^[A-Z\*](\d+)[A-Z\*]$", m)
        return (int(m_re.group(1)) if m_re else 10**9, m)

    print("mutation\tcount\tfirst_date")
    for mut in sorted(stats.keys(), key=sort_key):
        count, first_date = stats[mut]
        print(f"{mut}\t{count}\t{first_date if first_date is not None else ''}")

    # Optional CSV output
    if args.out_csv:
        out_df = pd.DataFrame(
            [(m, stats[m][0], stats[m][1]) for m in sorted(stats.keys(), key=sort_key)],
            columns=["mutation", "count", "first_date"],
        )
        out_df.to_csv(args.out_csv, index=False)
        print(f"\nWrote summary to {args.out_csv.resolve()}")


if __name__ == "__main__":
    main()
