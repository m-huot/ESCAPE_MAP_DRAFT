import argparse
import re
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple

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


def find_mutations_hitting_threshold(
    df_period: pd.DataFrame, subs_col: str, threshold: int
) -> Set[str]:
    """
    Analyzes a DataFrame for a single period (assumed to be date-sorted).
    Returns a set of all mutations that reached the count threshold
    *within* this period. Counting starts from 0 for each mutation.
    """
    # counts_this_period: tracks running count for each mutation in this period
    counts_this_period: Dict[str, int] = {}

    # mutations_found: tracks mutations that have *already hit* the threshold
    # This avoids adding them to the result list multiple times for this period.
    mutations_found: Set[str] = set()

    # Iterate through the DataFrame's rows (which should be date-sorted)
    for i, row in df_period.iterrows():
        muts = parse_spike_substitutions(row.get(subs_col))
        if not muts:
            continue

        for mut in muts:
            # If we've already found and logged this mutation for this period,
            # we can stop counting it.
            if mut in mutations_found:
                continue

            # Increment count for this period
            current_count = counts_this_period.get(mut, 0) + 1
            counts_this_period[mut] = current_count

            # Check if it hit the threshold
            if current_count >= threshold:
                mutations_found.add(mut)
                # We don't need to count it further in this period.

    return mutations_found


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find Spike substitutions that reach a specified count threshold "
            "within 3-month periods."
        )
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
        default="mutations_by_period.csv",
        help="Output CSV file (columns: period, mutation).",
    )
    parser.add_argument(
        "--start-date",
        default="2019-10-01",
        help="The start date for the first 3-month period (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="The count threshold to report a mutation (default: 100).",
    )
    args = parser.parse_args()

    # --- 1. Read and Prep Data ---
    print(f"Reading data from {args.tsv}...")
    try:
        df = pd.read_csv(
            args.tsv,
            sep="\t",
            dtype=str,
            keep_default_na=True,
            na_values=["", "NA", "NaN"],
        )
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.tsv}")
        return
    except Exception as e:
        print(f"Error reading TSV: {e}")
        return

    # --- 2. Validate Columns and Clean Dates ---
    if args.date_col not in df.columns:
        print(f"Error: Date column '{args.date_col}' not found in TSV.")
        return
    if args.subs_col not in df.columns:
        print(f"Error: Substitutions column '{args.subs_col}' not found in TSV.")
        return

    # Convert date column, coercing errors to NaT (Not a Time)
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")

    # Drop rows where the date could not be parsed
    df = df.dropna(subset=[args.date_col])

    # Sort by date. This is crucial for correctly finding when a threshold is met.
    df = df.sort_values(by=args.date_col)

    if df.empty:
        print("No valid data found after cleaning dates. Exiting.")
        return

    # --- 3. Define Time Periods ---
    try:
        start_date = pd.Timestamp(args.start_date)
    except ValueError:
        print(
            f"Error: Invalid --start-date '{args.start_date}'. Use YYYY-MM-DD format."
        )
        return

    end_date = df[args.date_col].max()

    if start_date > end_date:
        print(
            f"Start date {start_date.date()} is after the latest data point {end_date.date()}. No periods to analyze."
        )
        return

    # Generate period start dates (e.g., 2019-10-01, 2020-01-01, 2020-04-01, ...)
    # '3MS' means 3-Month-Start frequency
    period_starts = pd.date_range(start=start_date, end=end_date, freq="3MS")

    if period_starts.empty:
        print(f"No full 3-month periods found starting from {start_date.date()}.")
        return

    print(
        f"Analyzing {len(period_starts)} 3-month periods from {period_starts[0].date()} to {period_starts[-1].date()}..."
    )

    # --- 4. Process Each Period ---
    # This list will store tuples of (period_start_str, mutation)
    all_results: List[Tuple[str, str]] = []

    for period_start_date in period_starts:
        # Define the 3-month window [start, end)
        period_end_date = period_start_date + pd.DateOffset(months=3)
        period_str = period_start_date.date().isoformat()

        # Filter the DataFrame for data *within* this period
        df_period = df[
            (df[args.date_col] >= period_start_date)
            & (df[args.date_col] < period_end_date)
        ]

        if df_period.empty:
            continue

        # Find mutations hitting the threshold IN THIS PERIOD
        muts_hit_threshold = find_mutations_hitting_threshold(
            df_period, args.subs_col, args.threshold
        )

        # Add all found mutations to our main results list
        for mut in sorted(list(muts_hit_threshold)):  # Sort alphabetically
            all_results.append((period_str, mut))

    print(
        f"Found {len(all_results)} mutation-period pairs hitting threshold {args.threshold}."
    )

    # --- 5. Write Final Output CSV ---
    if not all_results:
        print("No mutations reached the threshold in any period.")
        # Still write an empty CSV with headers
        out_df = pd.DataFrame(columns=["period", "mutation"])
    else:
        # Create DataFrame from the list of tuples
        out_df = pd.DataFrame(all_results, columns=["period", "mutation"])

    try:
        out_df.to_csv(args.out_csv, index=False)
        print(f"\nSuccessfully wrote summary to {Path(args.out_csv).resolve()}")
    except Exception as e:
        print(f"\nError writing output CSV to {args.out_csv}: {e}")


if __name__ == "__main__":
    main()
