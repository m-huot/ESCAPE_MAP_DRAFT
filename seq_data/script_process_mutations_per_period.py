import pandas as pd
import itertools
from pathlib import Path

""" Process mutations per period file to create a full matrix of all possible mutations"""
# --- Configuration ---

WT_SEQ = "SVYAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCG"

# 2. This is the amino acid alphabet, in the order you specified.
AA = "ACDEFGHIKLMNPQRSTVWY"

# 3. This is the numbering offset.
#    Site 1 in your sequence (index 0) corresponds to real-life site 349.
#    So, the offset is 349 - 1 = 348.
SITE_OFFSET = 348

# 4. Input/Output file paths
INPUT_CSV = "mutations_by_period.csv"
OUTPUT_CSV = "all_mutations_by_period_status.csv"


def generate_all_mutations(wt_seq: str, aa_alphabet: str, offset: int) -> list[str]:
    """
    Generates a list of all possible single mutations in the
    specified order: by position, then by amino acid.
    """

    all_mutations = []
    # Enumerate from 1 (1-based indexing for residue numbers)
    for i, wt_aa in enumerate(wt_seq, start=1):
        if wt_aa not in aa_alphabet:
            continue  # Skip non-canonical amino acids

        # Calculate the "real" site number
        real_site_num = i + offset

        for alt_aa in aa_alphabet:
            if alt_aa == wt_aa:
                continue  # Skip non-mutations (e.g., A -> A)

            mutation_name = f"{wt_aa}{real_site_num}{alt_aa}"
            all_mutations.append(mutation_name)

    print(f"Generated a total of {len(all_mutations)} possible single mutations.")
    return all_mutations


def main():
    try:
        # --- Step 1: Generate the master list of all possible mutations ---
        # This list is in the specific order you requested.
        all_mutations_list = generate_all_mutations(WT_SEQ, AA, SITE_OFFSET)
    except ValueError as e:
        print(f"Stopping script due to error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during mutation generation: {e}")
        return

    # --- Step 2: Load the "appeared" mutations data ---
    input_path = Path(INPUT_CSV)
    if not input_path.exists():
        print(f"Error: Input file not found at '{INPUT_CSV}'")
        print("Please make sure the output from the first script is in this directory.")
        return

    try:
        df_appeared = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading {INPUT_CSV}: {e}")
        return

    if "period" not in df_appeared.columns or "mutation" not in df_appeared.columns:
        print(
            f"Error: The CSV file '{INPUT_CSV}' must contain 'period' and 'mutation' columns."
        )
        return

    # Add the 'appeared_100' marker column for merging
    df_appeared["appeared_100"] = 1

    # --- Step 3: Get a unique, sorted list of all periods ---
    all_periods_list = sorted(df_appeared["period"].unique())
    if not all_periods_list:
        print("No periods found in the input CSV. Exiting.")
        return

    print(
        f"Found {len(all_periods_list)} unique time periods, from {all_periods_list[0]} to {all_periods_list[-1]}."
    )

    # --- Step 4: Create the full "cross-product" DataFrame ---
    # This creates a row for every mutation * every period.
    # The order is correct because 'all_mutations_list' is the outer loop.
    print("Creating full matrix (all mutations x all periods)...")
    df_full_matrix = pd.DataFrame.from_records(
        itertools.product(all_mutations_list, all_periods_list),
        columns=["mutation", "period"],
    )
    print(f"Full matrix has {len(df_full_matrix)} rows.")

    # --- Step 5: Merge the "appeared" data onto the full matrix ---
    # We use a 'left' merge to keep every row from the full matrix.
    df_final = pd.merge(
        df_full_matrix, df_appeared, on=["mutation", "period"], how="left"
    )

    # `fillna(0)` replaces all non-matches (NaN) with 0.
    # `astype(int)` converts the column from float (1.0) to integer (1).
    df_final["appeared_100"] = df_final["appeared_100"].fillna(0).astype(int)

    # --- Step 6: Save the final output ---
    try:
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSuccessfully created '{OUTPUT_CSV}'")

    except Exception as e:
        print(f"\nError writing output file to {OUTPUT_CSV}: {e}")


if __name__ == "__main__":
    main()
