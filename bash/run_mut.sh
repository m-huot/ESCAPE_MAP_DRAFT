#!/bin/sh
#SBATCH -p shakhnovich,shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=50000
#SBATCH -o outfile_extract_mutations
#SBATCH -e errfile_extract_mutations
#SBATCH -t 10:00:00

# Use your existing environment
source activate lantern

# --- paths ---
SCRIPT="../seq_data/script_extract_mutations.py"
IN_TSV="../seq_data/metadata.tsv"            # change if your input is elsewhere
OUT_CSV="../seq_data/all_mutation_summary.csv"

echo "Running $SCRIPT on $IN_TSV"
python3 "$SCRIPT" "$IN_TSV" --out-csv "$OUT_CSV" \
  > "outfile_extract_mutations_run" \
  2> "errfile_extract_mutations_run"

echo "Done. Output written to: $OUT_CSV"
