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
SCRIPT="../seq_data/script_extract_mutations_per_period.py"
IN_TSV="../seq_data/metadata.tsv"  
OUT_CSV="../seq_data/mutations_by_period.csv"

echo "Running $SCRIPT on $IN_TSV"
python3 "$SCRIPT" --tsv "$IN_TSV" --out-csv "$OUT_CSV" \
  > "outfile_extract_mutations_period__run" \
  2> "errfile_extract_mutations_period_run"

echo "Done. Output written to: $OUT_CSV"
