#!/bin/sh
#SBATCH -p shakhnovich,shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=50000
#SBATCH -o outfile_extract_mutations
#SBATCH -e errfile_extract_mutations
#SBATCH -t 20:00:00

# Use your existing environment
source activate lantern

python3 ../fit_model.py
echo "Done. "
