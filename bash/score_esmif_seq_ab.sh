#!/bin/bash
#SBATCH --job-name=seq_scoring_esmif_gpu         # Job name
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --ntasks-per-node=1                  # Number of cores per node
#SBATCH --gpus=1                             # No GPUs requested
#SBATCH --time=18:00:00                      # Time limit hrs:min:sec
#SBATCH --mem=25000  # memory in Mb
#SBATCH -o outfile  # send stdout to outfile
#SBATCH -e errfile  # send stderr to errfile
#SBATCH --partition=gpu_test                # Partition name


# Set the library path to the known location of the libpython2.7.so.1.0

# Set the library path





echo "Activating esmfold environment..."
source activate esmfold 


echo "Running Python script..."

python3 score_log_likelihoods.py data/7KMG.pdb \
    data/desai_CoV555.fasta --chain C \
    --outpath output/desai_LY-CoV555_esmif.csv 
    --multichain-backbone
    
python3 score_log_likelihoods.py data/9LYP.pdb \
    data/desai_REGN10987.fasta --chain B \
    --outpath output/desai_REGN10987_esmif.csv 
    --multichain-backbone
    
python3 score_log_likelihoods.py data/7C01.pdb \
    data/desai_CB6.fasta --chain B \
    --outpath output/desai_CB6_esmif.csv 
    --multichain-backbone

