import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import sys
import torch
import argparse


# Ensure proper module paths
try:
    current_script_path = os.path.dirname(__file__)
except NameError:
    current_script_path = os.getcwd()

main_path = os.path.abspath(os.path.join(current_script_path, "../"))
os.chdir(main_path)

if main_path not in sys.path:
    sys.path.append(main_path)
    sys.path.append(main_path + "PGM/source/")
    sys.path.append(main_path + "PGM/utilities/")

try:
    from global_variables import *
    from AB_RBM_EnergyModel import *
    from utils_evaluate_seq import *
    from Proteins_utils import *
except ImportError as e:
    raise ImportError(f"Failed to import one or more modules: {e}")


def generate_variants(
    ab=None,
    n_sequences=100,
    n_chains=2,
    warming_steps=50,
    steps_between_sampling=2,
    total_beta=1,
):
    """
    Generate sequence variants at different antibody concentrations.

    Parameters:
    - ab: Specific antibody to bias concentration (default: None)
    - n_sequences: Number of sequences to generate
    - n_chains: Number of parallel MCMC chains
    - warming_steps: Number of warming steps
    - steps_between_sampling: Steps between samplings
    - total_beta: Total beta factor
    """
    kd_vec = KD_VECTORS | NEW_KD_VECTORS
    kd_ace2 = ACE2_KD_VECTOR

    ab_names = list(kd_vec.keys())
    raw_concentrations = [-14 for _ in range(len(ab_names))]

    if ab is not None:
        ab_index = ab_names.index(ab)
        raw_concentrations[ab_index] = -6

    model = ACE2_Beta_EnergyModel(
        raw_ace2=-9.36,
        raw_beta=np.log(0.74),
        raw_concentrations=raw_concentrations,
        kd_vectors=kd_vec,
        total_beta=total_beta,
    )

    print("Concentration:", model.raw_concentrations)

    init_seq = torch.tensor(WT_SEQ, dtype=torch.int16)

    generated_sequences = gen_artif_data(
        model, n_sequences, n_chains, warming_steps, steps_between_sampling, init_seq
    )

    seq = num2seq(generated_sequences)
    print(seq)  # list of sequences

    # Create and save fasta file
    if ab == None:
        fasta_file = (
            "esmif/generated_seq/generated_seq_Beta" + str(total_beta) + ".fasta"
        )
    else:
        fasta_file = (
            "esmif/generated_seq/generated_seq_Beta"
            + str(total_beta)
            + "_Ab_"
            + ab
            + ".fasta"
        )
    with open(fasta_file, "w") as f:
        for i, s in enumerate(seq):
            f.write(f">seq_{i}\n{s}\n")
    print(f"FASTA file saved: {fasta_file}")


def main():
    """Main function to execute script."""
    parser = argparse.ArgumentParser(
        description="Generate sequence variants at different antibody concentrations."
    )
    parser.add_argument(
        "--ab", type=str, default=None, help="Specific antibody to bias concentration"
    )
    parser.add_argument(
        "--n_sequences", type=int, default=1000, help="Number of sequences to generate"
    )
    parser.add_argument(
        "--n_chains", type=int, default=20, help="Number of parallel MCMC chains"
    )
    parser.add_argument(
        "--warming_steps", type=int, default=500, help="Number of warming steps"
    )
    parser.add_argument(
        "--steps_between_sampling", type=int, default=20, help="Steps between samplings"
    )
    parser.add_argument("--total_beta", type=float, default=1, help="Total beta factor")

    args = parser.parse_args()

    generate_variants(
        ab=args.ab,
        n_sequences=args.n_sequences,
        n_chains=args.n_chains,
        warming_steps=args.warming_steps,
        steps_between_sampling=args.steps_between_sampling,
        total_beta=args.total_beta,
    )


if __name__ == "__main__":
    main()
