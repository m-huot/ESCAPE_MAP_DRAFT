from EscapeMap import *
from global_variables import *

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import torch

import sys

pgm_path = "PGM/"
sys.path.append(pgm_path + "source/")
sys.path.append(pgm_path + "utilities/")

import utilities, Proteins_utils, sequence_logo, plots_utils
import rbm, RBM_utils
import numpy as np
import os
import random
from tqdm import tqdm

kd_vec = KD_VECTORS
kd_ace2 = ACE2_KD_VECTOR


def one_hot_encoding(encoded_sequences):
    """Assuming encoded_sequences is a 2D numpy array of size N*L
    where N is the number of sequences and L is the length of each sequence
    and encoded_sequences[i,j] is the integer encoding of the jth amino acid of the ith sequence
    (0 to 20 encoding)
    """

    N, L = encoded_sequences.shape
    num_amino_acids = 21  # Including 0 to 20 encoding

    # create array of size N*L*num_amino_acids
    one_hot_encoding = np.zeros((N, L, num_amino_acids))

    for position in range(L):
        one_hot_encoding[np.arange(N), position, encoded_sequences[:, position]] = 1
    rep = []
    for matrix in one_hot_encoding:
        rep.append(matrix.flatten().tolist())
    return np.array(rep)


def hamming(s1, s2):
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def distances_wt(generated_sequences):
    """
    Output: List of hamming distances between WT and generated sequences
    """
    return [hamming(WT_SEQ, s) for s in generated_sequences]


def first_logkd(generated_sequences, ab_index, kd_vec=KD_VECTORS):
    """
    Output: log10(Kd) of generated sequences for the antibody at index ab_index in kd_vec
    """
    generated_sequences = torch.tensor(generated_sequences)
    out = np.log10(
        np.array([get_Kd(s, kd_vec).numpy()[ab_index, 0] for s in generated_sequences])
    )
    # -15 if <-15 or inf
    out[out < -15] = -15
    # out[out == -np.inf] = -15
    # out[out == np.inf] = -15

    return out


def ace2_logkd(generated_sequences):
    """
    Output: log10(Kd) of generated sequences for the antibody at index ab_index in kd_vec
    """
    generated_sequences = torch.tensor(generated_sequences)
    out = np.log10(
        np.array([get_Kd(s, kd_ace2).numpy()[0, 0] for s in generated_sequences])
    )
    # -15 if <-15 or inf
    out[out < -15] = -15
    # out[out == -np.inf] = -15
    # out[out == np.inf] = -15

    return out


def rbm_energy(generated_sequences, rbm=RBM):
    """
    Output: RBM energy of generated sequences
    """
    return [rbm.free_energy(s)[0] * BETA_RBM for s in generated_sequences]


def av_distances_wt(generated_sequences):
    """
    Output: Average /std of hamming distance between WT and generated sequences
    """
    return np.mean([hamming(WT_SEQ, s) for s in generated_sequences]), np.std(
        [hamming(WT_SEQ, s) for s in generated_sequences]
    )


def av_rbm_energy(generated_sequences, rbm=RBM):
    """
    Output: Average /std of RBM energy of generated sequences
    """
    return np.mean(
        [rbm.free_energy(s)[0] * BETA_RBM for s in generated_sequences]
    ), np.std([rbm.free_energy(s)[0] * BETA_RBM for s in generated_sequences])


def av_first_logkd(generated_sequences, ab_index, kd_vec=KD_VECTORS):
    """
    Output: Average /std of log10(Kd) of generated sequences for the antibody at index ab_index in kd_vec
    """
    generated_sequences = torch.tensor(generated_sequences)
    return np.mean(
        np.log10(
            np.array(
                [get_Kd(s, kd_vec).numpy()[ab_index, 0] for s in generated_sequences]
            )
        )
    ), np.std(
        np.log10(
            np.array(
                [get_Kd(s, kd_vec).numpy()[ab_index, 0] for s in generated_sequences]
            )
        )
    )


# ace2_vector_directory = "exp_data/kd_ace2/ace2_delta_log10kd.npy"
# ace2_vector = np.load(ace2_vector_directory)
# ace2_kd_vec = {"ace2": ace2_vector}


def delta_log10kd_ace2(generated_sequences):
    """
    Output: log10(Kd) of generated sequences for ACE2
    """
    generated_sequences = torch.tensor(generated_sequences)
    return np.log10(
        np.array([get_Kd(s, kd_ace2).numpy()[0, 0] for s in generated_sequences])
    )


def av_delta_log10kd_ace2(generated_sequences):
    """
    Output: Average /std of log10(Kd) of generated sequences for ACE2
    """
    generated_sequences = torch.tensor(generated_sequences)
    return np.mean(
        np.log10(
            np.array([get_Kd(s, kd_ace2).numpy()[0, 0] for s in generated_sequences])
        )
    ), np.std(
        np.log10(
            np.array([get_Kd(s, kd_ace2).numpy()[0, 0] for s in generated_sequences])
        )
    )
