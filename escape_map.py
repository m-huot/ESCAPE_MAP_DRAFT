import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import random
from tqdm import tqdm

import sys

pgm_path = "PGM/"

sys.path.append(pgm_path + "source/")
sys.path.append(pgm_path + "utilities/")

import utilities, Proteins_utils, sequence_logo, plots_utils
import rbm, RBM_utils

from global_variables import *
from utils import *


def stable_logexp(x):
    """
    Compute log(1+exp(x)) in a numerically stable way when x is a NumPy array or scalar.
    """
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)
    mask = x > 1
    out[mask] = x[mask] + np.log1p(np.exp(-x[mask]))
    out[~mask] = np.log1p(np.exp(x[~mask]))
    return out


class EscapeMap:
    def __init__(
        self,
        rbm=RBM,
        kd_vectors=KD_VECTORS,
        ace2_vector=ACE2_KD_VECTOR,
        raw_concentrations=None,
        raw_ace2=None,
        raw_beta=None,
        total_beta=1,
    ):
        if raw_concentrations is None:
            self.raw_concentrations = -9.0 * np.ones(len(kd_vectors), dtype=np.float64)
        else:
            assert len(raw_concentrations) == len(kd_vectors)
            self.raw_concentrations = np.array(raw_concentrations, dtype=np.float64)

        self.raw_ace2 = -5.0 if raw_ace2 is None else float(raw_ace2)
        self.raw_beta = 0.0 if raw_beta is None else float(raw_beta)
        # create a dic with key 'ace2' and value ace2_vector
        ace2_vector = {"ace2": ace2_vector}

        self.kd_vectors = kd_vectors
        self.ace2_vector = ace2_vector
        self.rbm = rbm
        self.total_beta = float(total_beta)

    def forward(self, s):
        # Accept (L,) or (N, L)
        s = np.asarray(s, dtype=np.int16)
        single = s.ndim == 1
        seqs = s[np.newaxis, :] if single else s  # (N, L)

        beta = np.exp(self.raw_beta)
        ln10 = np.log(10.0)

        # Helper: batch-safe get_Kd
        def _get_Kd_batch(seqs_, vectors, log10=True):
            try:
                out = get_Kd(seqs_, vectors, log10=log10)
            except Exception:
                out = np.stack([get_Kd(x, vectors, log10=log10) for x in seqs_], axis=0)
            return np.asarray(out)

        # Antibody Kd: (N, A)
        kds = _get_Kd_batch(seqs, self.kd_vectors, log10=True) * ln10
        kds = np.squeeze(kds)  # keep (N, A) if batch

        # ACE2 Kd: (N,)
        kdace2 = _get_Kd_batch(seqs, self.ace2_vector, log10=True) * ln10
        kdace2 = np.squeeze(kdace2)
        kdace2 = np.clip(kdace2, -15, -5)

        # Antibody energy: sum over antibodies
        conc = np.asarray(self.raw_concentrations, dtype=float) * ln10  # (A,)
        logdiffs = -kds + conc  # (N, A) via broadcast
        energy = stable_logexp(logdiffs).sum(axis=-1)  # (N,)

        # ACE2 energy
        ace2_energy = stable_logexp(kdace2 - self.raw_ace2)  # (N,)
        energy = energy + ace2_energy

        # RBM free energy: rbm.free_energy returns (N,)
        energy = energy + beta * self.rbm.free_energy(seqs)

        energy = energy * self.total_beta  # (N,)
        return energy[0] if single else energy

    def __call__(self, s):
        return self.forward(s)

    def copy(self):
        return EscapeMap(
            rbm=self.rbm,
            kd_vectors=self.kd_vectors,
            ace2_vector=self.ace2_vector,
            raw_concentrations=self.raw_concentrations.copy(),
            raw_ace2=self.raw_ace2,
            raw_beta=self.raw_beta,
            total_beta=self.total_beta,
        )

    def sample(self, n=1):
        """return n sequences sampled from the model"""
        sequences = gen_artif_data(
            self,
            n_sequences=n,
            n_chains=1,
            warming_steps=500,
            steps_between_sampling=10,
        )
        return sequences.astype(np.int16)


def perturb_sequence(seq):
    """
    Perturb a given sequence. Choose a random site between 0 and 177, then choose a random number between 0 and 20 and put it in the position of the site
    """
    if seq.ndim == 1:
        seq = seq[np.newaxis, :]

    site = np.random.randint(0, 178)
    new_seq = seq.copy()
    new_seq[:, site] = np.random.randint(0, 20)
    return new_seq


def mcmc_sampling(model, steps=200, init_seq=None):
    """
    Perform MCMC sampling to generate sequences, initializing each new sample
    from the previously chosen sample from the distribution.
    """
    seq = init_seq
    for _ in range(steps):
        seq_proposal = perturb_sequence(seq)
        energy_diff = model.forward(seq_proposal[0]) - model.forward(seq[0])
        if np.random.rand() < np.exp(-energy_diff):
            seq = seq_proposal
    return seq


def gen_artif_data(
    model,
    n_sequences,
    n_chains,
    warming_steps,
    steps_between_sampling,
    init_seq=np.array(WT_SEQ, dtype=np.int16),
):
    """
    Generate artificial sequences using MCMC sampling, in an array of arrays
    """
    init_seq = init_seq[np.newaxis, :]

    sampled_sequences = []
    print("Generating sequences...")

    for chain in tqdm(range(n_chains)):
        last_seq = None

        for seq_round in range(n_sequences // n_chains):
            MCMC_steps = warming_steps if seq_round == 0 else steps_between_sampling
            if last_seq is None:
                last_seq = init_seq
            sampled_seq = mcmc_sampling(model, steps=MCMC_steps, init_seq=last_seq)
            last_seq = sampled_seq
            sampled_sequences.append(sampled_seq[0, :])

    sampled_sequences = np.array(sampled_sequences, dtype=np.int16)
    print("output shape:", sampled_sequences.shape)

    return sampled_sequences


def load_escape_map_from_csv(
    csv_path,
    rbm=RBM,
    kd_vectors=KD_VECTORS,
    ace2_vector=ACE2_KD_VECTOR,
    default_conc=-18.0,
):
    df = pd.read_csv(csv_path)
    row = df.iloc[0]

    ab_names = list(kd_vectors.keys())
    raw_concs = []
    for ab in ab_names:
        colname = f"raw_c_{ab}"
        if colname in row and not pd.isna(row[colname]):
            raw_concs.append(row[colname])
        else:
            print(
                f"Warning: Antibody {ab} concentration not found in CSV; using default {default_conc}."
            )
            raw_concs.append(default_conc)

    model = EscapeMap(
        rbm=rbm,
        kd_vectors=kd_vectors,
        ace2_vector=ace2_vector,
        raw_concentrations=raw_concs,
        raw_ace2=row["raw_ace2"],
        raw_beta=row["raw_beta"],
        total_beta=1.0,
    )
    return model


def score_seq_batch(model, seqs_np):
    X = seqs_np

    s = -model(X)  # score = -energy
    return s
