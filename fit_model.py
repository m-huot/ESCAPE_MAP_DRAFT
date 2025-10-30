#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train EscapeMapTorch per 3-month period from a FASTA file.

Example:
    python train_escape_map.py \
        --fasta_file data/spike_sequences.fasta \
        --out_dir params_by_period \
        --epochs 80 --lr 0.05 --batch_size 16 --gamma 0.97
"""

import argparse
import math
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Project imports
from escape_map_torch import EscapeMapTorch
from global_variables import (
    KD_VECTORS,
    ACE2_KD_VECTOR,
    RBM,
)  # adjust if names differ

import os, sys

# put this at the very top of your entry script or notebook, before importing numpy/torch
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

pgm_path = "PGM/"
if not os.path.isdir(pgm_path):  # check if folder exists
    from git import Repo

    Repo.clone_from("https://github.com/jertubiana/PGM.git", pgm_path)
sys.path.append(pgm_path + "source/")
sys.path.append(pgm_path + "utilities/")
# ------------------------------- Helpers -------------------------------------
import Proteins_utils

AA_CARDINALITY = 20  # change if your alphabet size differs


import matplotlib.pyplot as plt
from pathlib import Path


import matplotlib.pyplot as plt
from pathlib import Path


def _save_training_plot_params_only(label, out_dir, beta_hist, ace2_hist, sum_ab_hist):
    """
    Save a 3-row figure with one subplot per parameter:
      1) beta_rbm = exp(raw_beta)
      2) raw_ace2
      3) sum(raw_concentrations)
    """
    plot_dir = Path(out_dir) / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    epochs_axis = list(range(1, len(beta_hist) + 1))

    fig, axes = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    fig.suptitle(f"Training — {label}")

    ax = axes[0]
    ax.plot(epochs_axis, beta_hist)
    ax.set_ylabel("beta_rbm")

    ax = axes[1]
    ax.plot(epochs_axis, ace2_hist)
    ax.set_ylabel("raw_ace2")

    ax = axes[2]
    ax.plot(epochs_axis, sum_ab_hist)
    ax.set_ylabel("∑ raw_c")
    ax.set_xlabel("Epoch")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(plot_dir / f"training_{label}.png", dpi=150)
    plt.close(fig)


def _ensure_seq_np_int16(x: np.ndarray) -> np.ndarray:
    """Return a C-contiguous int16 numpy array of shape (L,) or (N, L)."""
    x = np.asarray(x)
    if x.dtype != np.int16:
        x = x.astype(np.int16, copy=False)
    return np.ascontiguousarray(x)


from datetime import datetime, date
from typing import List, Tuple


def read_fasta_headers_and_dates(fasta_path: str) -> Tuple[List[str], List[date]]:
    """
    Parse FASTA to extract headers and dates.
    Accepts headers starting with either:
      >YYYY-MM-DD|...
      >YYYY-MM|...
    For YYYY-MM, day defaults to the first of the month.
    Returns: (headers_without_gt, dates) aligned to sequence order.
    """
    headers: List[str] = []
    dates: List[date] = []

    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line or line[0] != ">":
                continue

            h = line[1:].strip()
            headers.append(h)
            first_field = h.split("|", 1)[0].strip()

            # Try YYYY-MM-DD
            try:
                dt = datetime.strptime(first_field, "%Y-%m-%d").date()
                dates.append(dt)
                continue
            except ValueError:
                pass

            # Try YYYY-MM (use day=1)
            try:
                dt_ym = datetime.strptime(first_field, "%Y-%m").date()
                dt_ym = date(dt_ym.year, dt_ym.month, 1)
                dates.append(dt_ym)
                continue
            except ValueError:
                pass

            # If neither format matched, raise a clear error
            raise ValueError(
                f"Header date must start with YYYY-MM-DD or YYYY-MM: '>{h}'"
            )

    return headers, dates


@dataclass(frozen=True)
class Period:
    start: date
    end: date  # inclusive end


def floor_to_quarter(dt: date, months: int = 3) -> date:
    """Return the period-start date by flooring to a months-sized bin starting from Jan."""
    # month in 1..12 -> bin 0..(12/months-1)
    bin_idx = (dt.month - 1) // months
    start_month = bin_idx * months + 1
    return date(dt.year, start_month, 1)


def period_end_from_start(start: date, months: int = 3) -> date:
    """Compute inclusive end date for a period starting at 'start' with 'months' duration."""
    year = start.year
    month = start.month + months - 1
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    # end is last day of that month
    if month in (1, 3, 5, 7, 8, 10, 12):
        day = 31
    elif month in (4, 6, 9, 11):
        day = 30
    else:
        # February
        y = year
        is_leap = (y % 400 == 0) or (y % 4 == 0 and y % 100 != 0)
        day = 29 if is_leap else 28
    return date(year, month, day)


def build_3month_period_bins(
    dates: List[date], months: int = 3
) -> Dict[str, List[int]]:
    """
    Group sequence indices into 3-month periods.
    Returns mapping: period_label -> list(indices), where label is the ISO date of period start.
    """
    buckets: Dict[str, List[int]] = {}
    for idx, dt in enumerate(dates):
        p_start = floor_to_quarter(dt, months=months)
        label = p_start.isoformat()  # e.g., '2020-10-01'
        buckets.setdefault(label, []).append(idx)
    return dict(sorted(buckets.items()))


@torch.no_grad()
def mcmc_sampling(
    model: EscapeMapTorch,
    steps: int,
    init_seq: np.ndarray,
    temperature: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Metropolis–Hastings on sequence space with single-site mutations.
    Target ∝ exp(-E(s)/T).
    Returns a sequence of shape (1, L), int16.
    """
    if rng is None:
        rng = np.random.default_rng()

    seq = _ensure_seq_np_int16(init_seq)
    if seq.ndim == 1:
        seq = seq[None, :]

    curr_E = float(model(seq)[0].item())
    L = seq.shape[1]

    for _ in range(steps):
        i = int(rng.integers(0, L))
        curr_token = int(seq[0, i])

        # propose a different residue
        prop_token = curr_token
        while prop_token == curr_token:
            prop_token = int(rng.integers(0, AA_CARDINALITY))

        prop = seq.copy()
        prop[0, i] = np.int16(prop_token)

        prop_E = float(model(prop)[0].item())
        dE = prop_E - curr_E
        if dE <= 0.0 or rng.random() < math.exp(-dE / max(1e-12, temperature)):
            seq = prop
            curr_E = prop_E

    return _ensure_seq_np_int16(seq)


def save_params_for_period(
    model: EscapeMapTorch, kd_vectors: Dict[str, np.ndarray], out_csv_path: Path
) -> None:
    """Save one-row CSV with raw_beta, raw_ace2, and raw_c_<AB> in KD_VECTORS key order."""
    if not isinstance(kd_vectors, dict):
        raise TypeError("KD_VECTORS must be a dict to preserve antibody order.")

    ab_names = list(kd_vectors.keys())
    rc = model.raw_concentrations.detach().cpu().numpy()
    assert rc.shape[0] == len(ab_names), (
        "raw_concentrations length must match KD_VECTORS."
    )

    row = {
        "raw_beta": float(model.raw_beta.detach().cpu().numpy()),
        "raw_ace2": float(model.raw_ace2.detach().cpu().numpy()),
    }
    for name, val in zip(ab_names, rc):
        row[f"raw_c_{name}"] = float(val)

    df = pd.DataFrame([row])
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)


# -------------------------- Training per period ------------------------------


def train_model_per_period(
    base_model: EscapeMapTorch,
    kd_vectors: Dict[str, np.ndarray],
    train_arrays: List[np.ndarray],
    period_labels: List[str],
    *,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 10,
    gamma: float = 0.95,
    mcmc_warmup_steps: int = 200,
    mcmc_steps_fast: int = 30,
    out_dir: Path = Path("params_by_period"),
) -> None:
    """Fit a fresh copy of the model per period, save params CSV and a plot of loss & params."""
    rng = np.random.default_rng()
    out_dir.mkdir(parents=True, exist_ok=True)

    for period_arr, label in tqdm(
        zip(train_arrays, period_labels), total=len(period_labels), desc="Periods"
    ):
        period_arr = _ensure_seq_np_int16(period_arr)
        assert period_arr.ndim == 2, "Each period array must be (N, L)."
        N, L = period_arr.shape

        model = deepcopy(base_model).to(base_model.device)
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        # --- histories for plotting ---
        loss_hist: List[float] = []
        beta_hist: List[float] = []
        ace2_hist: List[float] = []
        sum_ab_hist: List[float] = []

        last_chain = None
        for epoch in range(epochs):
            optimizer.zero_grad()
            total_loss = 0.0

            idxs = rng.integers(0, N, size=(batch_size,))
            warm = last_chain is None

            for idx in idxs:
                data_seq = period_arr[int(idx)][None, :]  # (1, L)
                data_E = model(data_seq)[0]

                steps = mcmc_warmup_steps if warm else mcmc_steps_fast
                init = last_chain if last_chain is not None else data_seq
                neg_seq = mcmc_sampling(
                    model, steps=steps, init_seq=init, temperature=1.0, rng=rng
                )
                last_chain = neg_seq

                neg_E = model(neg_seq)[0]
                loss = data_E - neg_E
                loss.backward()
                total_loss += float(loss.detach().cpu().numpy())

            optimizer.step()
            scheduler.step()

            # --- record metrics after optimizer step ---
            with torch.no_grad():
                beta_rbm = float(torch.exp(model.raw_beta).detach().cpu().numpy())
                raw_ace2_val = float(model.raw_ace2.detach().cpu().numpy())
                sum_ab = float(model.raw_concentrations.detach().cpu().numpy().sum())

            loss_hist.append(total_loss)
            beta_hist.append(beta_rbm)
            ace2_hist.append(raw_ace2_val)
            sum_ab_hist.append(sum_ab)

        # Save this period’s parameters
        csv_path = out_dir / f"param_period_{label}.csv"
        save_params_for_period(model, kd_vectors, csv_path)

        # Save the plot for this period
        _save_training_plot_params_only(
            label, out_dir, beta_hist, ace2_hist, sum_ab_hist
        )


# ------------------------------- Main ----------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train EscapeMapTorch per 3-month period from FASTA."
    )
    parser.add_argument(
        "--fasta_file",
        type=str,
        default="seq_data/ns_mutated_spike_100k.fasta",
        help="Input FASTA file with spike sequences",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="params_by_period",
        help="Output folder for CSV parameter files",
    )
    parser.add_argument(
        "--period_months",
        type=int,
        default=3,
        help="Size of time bin in months (default: 3)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date YYYY-MM-DD to include",
    )
    parser.add_argument(
        "--end", type=str, default=None, help="Optional end date YYYY-MM-DD to include"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,  # 0.03 default
    )
    parser.add_argument("--batch_size", type=int, default=10)  # new
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mcmc_warmup_steps", type=int, default=600)
    parser.add_argument("--mcmc_steps_fast", type=int, default=100)

    # Model init overrides (optional)
    parser.add_argument(
        "--raw_ace2",
        type=float,
        default=-8.0,  # -8
        help="ACE2 concentration in log10 space",
    )
    parser.add_argument("--raw_beta", type=float, default=-1.0, help="log(beta)")
    parser.add_argument(
        "--total_beta", type=float, default=1.0, help="global scaling of energy"
    )

    args = parser.parse_args()

    fasta_file = Path(args.fasta_file)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) Load sequences (keeps order)
    seqs = Proteins_utils.load_FASTA(
        str(fasta_file), drop_duplicates=False
    )  # expected (N, L) int-coded
    seqs = _ensure_seq_np_int16(seqs)

    # 2) Parse headers and dates aligned with the above order
    headers, dates = read_fasta_headers_and_dates(str(fasta_file))

    if len(seqs) != len(dates):
        raise RuntimeError(
            "Number of sequences from Proteins_utils.load_FASTA does not match number of FASTA headers."
        )

    # Optional date filtering
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    else:
        start_date = min(dates)

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    else:
        end_date = max(dates)

    # Keep only indices within [start_date, end_date]
    kept_idx = [i for i, d in enumerate(dates) if start_date <= d <= end_date]
    if not kept_idx:
        raise RuntimeError("No sequences fall within the requested date range.")

    seqs = seqs[kept_idx, :]
    dates = [dates[i] for i in kept_idx]

    # 3) Build 3-month period bins and materialize arrays per period
    buckets = build_3month_period_bins(dates, months=args.period_months)
    period_labels = list(buckets.keys())
    train_arrays = [
        seqs[np.asarray(buckets[label], dtype=int)] for label in period_labels
    ]

    # 4) Build base model from global variables (KD_VECTORS order is used for saving)
    base_model = EscapeMapTorch(
        rbm=RBM,
        kd_vectors=KD_VECTORS,
        ace2_vector=ACE2_KD_VECTOR,
        raw_concentrations=None,  # from global_variables
        raw_ace2=args.raw_ace2,
        raw_beta=args.raw_beta,
        total_beta=args.total_beta,
    )

    # 5) Train per period and save CSV parameters
    train_model_per_period(
        base_model=base_model,
        kd_vectors=KD_VECTORS,
        train_arrays=train_arrays,
        period_labels=period_labels,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        gamma=args.gamma,
        mcmc_warmup_steps=args.mcmc_warmup_steps,
        mcmc_steps_fast=args.mcmc_steps_fast,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
