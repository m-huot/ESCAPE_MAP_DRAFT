import math
import numpy as np
import torch
import torch.nn as nn

# expects RBM, KD_VECTORS, ACE2_KD_VECTOR, get_Kd to be available
from utils import get_Kd


def stable_logexp_torch(x: torch.Tensor) -> torch.Tensor:
    # exact branching as your NumPy stable_logexp
    out = torch.empty_like(x)
    mask = x > 1
    out[mask] = x[mask] + torch.log1p(torch.exp(-x[mask]))
    out[~mask] = torch.log1p(torch.exp(x[~mask]))
    return out


class EscapeMapTorch(nn.Module):
    def __init__(
        self,
        rbm,
        kd_vectors,
        ace2_vector,
        raw_concentrations=None,
        raw_ace2=None,
        raw_beta=None,
        total_beta=1.0,
        device=None,
        dtype=torch.float64,  # match np.float64
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        # store external components exactly as in NumPy version
        self.kd_vectors = kd_vectors
        self.ace2_vector = {"ace2": ace2_vector}
        self.rbm = rbm
        self.total_beta = float(total_beta)
        self.ln10 = math.log(10.0)

        A = len(kd_vectors)
        if raw_concentrations is None:
            rc = torch.full((A,), -12.0, dtype=dtype, device=self.device)
        else:
            rc = torch.as_tensor(raw_concentrations, dtype=dtype, device=self.device)
            assert rc.shape == (A,), (
                "raw_concentrations must have length = number of antibodies"
            )
        self.raw_concentrations = nn.Parameter(rc)

        ra = -5.0 if raw_ace2 is None else float(raw_ace2)
        self.raw_ace2 = nn.Parameter(torch.tensor(ra, dtype=dtype, device=self.device))

        rb = 0.0 if raw_beta is None else float(raw_beta)
        self.raw_beta = nn.Parameter(torch.tensor(rb, dtype=dtype, device=self.device))

    @staticmethod
    def _to_int16_c_contig(x: np.ndarray) -> np.ndarray:
        if x.dtype != np.int16:
            x = x.astype(np.int16, copy=False)
        return np.ascontiguousarray(x)

    def _get_Kd_batch_numpy(
        self, seqs_np: np.ndarray, vectors: dict, log10: bool
    ) -> np.ndarray:
        # identical control flow to your NumPy version
        try:
            out = get_Kd(seqs_np, vectors, log10=log10)
        except Exception:
            out = np.stack([get_Kd(x, vectors, log10=log10) for x in seqs_np], axis=0)
        return np.asarray(out)

    @torch.no_grad()
    def _rbm_free_energy(self, seqs_np: np.ndarray) -> torch.Tensor:
        fe = self.rbm.free_energy(seqs_np)  # NumPy [N]
        return torch.as_tensor(fe, dtype=self.dtype, device=self.device)

    def forward(self, s):
        # Accept (L,) or (N,L), keep NumPy path for Kd and RBM to match numerics
        s_np = np.asarray(s)
        single = s_np.ndim == 1
        seqs_np = s_np[None, :] if single else s_np
        seqs_np = self._to_int16_c_contig(seqs_np)  # numba-friendly

        beta = torch.exp(self.raw_beta)  # scalar
        ln10 = self.ln10

        # ----- Antibody Kd -----
        kds_np = self._get_Kd_batch_numpy(seqs_np, self.kd_vectors, log10=True) * ln10
        kds_np = np.squeeze(kds_np)  # (A,) or (N,A)
        kds = torch.as_tensor(kds_np, dtype=self.dtype, device=self.device)

        # ----- ACE2 Kd -----
        kdace2_np = (
            self._get_Kd_batch_numpy(seqs_np, self.ace2_vector, log10=True) * ln10
        )
        kdace2_np = np.squeeze(kdace2_np)  # () or (N,)
        kdace2_np = np.clip(kdace2_np, -15.0, -5.0)
        kdace2 = torch.as_tensor(kdace2_np, dtype=self.dtype, device=self.device)

        # ----- Antibody energy -----
        conc = self.raw_concentrations * ln10  # [A]
        logdiffs = -kds + conc  # (A,) or (N,A)
        energy = stable_logexp_torch(logdiffs).sum(dim=-1)  # scalar or [N]

        # ----- ACE2 energy -----
        energy = energy + stable_logexp_torch(kdace2 - self.raw_ace2)  # scalar or [N]

        # ----- RBM free energy -----
        fe = self._rbm_free_energy(seqs_np)  # [N]
        energy = energy + beta * fe

        energy = energy * self.total_beta  # scalar or [N]
        if single:
            return energy.reshape(())[()]  # scalar tensor
        return energy
