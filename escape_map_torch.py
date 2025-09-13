import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from math import ceil
from utils import get_Kd
import math


# ===== EscapeMapTorch with learnable total_beta =====
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
        dtype=torch.float64,
    ):
        super().__init__()
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.kd_vectors = kd_vectors
        self.ace2_vector = {"ace2": ace2_vector}
        self.rbm = rbm
        self.ln10 = math.log(10.0)

        A = len(kd_vectors)
        rc = (
            torch.full((A,), -10.0, dtype=dtype, device=self.device)
            if raw_concentrations is None
            else torch.as_tensor(raw_concentrations, dtype=dtype, device=self.device)
        )
        self.raw_concentrations = nn.Parameter(rc)

        ra = -10.0 if raw_ace2 is None else float(raw_ace2)
        rb = -2 if raw_beta is None else float(raw_beta)
        self.raw_ace2 = nn.Parameter(torch.tensor(ra, dtype=dtype, device=self.device))
        self.raw_beta = nn.Parameter(torch.tensor(rb, dtype=dtype, device=self.device))

        # NEW: learnable total_beta (positive)
        self.raw_total_beta = nn.Parameter(
            torch.tensor(float(np.log(total_beta)), dtype=dtype, device=self.device)
        )

    @staticmethod
    def _to_int16_c_contig(x):
        if x.dtype != np.int16:
            x = x.astype(np.int16, copy=False)
        return np.ascontiguousarray(x)

    def _get_Kd_batch_numpy(self, seqs_np, vectors, log10=True):
        try:
            out = get_Kd(seqs_np, vectors, log10=log10)
        except Exception:
            out = np.stack([get_Kd(x, vectors, log10=log10) for x in seqs_np], axis=0)
        return np.asarray(out)

    @torch.no_grad()
    def _rbm_free_energy(self, seqs_np):
        fe = self.rbm.free_energy(seqs_np)
        return torch.as_tensor(fe, dtype=self.dtype, device=self.device)

    def forward(self, s):
        s_np = np.asarray(s)
        single = s_np.ndim == 1
        seqs_np = s_np[None, :] if single else s_np
        seqs_np = self._to_int16_c_contig(seqs_np)

        beta = torch.exp(self.raw_beta)
        total_beta = torch.exp(self.raw_total_beta)  # positive
        ln10 = self.ln10

        kds_np = self._get_Kd_batch_numpy(seqs_np, self.kd_vectors, log10=True) * ln10
        kds = torch.as_tensor(np.squeeze(kds_np), dtype=self.dtype, device=self.device)

        kdace2_np = (
            self._get_Kd_batch_numpy(seqs_np, self.ace2_vector, log10=True) * ln10
        )
        kdace2_np = np.clip(np.squeeze(kdace2_np), -15.0, -5.0)
        kdace2 = torch.as_tensor(kdace2_np, dtype=self.dtype, device=self.device)

        conc = self.raw_concentrations * ln10

        def softplus_stable(x):
            out = torch.empty_like(x)
            m = x > 1
            out[m] = x[m] + torch.log1p(torch.exp(-x[m]))
            out[~m] = torch.log1p(torch.exp(x[~m]))
            return out

        energy = softplus_stable(-kds + conc).sum(dim=-1)
        energy = energy + softplus_stable(kdace2 - self.raw_ace2)

        fe = self._rbm_free_energy(seqs_np)
        energy = (energy + beta * fe) * total_beta
        return energy[0] if single else energy
