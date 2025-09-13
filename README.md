# ESCAPE_MAP_DRAFT

Probabilistic model for antigenic escape on SARS‑CoV‑2 Spike RBD. It combines:

- Antibody binding predictors (Kd per antibody)
- ACE2 binding predictor (Kd to receptor)
- A sequence prior from an RBM trained on natural RBD sequences

Two implementations are provided:

- `escape_map.py`: NumPy implementation with simple MCMC sequence generation
- `escape_map_torch.py`: PyTorch module with learnable parameters (for fitting / fine‑tuning)


## Quick Start

Install Python dependencies (CPU; GPU works if PyTorch with CUDA is installed):

```bash
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib tqdm torch scikit-learn jupyter
```

Ensure the expected data files are in place (see Data Layout below), then in Python:

```python
import numpy as np
from escape_map import EscapeMap
from global_variables import WT_SEQ  # integer-encoded WT RBD segment

model = EscapeMap()                 # uses defaults from global_variables.py
energy = model(WT_SEQ)              # lower is better (score = -energy)
score = -energy

# Sample sequences from the model (integer-encoded, shape: [N, L])
samples = model.sample(n=10)

# Convert to amino-acid strings for display
from PGM.utilities.Proteins_utils import num2seq
print(num2seq(samples[:3]))
```


## What This Model Computes

For a sequence `s` (integer-encoded, 0..20 per position):

- Antibody energy: softplus(concentration − logKd_ab(s)) summed over antibodies
- ACE2 energy: softplus(logKd_ACE2(s) − ACE2_level)
- RBM energy: β · FreeEnergy_RBM(s)

The total energy is scaled by a global factor `total_beta`. Lower energy corresponds to better “escape”/fitness in this convention; we typically use `score = -energy`.


## Repository Structure

- `escape_map.py`: Core NumPy model (`EscapeMap`) + MCMC sampling utilities
- `escape_map_torch.py`: PyTorch module (`EscapeMapTorch`) with learnable parameters
- `utils.py`: One‑hot encoding and Kd helpers (`get_Kd`, etc.)
- `global_variables.py`: Paths, data loading (RBM, KD vectors, ACE2 vector), WT sequence
- `utils_evaluate_seq.py`: Convenience functions to analyze generated/evaluated sequences
- `PGM/`: Supporting RBM/PGM code vendored with this repo
- `seq_data/`: Example sequence files (WT, alignments, etc.)
- `generative/`: Notebooks to generate/analyze sequences
- `fitness/`, `experiences/`: Analysis notebooks and assets


## Data Layout (expected files)

By default, paths are configured in `global_variables.py`. Make sure the following exist:

- RBM weights: `test_wt_RBM_Covid.data` (in repo root, already present)
- KD vectors (per antibody): place `.npy` files under `exp_data/kd_vectors/`
  - Each file stores a single 1D vector `q = [w ... w, b]` where `w` has length `L*21` (one‑hot features) and `b` is the intercept
  - Filenames can be arbitrary `.npy`; if they end with `delta_G.npy`, that suffix is removed for the antibody name
- ACE2 KD vector: `exp_data/ace2_kd_vector/ace2_delta_log10kd.npy`
- WT sequence: `seq_data/rbd_wt.fasta` (already present); the code trims positions defined by `BEGIN` and `END` in `global_variables.py`

If you store data elsewhere, update the paths in `global_variables.py` accordingly.


## Using the NumPy Model (`escape_map.py`)

```python
import numpy as np
from escape_map import EscapeMap, load_escape_map_from_csv
from global_variables import WT_SEQ

# 1) Default model using data from global_variables.py
model = EscapeMap()           # antibodies, ACE2, RBM, and defaults are injected
score = -model(WT_SEQ)        # score = -energy

# 2) Customize concentrations/ACE2/beta from a CSV row
#    CSV columns: raw_ace2, raw_beta, and one column per antibody named: raw_c_{antibody_name}
model2 = load_escape_map_from_csv("my_params.csv")

# 3) Sample sequences with basic MCMC
samples = model.sample(n=100)
```

Key parameters (log‑domain unless noted):

- `raw_concentrations`: per‑antibody log10 concentration, default `-12`
- `raw_ace2`: ACE2 competitor level, default `-5`
- `raw_beta`: log weight applied to RBM free energy
- `total_beta`: global positive scale of the whole energy


## Using the PyTorch Module (`escape_map_torch.py`)

`EscapeMapTorch` mirrors the NumPy model but exposes learnable parameters (`raw_concentrations`, `raw_ace2`, `raw_beta`, and `raw_total_beta`) for gradient‑based fitting:

```python
import torch
import numpy as np
from escape_map_torch import EscapeMapTorch
from global_variables import RBM, KD_VECTORS, ACE2_KD_VECTOR, WT_SEQ

torch_model = EscapeMapTorch(
    rbm=RBM,
    kd_vectors=KD_VECTORS,
    ace2_vector=ACE2_KD_VECTOR,
    total_beta=1.0,
)

energy = torch_model(WT_SEQ)     # torch scalar (dtype float64)
# Example: optimize parameters given some supervision on sequences
# optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-2)
```


## Notebooks

- `generative/gen_data.ipynb`: Generate sequences with MCMC
- `generative/analyse_gen_data.ipynb`: Analyze generated sequences (Kd, distances, RBM energy)
- `fitness/model_fit.ipynb`, `fitness/plot_*`: Exploratory analysis and plotting
- `experiences/plot_hugo.ipynb`: Additional experiments/plots


## Tips and Common Pitfalls

- Sequence representation is integer‑encoded (0..20 with gap/unknown), not letters. Use `PGM.utilities.Proteins_utils.num2seq` and `seq2num` to convert.
- KD vectors must match the one‑hot feature length `L*21 + 1` (intercept). If lengths mismatch, check trimming (`BEGIN`, `END`) and that your vectors were trained on the same segment.
- If `exp_data/` is missing, create it and add the expected subfolders/files as described above.


## License

This project is released under the MIT License. See `LICENSE` for details.


## Acknowledgements

This codebase builds on an RBM implementation and utilities in `PGM/` (included here for convenience). Please cite the relevant RBM/PGM works when publishing results derived from this repository.
