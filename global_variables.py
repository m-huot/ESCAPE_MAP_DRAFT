import sys, os
from pathlib import Path
import numpy as np

# --- Resolve project paths relative to THIS file ---
FILE_DIR = Path(__file__).resolve().parent
PGM_DIR = FILE_DIR / "PGM"
EXP_DIR = FILE_DIR / "exp_data"
SEQ_DIR = FILE_DIR / "seq_data"

# Make siblings importable (rbm.py sits next to this file)
if str(FILE_DIR) not in sys.path:
    sys.path.insert(0, str(FILE_DIR))

# If PGM subpackages are not proper packages, expose them
sys.path.append(str(PGM_DIR / "source"))
sys.path.append(str(PGM_DIR / "utilities"))

# Now safe to import
import utilities, Proteins_utils, sequence_logo, plots_utils
import rbm, RBM_utils  # rbm.py must be alongside this file

BEGIN = 18
END = 5

kd_vectors_directory = EXP_DIR / "kd_vectors"
new_kd_vectors_directory = EXP_DIR / "new_ab_kd_vectors"
ace2_vector_directory = EXP_DIR / "ace2_kd_vector" / "ace2_delta_log10kd.npy"


def load_kd_vectors(directory: Path):
    kd_vectors = {}
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            antibody_name = filename.replace("delta_G.npy", "")
            kd_vector = np.load(str(directory / filename))
            kd_vectors[antibody_name] = kd_vector
    for name, vec in kd_vectors.items():
        if np.any(np.isinf(vec)):
            raise ValueError(f"Inf value in {name}")
        if np.any(np.isnan(vec)):
            raise ValueError(f"NaN value in {name}")
    print(f"Loaded {len(kd_vectors)} KD vectors")

    return kd_vectors


# Use absolute path for the RBM data file
RBM = RBM_utils.loadRBM(str(FILE_DIR / "test_wt_RBM_Covid.data"))

KD_VECTORS = load_kd_vectors(kd_vectors_directory)
ACE2_KD_VECTOR = np.load(str(ace2_vector_directory))

with open(SEQ_DIR / "rbd_wt.fasta") as f:
    f.readline()  # skip header
    WT = f.readline().strip()  # sequence line

WT = WT[BEGIN:-END]
WT_SEQ = Proteins_utils.load_FASTA(str(SEQ_DIR / "rbd_wt.fasta"))[0]
WT_SEQ = WT_SEQ[BEGIN:-END]
