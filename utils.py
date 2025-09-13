import numpy as np


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


def get_deltaG(seq, kd_vectors):
    """
    For a given sequence seq (one-hot encoded), return an array of deltaG values for each antibody.
    seq: One-hot encoded sequence as a NumPy array of shape (1, n_features) or (n_features,).
    kd_vectors: Dictionary {antibody_name: q_vector}, where q_vector = [coeffs..., intercept].
    """
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)  # make 2D

    deltaG_values = []
    try:
        for antibody_name, q in kd_vectors.items():
            coeff = np.array(q[:-1], dtype=np.float64)
            intercept = float(q[-1])
            deltaG = np.dot(seq, coeff) + intercept  # shape (1,)
            deltaG_values.append(deltaG.item())
    except:
        q = kd_vectors
        coeff = np.array(q[:-1], dtype=np.float64)
        intercept = float(q[-1])
        deltaG = np.dot(seq, coeff) + intercept  # shape (1,)
        deltaG_values.append(deltaG)

    return np.array(deltaG_values)


def one_hot_encode_concat(s):
    """
    One-hot encode a sequence of integers (values 0..20).
    Returns a flattened 1D NumPy array of length len(s)*21.
    """
    num_categories = 21
    one_hot_matrix = np.zeros((len(s), num_categories), dtype=np.float64)
    one_hot_matrix[np.arange(len(s)), s] = 1.0
    return one_hot_matrix.flatten()


def get_Kd(s, kd_vectors, log10=False):
    """
    Compute Kd for a given sequence s.
    s: array of ints (sequence positions encoded as 0..20)
    kd_vectors: dict of q vectors
    log10: if True, return log10(Kd) instead of Kd
    """
    one_hot_s = one_hot_encode_concat(s).reshape(1, -1)  # [1, 3738]

    if log10:
        return get_deltaG(one_hot_s, kd_vectors) / np.log(10)
    else:
        return np.exp(get_deltaG(one_hot_s, kd_vectors))
