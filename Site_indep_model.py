import torch
import torch.nn as nn


class SiteIndependentModel(nn.Module):
    def __init__(self, beta=1):
        super(SiteIndependentModel, self).__init__()
        self.site_probabilities = None
        self.beta = beta

    def fit(self, sequences, frequencies=None):
        num_sequences, sequence_length = sequences.shape
        num_values = 21  # Only 21 possibilities per site (0 to 20)

        # If frequencies are not provided, assume uniform frequencies
        if frequencies is None:
            frequencies = torch.ones(num_sequences)

        # Ensure sequences and frequencies are tensors and of compatible types
        sequences = (
            sequences.long()
        )  # Ensure sequences use a type suitable for indexing
        frequencies = frequencies.float().unsqueeze(
            1
        )  # Make frequencies a column vector for broadcasting

        site_counts = torch.zeros((sequence_length, num_values), dtype=torch.float)

        # Accumulate weighted counts for each value at each site
        for value in range(num_values):
            mask = sequences == value  # Create a mask for the current value
            weighted_counts = mask.float() * frequencies  # Apply frequencies as weights
            site_counts[:, value] = weighted_counts.sum(
                dim=0
            )  # Sum weighted counts across sequences

        # Normalize counts to probabilities by dividing by the sum of frequencies
        total_frequencies = frequencies.sum()
        self.site_probabilities = site_counts / total_frequencies

        # Convert probabilities to log probabilities to avoid numerical underflow
        self.site_probabilities = torch.log(self.site_probabilities + 1e-9)

    def forward(self, sequence):
        if self.site_probabilities is None:
            raise ValueError("Model has not been fitted to data.")
        sequence = sequence.long()  # Convert sequence to long if not already
        # IndexError: tensors used as indices must be long, int, byte or bool tensors
        # sequence_log_probs = self.site_probabilities[torch.arange(sequence.size(0), dtype=torch.long), sequence]
        sequence_log_probs = self.site_probabilities[
            torch.arange(sequence.size(0), dtype=torch.long), sequence
        ]
        log_probability = sequence_log_probs.sum() * self.beta

        return -log_probability  # Note: Returning negative log probability

    def sample(self, n=1):
        """
        Sample n sequences according to the site-independent model.

        Parameters:
        - n: The number of sequences to sample.

        Returns:
        - sequences: A tensor of sampled sequences.
        """
        if self.site_probabilities is None:
            raise ValueError("Model has not been fitted to data.")

        # Convert log probabilities back to probabilities for sampling
        site_probabilities = torch.exp(self.site_probabilities)

        sequence_length, num_values = site_probabilities.shape
        sequences = torch.zeros((n, sequence_length), dtype=torch.long)

        for i in range(sequence_length):
            # Compute the cumulative distribution function (CDF) for each site
            cdf = torch.cumsum(site_probabilities[i, :], dim=0)

            # Generate random values for each sequence and find the corresponding amino acid
            random_values = torch.rand(n, 1)
            amino_acids = torch.searchsorted(cdf, random_values).squeeze(1)

            sequences[:, i] = amino_acids
            # if first dimension is 1, remove it
        # type torch int 16
        sequences = sequences.type(torch.int16)

        if n == 1:
            sequences = sequences.squeeze(0)

        return sequences


class RandomSiteIndependentModel(nn.Module):
    def __init__(self, sequence_length=178, beta=1):
        super(RandomSiteIndependentModel, self).__init__()
        self.sequence_length = sequence_length
        self.num_values = 21  # 21 possibilities per site (0 to 20)
        self.beta = beta
        self.site_probabilities = torch.full(
            (sequence_length, self.num_values), 1.0 / self.num_values
        )

    def fit(self, sequences=None, frequencies=None):
        # No fitting needed for the random model as probabilities are uniform
        pass

    def forward(self, sequence):
        if self.site_probabilities is None:
            raise ValueError("Model has not been initialized properly.")
        sequence = sequence.long()  # Convert sequence to long if not already

        # Calculate log probabilities for the sequence
        sequence_log_probs = self.site_probabilities[
            torch.arange(sequence.size(0), dtype=torch.long), sequence
        ]
        log_probability = torch.log(sequence_log_probs).sum() * self.beta

        return -log_probability  # Note: Returning negative log probability

    def sample(self, n=1):
        """
        Sample n sequences according to the site-independent model with equal probabilities.

        Parameters:
        - n: The number of sequences to sample.

        Returns:
        - sequences: A tensor of sampled sequences.
        """
        if self.site_probabilities is None:
            raise ValueError("Model has not been initialized properly.")

        sequence_length, num_values = self.site_probabilities.shape
        sequences = torch.zeros((n, sequence_length), dtype=torch.long)

        for i in range(sequence_length):
            # Generate random values for each site
            random_values = torch.randint(0, num_values, (n,))
            sequences[:, i] = random_values

        sequences = sequences.type(torch.int16)

        if n == 1:
            sequences = sequences.squeeze(0)

        return sequences
