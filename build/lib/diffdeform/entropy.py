import torch

def entropy(histogram):
    """Compute Shannon Entropy"""
    samples = []
    for sample in histogram:
        # Remove zeros
        sample = sample[sample > 0]
        result = -torch.sum(sample * torch.log2(sample)).unsqueeze(0)
        samples.append(result)
    return torch.cat(samples)