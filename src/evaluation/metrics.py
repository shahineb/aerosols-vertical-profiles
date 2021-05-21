import torch


def spearman_correlation(x, y):
    """Computes Spearman Correlation between x and y

    Args:
        x (torch.Tensor)
        y (torch.Tensor)

    Returns:
        type: torch.Tensor

    """
    x_rank = _get_ranks(x)
    y_rank = _get_ranks(y)
    n = x.size(0)
    numerator = 6 * torch.sum((x_rank - y_rank).pow(2))
    denominator = n * (n ** 2 - 1.0)
    return 1.0 - (numerator / denominator)


def _get_ranks(x):
    """Computes ranking of elements in tensor

    Args:
        x (torch.Tensor)

    Returns:
        type: torch.Tensor

    """
    sorted_indices = x.argsort()
    ranks = torch.zeros_like(sorted_indices)
    ranks[sorted_indices] = torch.arange(len(x)).to(x.device)
    return ranks
