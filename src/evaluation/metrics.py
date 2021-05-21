import torch


def compute_scores(prediction_3d, groundtruth_3d, targets_2d, aggregate_fn):
    """Computes prediction scores

    Args:
        prediction_3d (torch.Tensor): (time, lat, lon, lev)
        groundtruth_3d (torch.Tensor): (time, lat, lon, lev)
        targets_2d (torch.Tensor): (time, lat, lon)
        aggregate_fn (callable): callable used to aggregate (time, lat, lon, lev, -1) -> (time, lat, lon)

    Returns:
        type: Description of returned object.

    """
    scores_2d = compute_2d_aggregate_metrics(prediction_3d, targets_2d, aggregate_fn)
    scores_3d = compute_3d_metrics(prediction_3d, groundtruth_3d)
    output = {'2d': scores_2d, '3d': scores_3d}
    return output


def compute_2d_aggregate_metrics(prediction_3d, targets_2d, aggregate_fn):
    """Computes prediction scores between aggregation of 3D+t prediction and
    2D+t aggregate targets used for training

    Args:
        prediction_3d (torch.Tensor): (time, lat, lon, lev)
        targets_2d (torch.Tensor): (time, lat, lon)
        aggregate_fn (callable): callable used to aggregate (time, lat, lon, lev, -1) -> (time, lat, lon)

    Returns:
        type: dict[float]

    """
    aggregate_prediction_2d = aggregate_fn(prediction_3d.unsqueeze(-1)).squeeze()
    difference = aggregate_prediction_2d.sub(targets_2d)
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()
    corr = spearman_correlation(aggregate_prediction_2d.flatten(), targets_2d.flatten())
    output = {'rmse': rmse.item(), 'mae': mae.item(), 'corr': corr.item()}
    return output


def compute_3d_metrics(prediction_3d, groundtruth_3d):
    """Computes prediction scores between 3D+t prediction and 3D+t unobserved groundtruth

    Args:
        prediction_3d (torch.Tensor): (time, lat, lon, lev)
        groundtruth_3d (torch.Tensor): (time, lat, lon, lev)

    Returns:
        type: dict[float]

    """
    difference = prediction_3d.sub(groundtruth_3d)
    rmse = torch.square(difference).mean().sqrt()
    mae = torch.abs(difference).mean()
    corr = spearman_correlation(prediction_3d.flatten(), groundtruth_3d.flatten())
    output = {'rmse': rmse.item(), 'mae': mae.item(), 'corr': corr.item()}
    return output


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
