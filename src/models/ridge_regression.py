import torch
import torch.nn as nn
import gpytorch


class AggregateRidgeRegression(nn.Module):
    """Ridge Regression model when aggregate targets only are observed

        *** Current implementation assumes all bags have same size ***

    Args:
        alpha (float): regularization weight, greater = stronger L2 penalization
    """

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def fit(self, individuals_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
                samples must be organized by bags following which aggregation is taken
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        n_bags = individuals_covariates.size(0)
        d = individuals_covariates.size(-1)

        EXy = individuals_covariates.mean(dim=1).t()
        Q = EXy @ EXy.t() + n_bags * self.alpha * torch.eye(d)

        beta = gpytorch.inv_matmul(Q, EXy @ aggregate_targets)
        self.register_buffer('beta', beta)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        return x @ self.beta
