import torch
import gpytorch


class DisaggregateRidgeRegression:
    """Ridge Regression model when aggregate targets only are observed

    Args:
        alpha (float): regularization weight, greater = stronger L2 penalization
    """

    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            covariates (torch.Tensor): (n_bags, n_sample_per_bag, covariate_dimenionality)
                samples must be organized by bags following which aggregation is taken
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        n = covariates.size(0)
        d = covariates.size(-1)

        EXy = covariates.mean(dim=1).t()
        Q = EXy @ EXy.t() + n * self.alpha * torch.eye(d)

        self.beta = gpytorch.inv_matmul(Q, EXy @ aggregate_targets)

    def predict(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        return x @ self.beta
