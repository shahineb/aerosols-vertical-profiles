import torch
import torch.nn as nn
import gpytorch


class AggregateRidgeRegression(nn.Module):
    """Ridge Regression model when aggregate targets only are observed

        *** Current implementation assumes all bags have same size ***

    Args:
        alpha (float): regularization weight, greater = stronger L2 penalization
        aggregate_fn (callable): aggregation operator
        fit_intercept (bool): if True, pads inputs with constant offset
    """

    def __init__(self, alpha, aggregate_fn, fit_intercept=False):
        super().__init__()
        self.alpha = alpha
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept

    def pad_input(self, x):
        """Pads x with 1 along last dimension

        Args:
            x (torch.Tensor)

        Returns:
            type: torch.Tensor

        """
        x = torch.cat([x, torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)], dim=-1)
        return x

    def fit(self, individuals_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
                samples must be organized by bags following which aggregation is taken
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        if self.fit_intercept:
            individuals_covariates = self.pad_input(individuals_covariates)
        n_bags = individuals_covariates.size(0)
        d = individuals_covariates.size(-1)

        aggX = self.aggregate_fn(individuals_covariates).t()
        Q = aggX @ aggX.t() + n_bags * self.alpha * torch.eye(d)

        beta = gpytorch.inv_matmul(Q, aggX @ aggregate_targets)
        self.register_buffer('beta', beta)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        if self.fit_intercept:
            x = self.pad_input(x)
        return x @ self.beta


class TransformedAggregateRidgeRegression(nn.Module):
    """Short summary.

    Args:
        alpha (float): regularization weight, greater = stronger L2 penalization
        transform (callable): output transformation to apply to prediction
        aggregate_fn (callable): aggregation operator
        ndim (int): dimensionality of inputs
        fit_intercept (bool): if True, pads inputs with constant offset

    """
    def __init__(self, alpha, transform, aggregate_fn, ndim, fit_intercept=False):
        super().__init__()
        self.alpha = alpha
        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept
        self.ndim = ndim
        if self.fit_intercept:
            self.bias = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(0.000000001 * torch.randn(self.ndim))

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        output = x @ self.beta
        if self.fit_intercept:
            output = output + self.bias
        return self.transform(output)

    def aggregate_prediction(self, prediction):
        """Computes aggregation of individuals output prediction

        Args:
            prediction (torch.Tensor): (n_bag, bags_size) tensor output of forward

        Returns:
            type: torch.Tensor

        """
        aggregate_prediction = self.aggregate_fn(prediction)
        return aggregate_prediction

    def regularization_term(self):
        """Square L2 norm of beta

        Returns:
            type: torch.Tensor

        """
        return self.alpha * torch.dot(self.beta, self.beta)


class TwoStageRidgeRegression2D(nn.Module):
    """Ridge Regression model when aggregate targets only are observed

        *** Current implementation assumes all bags have same size ***

        Two-stage regression procedure:
            (1): Regresses 2D covariates against aggregated predictors
            (2): Regresses first step againt aggregate targetrs

    Args:
        alpha2d (float): regularization weight for first stage, greater = stronger L2 penalization
        alpha3d (float): regularization weight for second stage, greater = stronger L2 penalization
        aggregate_fn (callable): aggregation operator
        fit_intercept_2d (bool): if True, pads 2D inputs with constant offset
        fit_intercept_3d (bool): if True, pads 3D inputs with constant offset
    """

    def __init__(self, alpha2d, alpha3d, aggregate_fn, fit_intercept_2d=False, fit_intercept_3d=False):
        super().__init__()
        self.alpha2d = alpha2d
        self.alpha3d = alpha3d
        self.aggregate_fn = aggregate_fn
        self.fit_intercept_2d = fit_intercept_2d
        self.fit_intercept_3d = fit_intercept_3d

    def pad_input(self, x):
        """Pads x with 1 along last dimension

        Args:
            x (torch.Tensor)

        Returns:
            type: torch.Tensor

        """
        x = torch.cat([x, torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)], dim=-1)
        return x

    def fit(self, individuals_covariates, bags_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
                samples must be organized by bags following which aggregation is taken
            aggregate_targets (torch.Tensor): (n_bags,) of aggregate targets observed for each bag

        """
        if self.fit_intercept_2d:
            bags_covariates = self.pad_input(bags_covariates)
        if self.fit_intercept_3d:
            individuals_covariates = self.pad_input(individuals_covariates)

        # Extract tensors dimensionalities
        n_bags = aggregate_targets.size(0)
        d_2d = bags_covariates.size(-1)
        d_3d = individuals_covariates.size(-1)

        # Compute first regression stage
        Q_2d = (bags_covariates.t() @ bags_covariates + n_bags * self.alpha_2d * torch.eye(d_2d))
        upsilon = Q_2d.inv_matmul(bags_covariates.t())

        # Compute second regression stage
        aggX = self.aggregate_fn(individuals_covariates)
        Q_3d = aggX.t() @ upsilon.t() @  upsilon @ aggX + n_bags * self.alpha_3d * torch.eye(d_3d)
        beta = gpytorch.inv_matmul(Q_3d, aggX.t() @ upsilon.t() @ aggregate_targets)

        self.register_buffer('beta', beta)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        if self.fit_intercept:
            x = self.pad_input(x)
        return x @ self.beta
