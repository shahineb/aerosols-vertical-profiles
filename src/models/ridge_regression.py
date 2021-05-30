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
    """Aggregate ridge regression with warping transformation

    Args:
        alpha (float): regularization weight, greater = stronger L2 penalization
        transform (callable): output transformation to apply to prediction
        aggregate_fn (callable): aggregation operator
        ndim (int): dimensionality of inputs
        fit_intercept (bool): if True, pads inputs with constant offset

    """
    def __init__(self, alpha, transform, aggregation_support, aggregate_fn, ndim, fit_intercept=False):
        super().__init__()
        self.alpha = alpha
        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.fit_intercept = fit_intercept
        self.ndim = ndim
        if self.fit_intercept:
            self.bias = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(0.000000001 * torch.randn(self.ndim))
        self.register_buffer('aggregation_support', aggregation_support)

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
        aggregate_prediction = -torch.trapz(y=prediction, x=self.aggregation_support, dim=-2)
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

    def __init__(self, alpha_2d, alpha_3d, aggregate_fn, fit_intercept_2d=False, fit_intercept_3d=False):
        super().__init__()
        self.alpha_2d = alpha_2d
        self.alpha_3d = alpha_3d
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
        aggX = self.aggregate_fn(individuals_covariates)
        upsilon = gpytorch.inv_matmul(Q_2d, bags_covariates.t() @ aggX)
        y_upsilon = bags_covariates @ upsilon

        # Compute second regression stage
        Q_3d = (y_upsilon.t() @ y_upsilon + n_bags * self.alpha_3d * torch.eye(d_3d))
        with gpytorch.settings.cholesky_jitter(1e-3):
            beta = gpytorch.inv_matmul(Q_3d, y_upsilon.t() @ aggregate_targets)
        self.register_buffer('beta', beta)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        if self.fit_intercept_3d:
            x = self.pad_input(x)
        return x @ self.beta


class TwoStagedTransformedAggregateRidgeRegression(nn.Module):
    """Two staged aggregate ridge regression with warping transformation

    Args:
        alpha (float): regularization weight, greater = stronger L2 penalization
        transform (callable): output transformation to apply to prediction
        aggregate_fn (callable): aggregation operator
        ndim (int): dimensionality of inputs
        fit_intercept (bool): if True, pads inputs with constant offset

    """
    def __init__(self, alpha_2d, alpha_3d, transform, aggregation_support, aggregate_fn, ndim, fit_intercept_2d=False, fit_intercept_3d=False):
        super().__init__()
        self.alpha_2d = alpha_2d
        self.alpha_3d = alpha_3d

        self.fit_intercept_2d = fit_intercept_2d
        self.fit_intercept_3d = fit_intercept_3d

        self.ndim = ndim

        if self.fit_intercept_3d:
            self.bias_3d = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(0.000000001 * torch.randn(self.ndim))

        self.transform = transform
        self.aggregate_fn = aggregate_fn
        self.register_buffer('aggregation_support', aggregation_support)

    def pad_input(self, x):
        """Pads x with 1 along last dimension

        Args:
            x (torch.Tensor)

        Returns:
            type: torch.Tensor

        """
        x = torch.cat([x, torch.ones(x.shape[:-1], device=x.device).unsqueeze(-1)], dim=-1)
        return x

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, covariate_dimenionality)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        output = x @ self.beta
        if self.fit_intercept_3d:
            output = output + self.bias_3d
        return self.transform(output)

    def compute_loss(self, aggregate_prediction_2d, bags_covariates, aggregate_targets):
        """Compute mean square error between aggregate prediction and aggregate
        targets using two-staged regression

        Args:
            prediction_3d (torch.Tensor): (n_bags, lev)
            bags_covariates (torch.Tensor): (n_bags, d_2d)
            aggregate_targets (torch.Tensor): (n_bags,)

        Returns:
            type: torch.Tensor

        """
        if self.fit_intercept_2d:
            bags_covariates = self.pad_input(bags_covariates)

        # Extract dimensions
        n_bags, d_2d = bags_covariates.size(0), bags_covariates.size(1)

        # Compute regularized matrix to inverse
        Q_2d = (bags_covariates.t() @ bags_covariates + n_bags * self.alpha_2d * torch.eye(d_2d, device=bags_covariates.device))

        # Compute matrix inverse with matrix-vector multiplication trick
        upsilon = gpytorch.inv_matmul(Q_2d, bags_covariates.t() @ aggregate_prediction_2d)
        y_upsilon = bags_covariates @ upsilon

        # Compute and return mean square error
        difference = aggregate_targets - y_upsilon
        mean_squared_error = torch.square(difference).mean()
        return mean_squared_error

    def aggregate_prediction(self, prediction):
        """Computes aggregation of individuals output prediction

        Args:
            prediction (torch.Tensor): (n_bag, bags_size) tensor output of forward

        Returns:
            type: torch.Tensor

        """
        aggregate_prediction = -torch.trapz(y=prediction, x=self.aggregation_support, dim=-2)
        return aggregate_prediction

    def regularization_term(self):
        """Square L2 norm of beta

        Returns:
            type: torch.Tensor

        """
        return self.alpha_3d * torch.dot(self.beta, self.beta)
