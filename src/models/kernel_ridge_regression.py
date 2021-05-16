import torch
import torch.nn as nn
from gpytorch import lazy


class AggregateKRR(nn.Module):
    """Aggregate Kernel Ridge Regression with Kernel Mean Embeddings

        *** Current implementation assumes all bags have same size ***

    Args:
        individuals_kernel (gpytorch.kernels.Kernel): kernel for individuals covariates
        alpha (float): ridge regularization weight

    """
    def __init__(self, individuals_kernel, alpha):
        super().__init__()
        self.individuals_kernel = individuals_kernel
        self.alpha = alpha

    def fit(self, individuals_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
            aggregate_targets (torch.Tensor): (n_bags,)

        """
        # Extract tensors dimensions
        n_bags = individuals_covariates.size(0)
        bags_size = individuals_covariates.size(1)
        n_dim = individuals_covariates.size(-1)

        # Register flattened individuals covariates tensor as buffer for prediction
        self.register_buffer('covariates', individuals_covariates)

        # Compute aggregated kernel matrix ATKA - takes sooo long because of .evaluate()
        K = self.individuals_kernel(individuals_covariates.view(-1, n_dim)).evaluate()
        agg_K = lazy.lazify(K.reshape(n_bags, bags_size, n_bags, bags_size).mean(dim=(1, 3)))

        # Compute inverse term ATKA + αnI
        inverse_term = agg_K.add_diag(n_bags * self.alpha * torch.ones_like(aggregate_targets))

        # Derive interpolation coefficients
        beta = inverse_term.inv_matmul(aggregate_targets)
        self.register_buffer('beta', beta)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, n_dim_individuals)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor

        """
        K = self.individuals_kernel(self.covariates, x)
        K = K.sum(dim=1).mul(1 / self.covariates.size(1)).t()
        return K @ self.beta


class ConditionalAggregateKRR(nn.Module):
    """Aggregate Kernel Ridge Regression with Kernel Conditional Mean Embeddings

        *** Current implementation assumes all bags have same size ***

    Args:
        individuals_kernel (gpytorch.kernels.Kernel): kernel for individuals covariates
        bags_kernel (gpytorch.kernels.Kernel): kernel for bags covariates
        lbda (float): regularization weight for conditional mean operator estimation
        alpha (float): ridge regularization weight

    """
    def __init__(self, individuals_kernel, bags_kernel, lbda, alpha):
        super().__init__()
        self.individuals_kernel = individuals_kernel
        self.bags_kernel = bags_kernel
        self.lbda = lbda
        self.alpha = alpha

    def fit(self, individuals_covariates, bags_covariates, aggregate_targets):
        """Fits model following sklearn syntax

        Args:
            individuals_covariates (torch.Tensor): (n_bags, bags_size, n_dim_individuals)
            bags_covariates (torch.Tensor): (n_bags, n_dim_bags)
            aggregate_targets (torch.Tensor): (n_bags)

        """
        # Extract tensors dimensions
        n_bags = individuals_covariates.size(0)
        bags_size = individuals_covariates.size(1)
        n_sample = n_bags * bags_size

        # Register flattened individuals covariates tensor as buffer for prediction
        self.register_buffer('covariates', individuals_covariates.view(n_sample, -1))

        # Replicate bags covariates to match size of individuals tensor
        extended_bags_covariates = bags_covariates.unsqueeze(1).repeat(1, bags_size, 1).view(n_sample, -1)

        # Compute individuals gram matrix K and extended bags gram matrix L
        K = self.individuals_kernel(self.covariates)
        L = self.bags_kernel(extended_bags_covariates)

        # Compute transition bags gram matrix l(bags, extended_bags)
        L_bags_to_extended_bags = self.bags_kernel(bags_covariates, extended_bags_covariates)

        # Derive low rank unwhitened aggregation matrix A = l(bags, extended_bags) * (L + λNI)^{-1-2}
        root_inv_L = L.add_diag(self.lbda * n_sample * torch.ones(n_sample, device=individuals_covariates.device)).root_inv_decomposition().root
        A = L_bags_to_extended_bags @ root_inv_L

        # Compute inverse term A^T(L + λNI)^{-1-2}K(L + λNI)^{-1-2}A + αnI
        inverse_term = A @ (root_inv_L.t() @ K @ root_inv_L) @ A.t()
        inverse_term = inverse_term.add_diag(n_bags * self.alpha * torch.ones_like(aggregate_targets))

        # Derive interpolation coefficients
        beta = root_inv_L @ A.t() @ inverse_term.inv_matmul(aggregate_targets)
        self.register_buffer('beta', beta)

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, n_dim_individuals)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor (n_samples,)

        """
        K = self.individuals_kernel(x, self.covariates)
        return K @ self.beta


class TransformedConditionalAggregateKRR(nn.Module):
    """Transformed Aggregate Kernel Ridge Regression with Kernel Conditional Mean Embeddings

        Same as above except that output undergoes possibly non-linear transformation
        and closed form solution may not be available - i.e. no fit method

    Args:
        individuals_kernel (gpytorch.kernels.Kernel): kernel for individuals covariates
        bags_kernel (gpytorch.kernels.Kernel): kernel for bags covariates
        transform (callable): output transformation to apply to prediction
        individuals_covariates (torch.Tensor): (N, d) tensor of interpolation covariates
        lbda (float): regularization weight for conditional mean operator estimation
        alpha (float): ridge regularization weight
    """
    def __init__(self, individuals_kernel, bags_kernel, transform, individuals_covariates, lbda, alpha):
        super().__init__()
        self.individuals_kernel = individuals_kernel
        self.bags_kernel = bags_kernel
        self.transform = transform
        self.lbda = lbda
        self.alpha = alpha
        self.beta = nn.Parameter(torch.rand(individuals_covariates.size(0)))
        self.register_buffer('individuals_covariates', individuals_covariates)
        with torch.no_grad():
            self.register_buffer('K', self.individuals_kernel(individuals_covariates).evaluate())

    def forward(self, x):
        """Runs prediction

        Args:
            x (torch.Tensor): (n_samples, n_dim_individuals)
                samples must not need to be organized by bags

        Returns:
            type: torch.Tensor (n_samples,)

        """
        K = self.individuals_kernel(x, self.individuals_covariates)
        prediction = self.transform(K @ self.beta)
        return prediction

    def aggregate_prediction(self, prediction, bags_covariates, extended_bags_covariates):
        """Computes aggregation of individuals output prediction as

            aggregate_prediction = l(bags, extended_bags) * (L + λNI)^{-1} * pred

        Args:
            prediction (torch.Tensor): (N,) tensor output of forward
            bags_covariates (torch.Tensor): (n,) tensor of bags covariates corresponding to individuals
                used for prediction
            extended_bags_covariates (torch.Tensor): (N,) tensor of above bags covariates replicated
                to match individuals tensor size

        Returns:
            type: torch.Tensor

        """
        # Compute extended bags gram matrix L
        L = self.bags_kernel(extended_bags_covariates)

        # Compute transition bags gram matrix l(bags, extended_bags)
        L_bags_to_extended_bags = self.bags_kernel(bags_covariates, extended_bags_covariates)

        # Derive low rank unwhitened aggregation matrix A = l(bags, extended_bags) * (L + λNI)^{-1/2}
        N = extended_bags_covariates.size(0)
        root_inv_L = L.add_diag(self.lbda * N * torch.ones(N, device=prediction.device)).root_inv_decomposition().root
        A = L_bags_to_extended_bags @ root_inv_L

        # Aggregate prediction
        aggregate_prediction = A @ root_inv_L.t() @ prediction
        return aggregate_prediction

    def regularization_term(self):
        regularization_term = self.alpha * torch.dot(self.beta, self.K @ self.beta)
        return regularization_term
