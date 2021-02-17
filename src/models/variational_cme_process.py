import torch
from gpytorch import distributions
from gpytorch.models import ApproximateGP
from gpytorch import variational
from .cme_process import CMEProcess
from src.variational import VariationalStrategy


class VariationalCMEProcess(ApproximateGP, CMEProcess):
    """Sparse variational CME Process

    Args:
        inducing_points (torch.Tensor): tensor of landmark points from which to
            compute inducing values
        individuals_mean (gpytorch.means.Mean): mean module used for
            individuals GP prior
        individuals_kernel (gpytorch.kernels.Kernel): covariance module
            used for individuals GP prior
        bag_kernel (gpytorch.kernels.Kernel): kernel module used for bag values
        lbda (float): inversion regularization parameter
    """
    def __init__(self, inducing_points, individuals_mean, individuals_kernel,
                 bag_kernel, lbda):
        # Initialize variational strategy
        variational_strategy = self._set_variational_strategy(inducing_points)
        super().__init__(variational_strategy=variational_strategy)

        # Initialize CME model modules
        self.individuals_mean = individuals_mean
        self.individuals_kernel = individuals_kernel
        self.bag_kernel = bag_kernel
        self.lbda = lbda

    def _set_variational_strategy(self, inducing_points):
        """Sets variational family of distribution to use and variational approximation
            strategy module

        Args:
            inducing_points (torch.Tensor): tensor of landmark points from which to
                compute inducing values
        Returns:
            type: gpytorch.variational.VariationalStrategy

        """
        # Use gaussian variational family
        variational_distribution = variational.CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))

        # Set default variational approximation strategy
        variational_strategy = VariationalStrategy(model=self,
                                                   inducing_points=inducing_points,
                                                   variational_distribution=variational_distribution,
                                                   learn_inducing_locations=True)
        return variational_strategy

    def forward(self, inputs):
        """Defines prior distribution on input x as multivariate normal N(m(x), k(x, x))

        Args:
            inputs (torch.Tensor): input values

        Returns:
            type: gpytorch.distributions.MultivariateNormal

        """
        # Compute mean vector and covariance matrix on input samples
        mean = self.individuals_mean(inputs)
        covar = self.individuals_kernel(inputs)

        # Build multivariate normal distribution of model evaluated on input samples
        prior_distribution = distributions.MultivariateNormal(mean=mean,
                                                              covariance_matrix=covar)
        return prior_distribution

    def get_elbo_computation_parameters(self, bags_values, extended_bags_values):
        """Computes tensors required to derive expected logprob term in elbo loss

        Args:
            bags_values (torch.Tensor): (n, r) tensor of bags values
            extended_bags_values (torch.Tensor): (N, r) tensor of individuals bags values

        Returns:
            type: gpytorch.lazy.LazyTensor, gpytorch.lazy.LazyTensor

        """
        # Compute (L + λNI)^{-1/2} with L = l(extended_bags, extended_bags)
        N = len(extended_bags_values)
        extended_bags_covar = self.bag_kernel(extended_bags_values).add_diag(self.lbda * N * torch.ones(N))
        root_inv_extended_bags_covar = extended_bags_covar.root_inv_decomposition().root

        # Compute l(bags, extended_bags)
        bags_to_extended_bags_covar = self.bag_kernel(bags_values, extended_bags_values)
        return root_inv_extended_bags_covar, bags_to_extended_bags_covar


class GridVariationalCMEProcess(VariationalCMEProcess):

    def __init__(self, grid_size, grid_bounds, individuals_mean, individuals_kernel,
                 bag_kernel, lbda):
        # Initialize variational strategy
        variational_strategy = self._set_variational_strategy(grid_size, grid_bounds)
        super(VariationalCMEProcess, self).__init__(variational_strategy=variational_strategy)

        # Initialize CME model modules
        self.individuals_mean = individuals_mean
        self.individuals_kernel = individuals_kernel
        self.bag_kernel = bag_kernel
        self.lbda = lbda

    def _set_variational_strategy(self, grid_size, grid_bounds):
        """Sets variational family of distribution to use and variational approximation
            strategy module

        Args:
            grid_size (int): Size of the grid
            grid_bounds (list[tuple[float]]): Bounds of each dimension of the grid
                (should be a list of (float, float) tuples)
        Returns:
            type: gpytorch.variational.VariationalStrategy

        """
        # Use gaussian variational family
        variational_distribution = variational.CholeskyVariationalDistribution(num_inducing_points=grid_size)

        # Set grid variational approximation strategy
        variational_strategy = variational.GridInterpolationVariationalStrategy(model=self,
                                                                                grid_size=grid_size,
                                                                                grid_bounds=grid_bounds,
                                                                                variational_distribution=variational_distribution)
        return variational_strategy
