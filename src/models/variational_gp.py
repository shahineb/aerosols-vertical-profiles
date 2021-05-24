import torch
from gpytorch import distributions
from gpytorch.models import ApproximateGP
from gpytorch import variational


class VariationalCIP(ApproximateGP):
    """Sparse Variational Kernel Conditional Mean Process

    Args:
        inducing_points (torch.Tensor): tensor of landmark points from which to
            compute inducing values
        individuals_mean (gpytorch.means.Mean): mean module used for
            individuals GP prior
        individuals_kernel (gpytorch.kernels.Kernel): covariance module
            used for individuals GP prior
        bag_kernel (gpytorch.kernels.Kernel): kernel module used for bag values
        transform (callable): output transformation to apply to prediction
        lbda (float): CMO inversion regularization parameter

    """
    def __init__(self, inducing_points, individuals_mean, individuals_kernel, bag_kernel, transform, lbda):
        # Initialize variational strategy
        variational_strategy = self._set_variational_strategy(inducing_points)
        super().__init__(variational_strategy=variational_strategy)

        # Initialize CME model modules
        self.individuals_mean = individuals_mean
        self.individuals_kernel = individuals_kernel
        self.bag_kernel = bag_kernel
        self.noise_kernel = None
        self.lbda = lbda

        # Register transform
        self.transform = transform

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
        variational_strategy = variational.VariationalStrategy(model=self,
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

    def get_elbo_computation_parameters(self, bags_covariates_1, bags_covariates_2):
        """Computes tensors required to derive expected logprob term in elbo loss

        Args:
            bags_covariates_1 (torch.Tensor): (n1, r) tensor of bags-level covariates used for CMO estimation
            bags_covariates_2 (torch.Tensor): (n2, r) tensor of bags-level covariates used for DMO estimation

        Returns:
            type: gpytorch.lazy.LazyTensor, gpytorch.lazy.LazyTensor

        """
        # Compute (L + λNI)^{-1/2} with L = l(bags_covariates_1, bags_covariates_1)
        n1 = len(bags_covariates_1)
        bags_covar_11 = self.bag_kernel(bags_covariates_1).add_diag(self.lbda * n1 * torch.ones(n1, device=bags_covariates_1.device))
        root_inv_bags_covar_11 = bags_covar_11.root_inv_decomposition().root

        # Compute l(bags_covariates_1, bags_covariates_2)
        bags_covar_12 = self.bag_kernel(bags_covariates_1, bags_covariates_2)

        # Record in output dictionnary
        output = {'root_inv_bags_covar_11': root_inv_bags_covar_11,
                  'bags_covar_12': bags_covar_12}
        return output
