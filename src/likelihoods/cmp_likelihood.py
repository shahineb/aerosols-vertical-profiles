import numpy as np
import torch
from gpytorch import lazy
from gpytorch.likelihoods import GaussianLikelihood


class CMPLikelihood(GaussianLikelihood):

    def expected_log_prob(self, observations, variational_dist, root_inv_bags_covar_11, bags_covar_12, transform):
        """Computes 1-sample Monte Carlo esimate of expected loglikelihood under posterior variational distribution
            with reparametrization trick

        Args:
            observations (torch.Tensor): (n, ) tensor of aggregate observations
            variational_dist (gpytorch.distributions.MultivariateNormal): posterior
                variational distribution evaluated on joint individuals
            root_inv_bags_covar_11 (gpytorch.lazy.LazyTensor): (L + NλI)^{-1/2}
            bags_covar_12 (gpytorch.lazy.LazyTensor): l(y, extended_y)

        Returns:
            type: torch.Tensor

        """
        # Extract variational posterior parameters
        variational_mean = variational_dist.mean
        # variational_covar_cholesky = variational_dist.lazy_covariance_matrix.cholesky()
        # variational_covar = variational_dist.lazy_covariance_matrix
        variational_covar_root = lazy.DiagLazyTensor(diag=variational_dist.std)

        # Compute low rank A^T aggregation term, agg_term = l(bags_covariates_2, bags_covariates_1)(L + λNI)^{-1/2}
        agg_term = bags_covar_12.t() @ root_inv_bags_covar_11

        # Sample standard multivariate normal vector
        eps = torch.randn_like(variational_mean)

        # Shift and rescale by posterior variational parameters
        f = variational_mean + variational_covar_root @ eps

        # Apply transformation and aggregate
        agg_transformed_f = agg_term @ root_inv_bags_covar_11.t() @ transform(f)

        # Compute loglikelihood
        constant_term = len(observations) * torch.log(2 * np.pi * self.noise)
        error_term = observations - agg_transformed_f
        error_term = torch.dot(error_term, error_term).div(self.noise)
        output = -0.5 * (constant_term + error_term)
        return output
