import numpy as np
import torch
from gpytorch import lazy
from gpytorch.likelihoods import GaussianLikelihood


class CMEProcessLikelihood(GaussianLikelihood):

    def expected_log_prob(self, observations, variational_dist, root_inv_extended_bags_covar,
                          bags_to_extended_bags_covar):
        """Computes expected loglikelihood under posterior variational distribution

        Args:
            observations (torch.Tensor): (n, ) tensor of aggregate observations
            variational_dist (gpytorch.distributions.MultivariateNormal): posterior
                variational distribution evaluated on joint individuals
            root_inv_extended_bags_covar (gpytorch.lazy.LazyTensor): (L + NλI)^{-1/2}
            bags_to_extended_bags_covar (gpytorch.lazy.LazyTensor): l(y, extended_y)

        Returns:
            type: torch.Tensor

        """
        # Extract variational posterior parameters
        variational_mean = variational_dist.mean
        variational_root_covar = variational_dist.lazy_covariance_matrix.root_decomposition().root

        # Setup identity lazy tensor for efficient quad computations
        Id_n = lazy.DiagLazyTensor(diag=torch.ones(len(observations)))
        Id_N = lazy.DiagLazyTensor(diag=torch.ones(variational_root_covar.size(-1)))

        # Make bags to extended bags buffer matrix
        buffer = bags_to_extended_bags_covar @ root_inv_extended_bags_covar

        # Compute mean loss term
        mean_term = Id_n.inv_quad(observations - buffer @ root_inv_extended_bags_covar.t() @ variational_mean)

        # Compute covariance loss term
        covar_term = Id_N.inv_quad(variational_root_covar.t() @ root_inv_extended_bags_covar @ buffer.t())

        # Sum up everything to obtain expected logprob under variational distribution
        constant_term = len(observations) * (np.log(2 * np.pi) + torch.log(self.noise))
        output = -0.5 * (constant_term + (mean_term + covar_term) / self.noise)
        return output
