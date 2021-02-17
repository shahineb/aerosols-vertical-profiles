import torch
from gpytorch import variational
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import pop_from_cache_ignore_args
from gpytorch.lazy import SumLazyTensor, MatmulLazyTensor


class VariationalStrategy(variational.VariationalStrategy):
    """Overrides default gpytorch variationals strategy by using low rank
        square root approximation of kernel matrix for scalability"""

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        # Compute distributions
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix
        data_output = self.model.forward(x, **kwargs)
        data_data_covar = data_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter()
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate()

        # Use low rank square root approximation to evaluate input covariance
        data_data_root_covar = data_data_covar.root_decomposition().root
        data_data_covar = MatmulLazyTensor(data_data_root_covar, data_data_root_covar.t())

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.inv_matmul(induc_data_covar.double()).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLazyTensor(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(1e-4).evaluate()
                + interp_term.transpose(-1, -2) @ middle_term.evaluate() @ interp_term
            )
        else:
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)
