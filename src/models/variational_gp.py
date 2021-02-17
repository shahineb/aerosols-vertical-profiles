from gpytorch import distributions
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class VariationalGP(ApproximateGP):
    """Approximate variational GP with inducing points module

    Args:
        inducing_points (torch.Tensor): tensor of landmark points from which to
            compute inducing values
        mean_module (gpytorch.means.Mean): mean module to compute mean vectors on inputs samples
        covar_module (gpytorch.kernels.Kernel): kernel module to compute covar matrix on input samples
    """
    def __init__(self, inducing_points, mean_module, covar_module):
        variational_strategy = self._set_variational_strategy(inducing_points)
        super().__init__(variational_strategy=variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

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
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))

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
        mean_x = self.mean_module(inputs)
        covar_x = self.covar_module(inputs)

        # Build multivariate normal distribution of model evaluated on input samples
        samples_prior_distribution = distributions.MultivariateNormal(mean=mean_x,
                                                                      covariance_matrix=covar_x)
        return samples_prior_distribution

    def __call__(self, inputs, prior=False, **kwargs):
        """Parent class method rewritten here just of the sake of clarity
        Unlike to nn.Module child classes in pytorch, the __call__ method here
        wraps the forward and adds additional steps such that the behavior from a
        __call__ might not match the expected behavior given forward method implementation

        In here:
            if prior=True, then variational strategy executes self.forward
            elif prior=False, then variational strategy executes self.variational_strategy.__call__

        Args:
            inputs (torch.Tensor): input values
            prior (bool): if True, computes prior p(f), else computes variational posterior q(f)

        Returns:
            type: gpytorch.distributions.MultivariateNormal
        """
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(-1)
        return self.variational_strategy(inputs, prior=prior)
