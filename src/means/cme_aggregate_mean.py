from gpytorch import means


class CMEAggregateMean(means.Mean):
    """Short summary.

    Args:
        bag_kernel (gpytorch.kernels.Kernel): base kernel used for bag values comparison
        bags_values (torch.Tensor): (N,) or (N, d) tensor of bags used for CME estimation
        individuals_mean (gpytorch.lazy.LazyTensor, torch.Tensor): (N, ) tensor of individuals
            mean used for CME estimation
        root_inv_bags_covar (gpytorch.lazy.LazyTensor, torch.Tensor): (N, ?) usually
            low-rank square root decomposition of inverse covariance
            or precision matrix of bags used for CME estimation

    """
    def __init__(self, bag_kernel, bags_values, individuals_mean, root_inv_bags_covar):
        super().__init__()
        self.bag_kernel = bag_kernel
        self.bags_values = bags_values
        self.individuals_mean = individuals_mean
        self.root_inv_bags_covar = root_inv_bags_covar

    def forward(self, x):
        """Compute CME aggregate mean

        Args:
            x (torch.Tensor): (N',) or (N', d) tensor of bags values to compute
                the mean on

        Returns:
            type: torch.Tensor (N',)

        """
        # Compute covariance of inputs with the reference bag values used for CME estimate
        bag_to_x_covar = self.bag_kernel(self.bags_values, x)

        # Derive CME aggregate mean vector
        cme_aggregate_mean = self._compute_mean(bag_to_x_covar=bag_to_x_covar)
        return cme_aggregate_mean

    def _compute_mean(self, bag_to_x_covar):
        """
        Args:
            bag_to_x_covar (gpytorch.lazy.LazyTensor): (N, N') tensor of
                covariance between reference bag values used for CME estimation
                and input bag values

        Returns:
            type: (torch.Tensor): (N',) tensor of CME aggregate mean
        """
        foo = bag_to_x_covar.t().matmul(self.root_inv_bags_covar)
        bar = self.root_inv_bags_covar.t().matmul(self.individuals_mean)
        output = foo.matmul(bar)
        return output
