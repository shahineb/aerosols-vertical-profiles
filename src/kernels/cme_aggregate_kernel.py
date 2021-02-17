from gpytorch import kernels


class CMEAggregateKernel(kernels.Kernel):
    """Kernel of CME Process using CME to compute aggregation of covariance among
        individuals

    Args:
        bag_kernel (gpytorch.kernels.Kernel): base kernel used for bag values comparison
        bags_values (torch.Tensor): (N,) or (N, d) tensor of bags used for CME estimation
        individuals_covar (gpytorch.lazy.LazyTensor, torch.Tensor): (N, N) tensor of individuals
            covariance matrix used for CME estimation
        root_inv_bags_covar (gpytorch.lazy.LazyTensor, torch.Tensor): (N, ?) usually
            low-rank square root decomposition of inverse covariance
            or precision matrix of bags used for CME estimation
    """
    def __init__(self, bag_kernel, bags_values, individuals_covar, root_inv_bags_covar):
        super().__init__()
        self.bag_kernel = bag_kernel
        self.bags_values = bags_values
        self.individuals_covar = individuals_covar
        self.root_inv_bags_covar = root_inv_bags_covar

    def forward(self, x1, x2, **kwargs):
        """Computes CME aggregate covariance matrix

        Args:
            x1 (torch.Tensor): (N1,) or (N1, d) left tensor of bag values to compute
                the covariance on
            x2 (torch.Tensor): (N2,) or (N2, d) right tensor of bag values to compute
                the covariance on

        Returns:
            type: gpytorch.lazy.LazyTensor (N1, N2)

        """
        # Compute covariance of inputs with the reference bag values used for CME estimate
        bags_to_x1_covar = self.bag_kernel(self.bags_values, x1)
        bags_to_x2_covar = self.bag_kernel(self.bags_values, x2)

        # Derive CME aggregate covariance matrix
        cme_aggregate_covar = self._compute_covar(bags_to_x1_covar=bags_to_x1_covar,
                                                  bags_to_x2_covar=bags_to_x2_covar)

        return cme_aggregate_covar

    def _compute_covar(self, bags_to_x1_covar, bags_to_x2_covar):
        """Runs computation of aggregate CME covariance matrix based on
        covariances of left and right terms with bags used for CME estimation

        Args:
            bags_to_x1_covar (gpytorch.lazy.LazyTensor): (N, N1) left tensor of
                covariance between reference bag values used for CME estimation
                and input bag values
            bags_to_x2_covar (gpytorch.lazy.LazyTensor): (N, N2) right tensor of
                covariance between reference bag values used for CME estimation
                and input bag values

        Returns:
            type: torch.Tensor (N1, N2) tensor of CME aggregate covariance

        """
        # Normalize left and right terms with inverse root bag covariance matrix
        foo_1 = bags_to_x1_covar.t().matmul(self.root_inv_bags_covar)
        foo_2 = self.root_inv_bags_covar.t().matmul(bags_to_x2_covar)

        # Aggregate individuals covariances with normalized bags covariance terms
        bar = self.root_inv_bags_covar.t() @ self.individuals_covar @ self.root_inv_bags_covar
        output = foo_1 @ bar @ foo_2
        return output
