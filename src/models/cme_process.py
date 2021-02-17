from abc import ABC
import torch
from src.means import CMEAggregateMean
from src.kernels import CMEAggregateKernel


class CMEProcess(ABC):
    """General class interface methods common to variations of CME process"""

    def _init_cme_mean_covar_modules(self, individuals, extended_bags_values):
        """Initializes CME aggregate mean and covariance modules based on provided
            individuals and bags values tensors

        Args:
            individuals (torch.Tensor): (N, d) tensor of individuals inputs
            extended_bags_values (torch.Tensor): (N, r) tensor of individuals bags values
        """
        # Evaluate tensors needed to compute CME estimate
        latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar = self._get_cme_estimate_parameters(individuals=individuals,
                                                                                                                   extended_bags_values=extended_bags_values)

        # Initialize CME aggregate mean and covariance functions
        mean_module_kwargs = {'bag_kernel': self.bag_kernel,
                              'bags_values': extended_bags_values,
                              'individuals_mean': latent_individuals_mean,
                              'root_inv_bags_covar': root_inv_bags_covar}
        self.mean_module = CMEAggregateMean(**mean_module_kwargs)

        covar_module_kwargs = {'bag_kernel': self.bag_kernel,
                               'bags_values': extended_bags_values,
                               'individuals_covar': latent_individuals_covar,
                               'root_inv_bags_covar': root_inv_bags_covar}
        self.covar_module = CMEAggregateKernel(**covar_module_kwargs)

    def _get_cme_estimate_parameters(self, individuals, extended_bags_values):
        """Computes tensors required to get an estimation of the CME

        individuals (torch.Tensor): (N, d) tensor of individuals inputs
        extended_bags_values (torch.Tensor): (N, r) tensor of individuals bags values

        Returns:
            type: torch.Tensor, gpytorch.lazy.LazyTensor, gpytorch.lazy.LazyTensor

        """
        # Evaluate underlying GP mean and covariance on individuals
        latent_individuals_mean = self.individuals_mean(individuals)
        latent_individuals_covar = self.individuals_kernel(individuals)

        # Compute precision matrix of bags values
        bags_covar = self.bag_kernel(extended_bags_values)
        foo = bags_covar.add_diag(self.lbda * len(extended_bags_values) * torch.ones(len(extended_bags_values)))
        root_inv_bags_covar = foo.root_inv_decomposition().root
        return latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar

    def update_cme_estimate_parameters(self, individuals, extended_bags_values):
        """Update values of parameters used for CME estimate in mean and
            covariance modules

        individuals (torch.Tensor): (N, d) tensor of individuals inputs
        extended_bags_values (torch.Tensor): (N, r) tensor of individuals bags values

        """
        latent_individuals_mean, latent_individuals_covar, root_inv_bags_covar = self._get_cme_estimate_parameters(individuals=individuals,
                                                                                                                   extended_bags_values=extended_bags_values)
        self.mean_module.root_inv_bags_covar = root_inv_bags_covar
        self.mean_module.bags_values = extended_bags_values
        self.covar_module.individuals_covar = latent_individuals_covar
        self.covar_module.root_inv_bags_covar = root_inv_bags_covar
        self.covar_module.bags_values = extended_bags_values

    def get_individuals_to_cme_covar(self, input_individuals, individuals, bags_values, extended_bags_values):
        """Computes covariance between latent individuals GP evaluated on input
            and CME aggregate process GP distribution on train data

        Args:
            individuals (torch.Tensor): input individuals

        Returns:
            type: torch.Tensor

        """
        individuals_covar_map = self.individuals_kernel(input_individuals, individuals)
        bags_covar = self.bag_kernel(bags_values, extended_bags_values)

        foo = individuals_covar_map.matmul(self.covar_module.root_inv_bags_covar)
        bar = bags_covar.matmul(self.covar_module.root_inv_bags_covar)
        output = foo.matmul(bar.t())
        return output
