import torch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from .cme_process import CMEProcess
from .cme_prediction_strategy import ExactCMEPredictionStrategy


class ExactCMEProcess(ExactGP, CMEProcess):
    """Class implementing exact formulation of CME Process, suitable for small datasets

    Args:
        train_individuals (torch.Tensor): (N, d) tensor of individuals inputs used
            for training
        train_bags (torch.Tensor): (n, r) tensor of bags values used for training
        train_aggregate_targets (torch.Tensor): (n,) tensor of aggregate values
            observed for each bag
        individuals_mean (gpytorch.means.Mean): mean module used for
            individuals GP prior
        individuals_kernel (gpytorch.kernels.Kernel): covariance module
            used for individuals GP prior
        bag_kernel (gpytorch.kernels.Kernel): kernel module used for bag values
        bags_sizes (list[int]): sizes of bags used
        lbda (float): inversion regularization parameter
        likelihood (gpytorch.likelihoods.Likelihood): observation noise likelihood model

    """
    def __init__(self, train_individuals, train_bags, train_aggregate_targets,
                 individuals_mean, individuals_kernel,
                 bag_kernel, bags_sizes, lbda, likelihood, extended_train_bags=None):

        # Initialize exact GP model attributes
        super().__init__(train_inputs=train_bags,
                         train_targets=train_aggregate_targets,
                         likelihood=likelihood)

        # Setup model attributes
        self.train_individuals = train_individuals
        self.train_bags = train_bags
        self.train_aggregate_targets = train_aggregate_targets
        self.individuals_mean = individuals_mean
        self.individuals_kernel = individuals_kernel
        self.bag_kernel = bag_kernel
        self.bags_sizes = bags_sizes
        self.lbda = lbda
        self.extended_train_bags = torch.cat([bag_value.repeat(bag_size, 1)
                                              for (bag_size, bag_value) in zip(bags_sizes, train_bags)]).squeeze()

        # Initialize CME aggregate mean and covariance functions
        self._init_cme_mean_covar_modules(individuals=self.train_individuals,
                                          extended_bags_values=self.extended_train_bags)

        # Initialize individuals posterior prediction strategy attribute
        self.individuals_prediction_strategy = None

    def setup_individuals_prediction_strategy(self):
        """Defines computational strategy for deriving predictive posterior on individuals
        """
        train_aggregate_prior_dist = self.forward(self.train_bags)
        individuals_prediction_strategy_kwargs = {'train_individuals': self.train_individuals,
                                                  'extended_train_bags': self.extended_train_bags,
                                                  'train_aggregate_prior_dist': train_aggregate_prior_dist,
                                                  'train_aggregate_targets': self.train_aggregate_targets,
                                                  'likelihood': self.likelihood}
        self.individuals_prediction_strategy = ExactCMEPredictionStrategy(**individuals_prediction_strategy_kwargs)

    def update_cme_estimate_parameters(self):
        """Update values of parameters used for CME estimate in mean and
            covariance modules

        individuals (torch.Tensor): (N, d) tensor of individuals inputs
        extended_bags_values (torch.Tensor): (N, r) tensor of individuals bags values

        """
        return super().update_cme_estimate_parameters(individuals=self.train_individuals,
                                                      extended_bags_values=self.extended_train_bags)

    def get_individuals_to_cme_covar(self, input_individuals):
        """Computes covariance between latent individuals GP evaluated on input
            and CME aggregate process GP distribution on train data

        Args:
            input_individuals (torch.Tensor): input individuals

        Returns:
            type: torch.Tensor

        """
        return super().get_individuals_to_cme_covar(input_individuals=input_individuals,
                                                    individuals=self.train_individuals,
                                                    bags_values=self.train_bags,
                                                    extended_bags_values=self.extended_train_bags)

    def forward(self, inputs):
        """CME Process Prior computation

        Args:
            inputs (torch.Tensor): input individuals

        Returns:
            type: gpytorch.distributions.MultivariateNormal

        """
        # Compute mean vector and covariance matrix on input samples
        mean = self.mean_module(inputs)
        covar = self.covar_module(inputs)

        # Build multivariate normal distribution of model evaluated on input samples
        prior_distribution = MultivariateNormal(mean=mean,
                                                covariance_matrix=covar)
        return prior_distribution

    def predict(self, individuals):
        """Run prediction of individuals posterior distribution

        Args:
            individuals (torch.Tensor): input individuals to compute posterior on

        Returns:
            type: gpytorch.distributions.MultivariateNormal

        """
        if self.individuals_prediction_strategy is None:
            self.setup_individuals_prediction_strategy()

        # Compute underlying GP prior mean and covariance on input
        individuals_mean = self.individuals_mean(individuals)
        individuals_covar = self.individuals_kernel(individuals)

        # Compute covariance of latent individual with CME aggregate process
        individuals_to_cme_covar = self.get_individuals_to_cme_covar(individuals)

        # Compute predictive mean and covariance
        individuals_posterior_mean = self.individuals_prediction_strategy.exact_predictive_mean(individuals_mean=individuals_mean,
                                                                                                individuals_to_cme_covar=individuals_to_cme_covar)
        individuals_posterior_covar = self.individuals_prediction_strategy.exact_predictive_covar(individuals_covar=individuals_covar,
                                                                                                  individuals_to_cme_covar=individuals_to_cme_covar)

        # Encapsulate as gaussian vector
        individuals_posterior = MultivariateNormal(mean=individuals_posterior_mean,
                                                   covariance_matrix=individuals_posterior_covar)
        return individuals_posterior
