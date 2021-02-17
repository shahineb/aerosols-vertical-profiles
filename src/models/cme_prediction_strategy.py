import functools
import torch
from gpytorch import settings, lazy
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.utils.memoize import cached, clear_cache_hook


class ExactCMEPredictionStrategy(DefaultPredictionStrategy):
    """Implements computation of predictive posterior on individuals distribution

    Args:
        train_individuals (torch.Tensor): (N, d) tensor of individuals inputs used
            for training
        extended_train_bags (torch.Tensor): (N, r) tensor of bags values used for training
        train_aggregate_prior_dist (gpytorch.distribution.MultivariateNormal): CME process
            prior distribution on training samples
        train_aggregate_targets (torch.Tensor): (n,) tensor of aggregate values
        observed for each bag
        likelihood (gpytorch.likelihoods.Likelihood): observation noise likelihood model

    """
    def __init__(self, train_individuals, extended_train_bags, train_aggregate_prior_dist,
                 train_aggregate_targets, likelihood):
        super().__init__(train_inputs=extended_train_bags,
                         train_prior_dist=train_aggregate_prior_dist,
                         train_labels=train_aggregate_targets,
                         likelihood=likelihood)
        self.train_individuals = train_individuals

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        raise NotImplementedError

    @property
    @cached(name="covar_inv_root_cache")
    def covar_inv_root_cache(self):
        """Computes cached inverse square root of model likelihood covariance matrix

        Returns:
            type: gpytorch.lazy.LazyTensor

        """
        train_bags_covar = self.lik_train_train_covar
        train_bags_covar_inv_root = lazy.delazify(train_bags_covar.root_inv_decomposition().root)
        train_bags_covar_inv_root = self._handle_cache_autograd(train_bags_covar_inv_root)
        return train_bags_covar_inv_root

    @property
    @cached(name="posterior_mean_correction_cache")
    def posterior_mean_correction_cache(self):
        """Computes cached posterior mean correction based on model train likelihood

        Returns:
            type: torch.Tensor

        """
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar = mvn.loc, mvn.lazy_covariance_matrix

        train_labels_offset = (self.train_labels - train_mean).unsqueeze(-1)
        mean_cache = train_train_covar.evaluate_kernel().inv_matmul(train_labels_offset).squeeze(-1)

        mean_cache = self._handle_cache_autograd(mean_cache)
        return mean_cache

    def _handle_cache_autograd(self, cache_input):
        """Makes sure gradients are properly tracked or not depending on context

        Args:
            cache_input (torch.Tensor, gpytorch.lazy.LazyTensor)

        Returns:
            type: torch.Tensor, gpytorch.lazy.LazyTensor

        """
        if settings.detach_test_caches.on():
            cache_input = cache_input.detach()

        if cache_input.grad_fn is not None:
            wrapper = functools.partial(clear_cache_hook, self)
            functools.update_wrapper(wrapper, clear_cache_hook)
            cache_input.grad_fn.register_hook(wrapper)

        return cache_input

    def exact_prediction(self, individuals_mean, individuals_covar,
                         individuals_to_cme_covar):
        """Compute exact predictive mean and covariance given input test individuals

        Args:
            individuals_mean (torch.Tensor)
            individuals_covar (torch.Tensor, gpytorch.lazy.LazyTensor)
            individuals_to_cme_covar (torch.Tensor, gpytorch.lazy.LazyTensor)

        Returns:
            type: torch.Tensor, gpytorch.lazy.LazyTensor

        """
        return (self.exact_predictive_mean(individuals_mean, individuals_to_cme_covar),
                self.exact_predictive_covar(individuals_covar, individuals_to_cme_covar))

    def exact_predictive_mean(self, individuals_mean, individuals_to_cme_covar):
        """Computes exact predictive mean given input test individuals

        Args:
            individuals_mean (torch.Tensor)
            individuals_to_cme_covar (torch.Tensor, gpytorch.lazy.LazyTensor)

        Returns:
            type: torch.Tensor

        """
        output = (individuals_to_cme_covar @ self.posterior_mean_correction_cache.unsqueeze(-1)).squeeze(-1)
        output = output + individuals_mean
        return output

    def exact_predictive_covar(self, individuals_covar, individuals_to_cme_covar):
        """Computes exact predictive covariance given input test individuals

        Args:
            individuals_covar (torch.Tensor, gpytorch.lazy.LazyTensor)
            individuals_to_cme_covar (torch.Tensor, gpytorch.lazy.LazyTensor)

        Returns:
            type: gpytorch.lazy.LazyTensor

        """
        # Full computation without using cached matrix
        if settings.fast_pred_var.off():
            output = self._compute_predictive_covar(individuals_covar, individuals_to_cme_covar)

        # Using cached inverse square root covariance matrix
        else:
            covar_inv_quad_form_root = self.covar_cache.matmul(individuals_to_cme_covar)
            if torch.is_tensor(individuals_covar):
                output = torch.add(individuals_covar, covar_inv_quad_form_root @ covar_inv_quad_form_root.transpose(-1, -2), alpha=-1)
                output = lazy.lazify(output)
            else:
                output = individuals_covar + lazy.MatmulLazyTensor(covar_inv_quad_form_root, covar_inv_quad_form_root.transpose(-1, -2).mul(-1))
        return output

    def _compute_predictive_covar(self, individuals_covar, individuals_to_cme_covar):
        """Runs full computation of exact predictive covariance matrix as efficiently
        as possible depending on input data type

        Args:
            individuals_covar (torch.Tensor, gpytorch.lazy.LazyTensor)
            individuals_to_cme_covar (torch.Tensor, gpytorch.lazy.LazyTensor)

        Returns:
            type: gpytorch.lazy.LazyTensor

        """
        # Derive noisy model prior covariance of cme on training bags
        mvn = self.train_prior_dist.__class__(
            torch.zeros_like(self.train_prior_dist.mean), self.train_prior_dist.lazy_covariance_matrix
        )
        if settings.detach_test_caches.on():
            train_cme_aggregate_covar = self.likelihood(mvn, self.train_inputs).lazy_covariance_matrix.detach()
        else:
            train_cme_aggregate_covar = self.likelihood(mvn, self.train_inputs).lazy_covariance_matrix

        # Compute inverse model covariance X train bags to individuals covariance
        individuals_to_cme_covar = lazy.delazify(individuals_to_cme_covar)
        covar_correction_rhs = train_cme_aggregate_covar.inv_matmul(individuals_to_cme_covar.transpose(-1, -2))

        # For efficiency
        if torch.is_tensor(individuals_covar):
            # We can use addmm in the 2d case
            if individuals_covar.dim() == 2:
                output = torch.addmm(individuals_covar, individuals_to_cme_covar, covar_correction_rhs, beta=1, alpha=-1)
                output = lazy.lazify(output)
            else:
                output = lazy.lazify(individuals_covar + individuals_to_cme_covar @ covar_correction_rhs.mul(-1))

        # In other cases - we'll use the standard infrastructure
        else:
            output = individuals_covar + lazy.MatmulLazyTensor(individuals_to_cme_covar, covar_correction_rhs.mul(-1))
        return output
