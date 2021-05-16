from .ridge_regression import AggregateRidgeRegression
from .kernel_ridge_regression import AggregateKRR, ConditionalAggregateKRR, TransformedConditionalAggregateKRR
from .variational_gp import VariationalCMP

__all__ = ['AggregateKRR', 'ConditionalAggregateKRR',
           'AggregateRidgeRegression', 'TransformedConditionalAggregateKRR',
           'VariationalCMP']
