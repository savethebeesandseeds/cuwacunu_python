from typing import Iterable

class begin:
  def __init__(self, fun, *args, **kwargs):
    self.fun = fun
    self.args = args
    self.kwargs = kwargs
    self.exception_types_and_handlers = []
    self.finalize = None
  def rescue(self, exception_types, handler):
    if not isinstance(exception_types, Iterable):
      exception_types = (exception_types,)
    self.exception_types_and_handlers.append((exception_types, handler))
    return self
  def ensure(self, finalize, *finalize_args, **finalize_kwargs):
    if self.finalize is not None:
      raise Exception('ensure() called twice')
    self.finalize = finalize
    self.finalize_args = finalize_args
    self.finalize_kwargs = finalize_kwargs
    return self
  def __call__(self):
    try:
      return self.fun(*self.args, **self.kwargs)
    except BaseException as exc:
      handler = self.find_applicable_handler(exc)
      if handler is None:
        raise
      return handler(exc)
    finally:
      if self.finalize is not None:
        self.finalize()
  def find_applicable_handler(self, exc):
    applicable_handlers = (
      handler
      for exception_types, handler in self.exception_types_and_handlers
      if isinstance(exc, exception_types)
    )
    return next(applicable_handlers, None)
#kyrill
# modules = filter(None, (
#   begin(torch.distributions, modname).rescue(lambda exc: None)()
#   for modname in a
# ))
# # print(a)
# print_msg = lambda msg, **print_kwargs: \
#     begin(
#         print, msg, end=' ', **print_kwargs
#     ).\
#     rescue(
#         (TypeError, AttributeError, StopIteration),
#         lambda exc: print(f'just caught "{exc}"! how fun!')
#     ).\
#     ensure(print, 'ok done.')()

import sys
import torch.distributions
# # # Use the webpage api to find everything
# https://pytorch.org/docs/stable/search.html?q=requires_grad&check_keywords=yes&area=default
lam=lambda x : (_ for _ in x.__dict__)
# sys.stdout.write
# sys.stdout.write()
print("(torch.distributions).\n\t",end='')
a=lam(torch.distributions)


import ast

munamunaake_distributions={
'__name__':{'':None},
'__doc__':{'':None},
'__package__':{'':None},
'__loader__':{'':None},
'__spec__':{'':None},
'__path__':{'':None},
'__file__':{'':None},
'__cached__':{'':None},
'__builtins__':{'':None},
'constraints':{'':None},
'utils':{'':None},
'distribution':{'':None},
'exp_family':{'':None},
'bernoulli':{'':None},
'Bernoulli':{'':None},
'dirichlet':{'':None},
'beta':{'':None},
'Beta':{'':None},
'binomial':{'':None},
'Binomial':{'':None},
'categorical':{'':None},
'Categorical':{'':None},
'cauchy':{'':None},
'Cauchy':{'':None},
'gamma':{'':None},
'chi2':{'':None},
'Chi2':{'':None},
'transforms':{'':None},
'constraint_registry':{'':None},
'biject_to':{'':None},
'transform_to':{'':None},
'continuous_bernoulli':{'':None},
'ContinuousBernoulli':{'':None},
'Dirichlet':{'':None},
'Distribution':{'':None},
'ExponentialFamily':{'':None},
'exponential':{'':None},
'Exponential':{'':None},
'fishersnedecor':{'':None},
'FisherSnedecor':{'':None},
'Gamma':{'':None},
'geometric':{'':None},
'Geometric':{'':None},
'uniform':{'':None},
'independent':{'':None},
'transformed_distribution':{'':None},
'gumbel':{'':None},
'Gumbel':{'':None},
'half_cauchy':{'':None},
'HalfCauchy':{'':None},
'normal':{'':None},
'half_normal':{'':None},
'HalfNormal':{'':None},
'Independent':{'':None},
'laplace':{'':None},
'multivariate_normal':{'':None},
'lowrank_multivariate_normal':{'':None},
'one_hot_categorical':{'':None},
'pareto':{'':None},
'poisson':{'':None},
'kl':{'':None},
'kl_divergence':{'':None},
'register_kl':{'':None},
'kumaraswamy':{'':None},
'Kumaraswamy':{'':None},
'Laplace':{'':None},
'lkj_cholesky':{'':None},
'LKJCholesky':{'':None},
'log_normal':{'':None},
'LogNormal':{'':None},
'logistic_normal':{'':None},
'LogisticNormal':{'':None},
'LowRankMultivariateNormal':{'':None},
'mixture_same_family':{'':None},
'MixtureSameFamily':{'':None},
'multinomial':{'':None},
'Multinomial':{'':None},
'MultivariateNormal':{'':None},
'negative_binomial':{'':None},
'NegativeBinomial':{'':None},
'Normal':{'':None},
'OneHotCategorical':{'':None},
'OneHotCategoricalStraightThrough':{'':None},
'Pareto':{'':None},
'Poisson':{'':None},
'relaxed_bernoulli':{'':None},
'RelaxedBernoulli':{'':None},
'relaxed_categorical':{'':None},
'RelaxedOneHotCategorical':{'':None},
'studentT':{'':None},
'StudentT':{'':None},
'TransformedDistribution':{'':None},
'AbsTransform':{'':None},
'AffineTransform':{'':None},
'CatTransform':{'':None},
'ComposeTransform':{'':None},
'CorrCholeskyTransform':{'':None},
'ExpTransform':{'':None},
'IndependentTransform':{'':None},
'LowerCholeskyTransform':{'':None},
'PowerTransform':{'':None},
'ReshapeTransform':{'':None},
'SigmoidTransform':{'':None},
'TanhTransform':{'':None},
'SoftmaxTransform':{'':None},
'StackTransform':{'':None},
'StickBreakingTransform':{'':None},
'Transform':{'':None},
'identity_transform':{'':None},
'Uniform':{'':None},
'von_mises':{'':None},
'VonMises':{'':None},
'weibull':{'':None},
'Weibull':{'':None},
'__all__':{'':None},
}





















torch.distributions.__dict__['beta'].__dict__['Beta'].__dict__['__init__'].__code__
torch.distributions.__dict__['beta'].__dict__['Beta'].__dict__['__init__'].__code__.co_varnames
torch.distributions.__dict__['beta'].__dict__['Beta'].__dict__['__init__'].__code__.co_argcount

































