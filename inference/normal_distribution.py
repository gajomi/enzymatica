import numpy as np
import scipy.optimize as optimize
from scipy.stats import norm
from scipy import integrate
import pylab as plt
from collections import namedtuple
from inference import RandomVariable
from functools import partial

class NormalDistribution(RandomVariable):
  def __init__(self, mean, variance, **kwargs):
    if type(mean) == type(0.0) or type(mean) == type(0):
      self.__pdf_function = partial(norm.pdf, loc = mean, scale = np.sqrt(variance))
      self.__cdf_function = partial(norm.cdf, loc = mean, scale = np.sqrt(variance))
      self._dimensions = 1
    else:
      # This now becomes a multivariate normal distribution.
      self.__pdf_function = partial(norm.pdf, loc = 0, scale = 1)
      self.__cdf_function = partial(norm.cdf, loc = 0, scale = 1)
      self._dimensions = len(mean)
    
    super(NormalDistribution, self).__init__(self.__pdf_function, **kwargs)
    self._mean = mean
    self._variance = variance
    self._sigma = np.sqrt(variance)

  def cdf(self, x = None, interval = None):
    """ see enzymatica.inference.random_variable#cdf """
    if interval is not None:
      return norm.cdf(event.upper, self._mean, np.sqrt(self._variance)) \
          - norm.cdf(event.lower, self._mean, np.sqrt(self._variance))
    elif x is not None:
      return 0.

  def expectation(self, interval = None):
    """ see enzymatica.inference.random_variable#expectation. """
    if interval is None:
      return self._mean
    else:
      if self._dimensions == 1:
        expected_point_fn = lambda x: x * self.__pdf_function(x)
        weighted_sum = integrate.quad(expected_point_fn, interval[0], interval[1])[0]
        cumulative_density = self.__cdf_function(interval[1]) - self.__cdf_function(interval[0])
        return weighted_sum / cumulative_density
      else:
        error_msg = 'Expected value of an interval for a '
        error_msg += 'multivariate normal distribution is not supported'
        raise NotImplementedError(error_msg)
  
  def likelihood(self, x):
    if self._dimensions > 1:
      # Assuming the variables are independent of each other, we can multiply the probabilities
      # to find the likelihood.
      values = np.array([norm.pdf(x_i, mu_i, self._sigma) for x_i, mu_i in zip(x, self._mean)])
      log_values = np.array([np.log(value) for value in values])
      return np.exp(np.sum(log_values))
    else:
      return norm.pdf(x, self.mean, np.sqrt(self.variance))

  def log_likelihood(self, x):
    if self._dimensions > 1:
      # Assuming the variables are independent of each other, we can multiply the probabilities
      # to find the likelihood.
      return -np.sum((x - self._mean) ** 2)
    else:
      return np.log(norm.pdf(x, self._mean, np.sqrt(self._variance)))

  def maximum_likelihood(self):
    return self.expectation()

  def mean(self):
    return self._mean

  def sigma(self):
    return self._sigma

  def variance(self):
    return self._variance

