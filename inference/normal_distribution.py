import numpy as np
import scipy.optimize as optimize
from scipy.stats import norm
import pylab as plt
from collections import namedtuple
from inference import RandomVariable

class NormalDistribution(RandomVariable):
  def __init__(self, mean, variance):
    if type(mean) == type(0.0):
      super(NormalDistribution, self).__init__(self, None)
      self.N = 1
    else:
      # Do something fancy.
      self.N = self.num_vars = len(mean)
    
    self.mean = mean
    self.variance = variance
    self.sigma = np.sqrt(variance)

  def E(self,f = None):
    if f is None:
      return self.mean
    else:
      raise NotImplementedError
  
  def L(self, x):
    if self.N > 1:
      # assuming the variables are independent of each other
      values = np.array([np.log(norm.pdf(x_i, mu_i, self.sigma)) for x_i, mu_i in zip(x, self.mean)])
      return np.exp(np.sum(values))
    else:
      return norm.pdf(x, self.mean, np.sqrt(self.variance))

  def P(self,event):
    """ The probability of the event """
    if isinstance(event,RealInterval):
      return norm.cdf(event.upper,self.mean,np.sqrt(self.variance))-norm.cdf(event.lower,self.mean,np.sqrt(self.variance))
    else:
      return 0.

  def maximum_likelihood(self):
    return self.E()

