import numpy as np
import scipy.optimize as optimize
from scipy.stats import norm
import pylab as plt
from collections import namedtuple
from inference import RandomVariable

class NormalDistribution(RandomVariable):
  def __init__(self, mean, variance):
    super(NormalDistribution, self).__init__(

  def E(self,f = None):
    if f is None:
      return self.mean
    else:
      raise NotImplementedError
  
  def L(self,x):
    return norm.pdf(x,self.mean,np.sqrt(self.variance))

  def P(self,event):
    """ The probability of the event """
    if isinstance(event,RealInterval):
      return norm.cdf(event.upper,self.mean,np.sqrt(self.variance))-norm.cdf(event.lower,self.mean,np.sqrt(self.variance))
    else:
      return 0.

  def maximum_likelihood(self):
    return self.E()

