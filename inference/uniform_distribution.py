import numpy as np
import scipy.optimize as optimize
import pylab as plt
from collections import namedtuple
from inference import RandomVariable

class UniformDistribution(RandomVariable):
  def __init__(self, interval):
    self.space = interval

  def expectation(self, f = None):
    if f is None:
      return [.5*(x[0]+x[1]) for x in space]
    else:
      raise NotImplementedError

  def likelihood(self,x):
    return 1.0 # The likelihood of any point within the uniform distribution are equally likely

  def log_likelihood(self, x):
    # If the value falls outside of the bounds, return negative infinity.

    return 0.0

