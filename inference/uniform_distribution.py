import numpy as np
import scipy.optimize as optimize
import pylab as plt
from collections import namedtuple
from inference import RandomVariable

class UniformDistribution(RandomVariable):
  def __init__(self, interval):
    self.space = interval

  def E(self,f = None):
    if f is None:
      return [.5*(x[0]+x[1]) for x in space]
    else:
      raise NotImplementedError

  def L(self,x):
    return 1.


