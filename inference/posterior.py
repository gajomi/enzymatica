import numpy as np
import scipy.optimize as optimize
import pylab as plt
from collections import namedtuple
from inference import RandomVariable, UniformDistribution

class Posterior(RandomVariable):
  def __init__(self, prior, model, data):
    self.prior = prior
    self.model = model
    self.data = data

  def expectation(self, f = None):
    raise NotImplementedError
  
  def likelihood(self, x):
    """
    Assuming a deterministic parameter for the model x. In this case the model is turbidity. 
    For any particular parameter set (kinetic and shape parameters). 
    Multiply the likelihood of the prior by the likelihood of the data at that point.
    """
    if self.model.is_deterministic:
      d = self.model.model(x)
      return self.prior.likelihood(x) * self.data.likelihood(d)
    else:
      raise NotImplementedError

  def log_likelihood(self, x):
    if self.model.is_deterministic:
      d = self.model.model(x)
      return self.prior.log_likelihood(x) + self.data.log_likelihood(d)
    else:
      raise NotImplementedError

  def maximum_likelihood(self, x0 = None):
    """
    Provided a space of likelihoods, optimize and find the maximum.
    """
    if x0 is None:
      if isinstance(self.prior, UniformDistribution):
        x0 = [np.mean(y) for y in self.prior.space]
      else:
        x0 = self.prior.maximum_likelihood()

    x0 = np.array(x0).T
    f = lambda x: -self.log_likelihood(x) # Once the function f has been initialized.
    # This is expecting x to be a vector.
    result = optimize.minimize(f,x0, method='BFGS')
    return result # Estimate and covariance structure.

