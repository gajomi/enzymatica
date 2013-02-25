import numpy as np
import scipy.optimize as optimize
import pylab as plt
from collections import namedtuple
from inference import RandomVariable

class Posterior(RandomVariable):
  def __init__(self,prior,model,data):
    self.prior = prior
    self.model = model
    self.data = data
  def E(self,f = None):
    raise NotImplementedError
  def L(self,x):
    """
    Assuming a deterministic parameter for the model x. In this case the model is turbidity. 
    For any particular parameter set (kinetic and shape parameters). 
    Multiply the likelihood of the prior by the likelihood of the data at that point.
    """
    if self.model.is_deterministic:
      d = self.model(x)
      return self.prior.L(x)*self.data.L(d)
    else:
      raise NotImplementedError

  def maximum_likelihood(self,x0=None):
    """
    Provided a space of likelihoods, optimize and find the maximum.
    """
    if x0 is None:
      if isinstance(self.prior,UniformRV):
        x0 = [np.mean(y) for y in self.prior.space]
      else:
        x0 = self.prior.maximum_likelihood()
    f = lambda x: -np.log(self.L(x)) # Once the function f has been initialized.
    # This is expecting x to be a vector.
    params_est, params_est_cov, infodict, message,flag = optimize.leastsq(f,x0,full_output=1)
    return (params_est,params_est_cov) # Estimate and covariance structure.

