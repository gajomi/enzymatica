import numpy as np
import scipy.optimize as optimize
import pylab as plt
from collections import namedtuple
from inference import RandomVariable, UniformDistribution
from random import random

def debug_optimize(f, iter_steps, num_trials):
  options = {'maxiter' : iter_steps}
  trials = [0 for i in range(num_trials)]
  iterations = [[] for i in range(num_trials)]
  start_points = []
  best_trial_run = None
  best_val = None
  
  for i in range(num_trials):
    run = None
    x0 = [random(), random()]
    start_points.append(x0)
    trial_iterations = iterations[i]

    while run is None or not run.success: 
      run = optimize.minimize(f, x0, method='BFGS', options = options)
      x0 = run.x
      trial_iterations.append(x0)

    trials[i] = run.x
    val = f(x0)
    
    if best_val is None or best_val < val:
      best_val = val
      best_trial_run = run
  
  return {
      'num_trials' : num_trials,
      'iter_steps' : iter_steps,
      'start_points' : start_points,
      'iterations' : iterations,
      'trials' : trials,
      'best_trial' : best_trial_run}

class Posterior(RandomVariable):
  NUM_RANDOM_RUNS = 5
  
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

  def maximum_likelihood(self, x0 = None, num_trials = None):
    """
    Provided a space of likelihoods, optimize and find the maximum.

    Arguments:
    num_restarts - Integer specifying the number of optimization restarts.
    """
    if x0 is None:
      if isinstance(self.prior, UniformDistribution):
        x0 = [np.mean(y) for y in self.prior.space]
      else:
        x0 = self.prior.maximum_likelihood()

    x0 = np.array(x0).T
    f = lambda x: -self.log_likelihood(x) # Once the function f has been initialized.
    # This is expecting x to be a vector.

    if num_trials is not None:
      result = self.optimize(f, num_trials)
    else:
      result = optimize.minimize(f,x0, method='BFGS')
    
    return result # Estimate and covariance structure.

  def optimize(self, f, num_trials):
    iter_steps = 5
    options = {'maxiter' : iter_steps}
    best_trial_run = None
    best_val = None
    
    for i in range(num_trials):
      run = None
      x0 = [random(), random()]
  
      while run is None or not run.success: 
        run = optimize.minimize(f, x0, method='BFGS', options = options)
        x0 = run.x
  
      val = f(x0)
      
      if best_val is None or best_val < val:
        best_val = val
        best_trial_run = run
   
    return best_trial_run

