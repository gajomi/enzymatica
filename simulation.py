from enzymatica import TurbidityExperiment
from functools import partial
from itertools import dropwhile, takewhile
import numpy as np
from scipy.integrate import odeint
from scipy.stats import beta

def mm_rate(k, z):
  """The Michealis-Menten reaction rate a concentration z and rate parameter k"""
  s,e,c,j = z
  k_bind,k_unbind,k_cat = k
  return np.array([-k_bind * s * e + k_unbind * c,
                   -k_bind * s * e + k_unbind * c + k_cat * c,
                   k_bind * s * e - k_unbind * c - k_cat * c,
                   k_cat * c])

def beta_susceptibility(phi,z):
  """Returns a susceptibility given shape parameters and chemical state. This is expecting
  a single initial condition."""
  fraction = z[:,3] / z[0,0]
  return beta.cdf(fraction, phi[0],phi[1])

class Simulation(object):
  """
  This is a class for simulating turbidity time series. The input is a turbidity setup,
  a reaction model
  """
  def __init__(self,turbiditysetup,k, phi,reaction_model = mm_rate,susceptiblity_model = beta_susceptibility):
    self.turbiditysetup = turbiditysetup
    self.reaction_model = reaction_model
    self.susceptiblity_model = susceptiblity_model
    self.k = k
    self.phi = phi

  def reaction_time_series(self):
    """Returns a reaction time series for this simulation"""
    Z0 = self.turbiditysetup.initial_conditions
    rate_fun = partial(self.reaction_model,self.k)
    return np.array([odeint(lambda zed,tau: rate_fun(zed),z0,self.turbiditysetup.t) for z0 in Z0])

  def susceptibility(self,z):
    """Returns the suscetibility at the specified chemical state"""
    return self.susceptiblity_model(self.phi,z)

  def turbidity_time_series(self):
    """Returns a turbidity time series for the simulation"""
    calib = self.turbiditysetup.calibration
    reaction_state = self.reaction_time_series()
    c0 = calib[0] # Turbidity per unit substrate
    c1 = calib[1] # Turbidity per unit product
    delta = c1 - c0 # Differential specific turbidity
    nconditions = reaction_state.shape[0]
    return np.array([reaction_state[i,0,0] * (c0 + delta * self.susceptibility(reaction_state[i,:,:])) for i in range(nconditions)])
#This may be desireable to wrap in inference. Leave it out for now
#  def _model(self, k1, k2, k3, a, b):
#    k = [k1, k2, k3]
#    Z = [self.reaction_time_series(partial(self.mm_rate, k), z0, self.time) for z0 in self.initial_conditions]
#    susceptibility = lambda z0, z: self.basic_susceptibility((a, b), z0, z)
#    return np.array([self.turbidity_from(z, partial(susceptibility, z[0, :]), self.calib) for z in Z])
