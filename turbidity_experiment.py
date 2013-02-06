import numpy as np
import scipy.optimize as optimize
import pylab as plt
from scipy.integrate import odeint
from scipy.stats import beta
from functools import partial
from itertools import dropwhile, takewhile

class TurbidityExperiment(object):
  """
  A turbidity experiment. This is an informational class that will define the attributes of an
  experiment that is to be conducted. Provided some initial conditions, calibration constants and
  other meta data, we can define the experiment that is run.

  --------------------------------------------------------------------------------------------------
  Superclass and Subtypes
  --------------------------------------------------------------------------------------------------
  This is going to be used as a super class for the sub types of experimentation. There will be 2
  specific subsets of experimentation, which will be simulation and analysis. The analysis will be
  working on real data, and will be provided turbidity information. The simulation will be given
  known chemistry and curve parameters (k, phi), and will simulate the turbidity provided a model.
  """

  def __init__(self, **config):
    self.meta = ('meta' in config and config['meta']) or None
    self.calibration = ('calibration' in config and config['calibration']) or None
    self.sigma = ('sigma' in config and config['sigma']) or None
    self.initial_conditions = ('initial_conditions' in config and config['initial_conditions']) or None
    self.t = ('t' in config and config['t']) or None
    self.turbidity = ('turbidity' in config and config['turbidity']) or None
    self.phi0 = ('phi0' in config and config['phi0']) or None
    self.time_series = None

  def basic_susceptibility(self, phi, z0, z):
    """
    Returns a simple estimate of the susceptibiltiy based on the chemical state.
    This is where the implementation of the beta distribution occurs.
    """
    fraction = z[:,3] / z0[0]
    return beta.cdf(fraction,phi[0],phi[1])

  def infer_ml_parameters_given(self, k_bounds = None, k0 = None):
    """Return maximum likelihood solution the basic instance of the inference problem"""
    if k0 is None:
      k0 = np.mean(k_bounds,1)
    
    x0 = np.concatenate((k0, self.phi0))
    f = lambda x: np.ndarray.flatten(self._model(*x) - self.turbidity) / self.sigma
    params_est, params_est_cov, infodict, message,flag = optimize.leastsq(f,x0,full_output=1)
    self.params_est = params_est
    self.params_est_cov = params_est_cov
    return (params_est, params_est_cov)

  def mm_rate(self, k, z):
    """The Michealis-Menten reaction rate a concentration z and rate parameter k"""
    s,e,c,j = z
    k_bind,k_unbind,k_cat = k
    return np.array([-k_bind * s * e + k_unbind * c,
                     -k_bind * s * e + k_unbind * c + k_cat * c,
                     k_bind * s * e - k_unbind * c - k_cat * c,
                     k_cat * c])

  def parse_turbidity_data(self, filename):
    """
    The meta information for the experiment can be parsed from the file provided as an argument.
    If the file has turbidity data associated with it, then it will assume that it has either been
    generated or provided.
    """
    with open(filename) as f:
      meta, sigma, calib, initial_conditions, time_series = self._segment_data(f.readlines())
    sigma = float(sigma[0])
    calib = np.fromstring(calib[0],sep=' ')
    initial_conditions = zip(np.fromstring(initial_conditions[0], sep=' '), 
                             np.fromstring(initial_conditions[1], sep=' '), 
                             np.zeros(np.shape(np.fromstring(initial_conditions[0], sep=' '))), 
                             np.zeros(np.shape(np.fromstring(initial_conditions[0], sep=' '))))

    time_series = np.array([np.fromstring(row,sep=' ') for row in time_series]).T
    t = time_series[0]
    turbidity = time_series[1:]
    
    # Set the enzymatica values based on the parsed data.
    self.meta = meta
    self.calib = calib
    self.sigma = sigma
    self.initial_conditions = initial_conditions
    self.t = t
    self.turbidity = turbidity

    return (meta, calib, sigma, initial_conditions, t, turbidity)

  def reaction_time_series(self, rate_fun, z0, T):
    """Returns a reaction time series given the rate function, initial condition, and range of times"""
    if np.size(T)==1:
      t = np.linspace(0,T,1000).T
    else:
      t = T
    return odeint(lambda zed,tau: rate_fun(zed),z0,t)

  def turbidity_from(self, reaction_state, susceptibility, calibration):
    """Returns the turbidity given the reaction sate and susceptibilty functions"""
    return reaction_state[0,0] * (calibration[0] + (calibration[1] - calibration[0]) * susceptibility(reaction_state))

  def _drop_then_take_until(self, pred,seq):
    return takewhile(lambda x: not pred(x),dropwhile(pred,seq))

  def _model(self, k1, k2, k3, a, b):
    k = [k1, k2, k3]
    Z = [self.reaction_time_series(partial(self.mm_rate, k), z0, self.time) for z0 in self.initial_conditions]
    susceptibility = lambda z0, z: self.basic_susceptibility((a, b), z0, z)
    return np.array([self.turbidity_from(z, partial(susceptibility, z[0, :]), self.calib) for z in Z])

  def _segment_data(self, data):
    data = iter(d[:-1]+' ' for d in data)
    is_brak_line = lambda d : d[0] == '>'
    return [list(self._drop_then_take_until(is_brak_line,data)) for i in range(5)]

