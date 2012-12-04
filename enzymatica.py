import numpy as np
import scipy.optimize as optimize
import pylab as plt
from scipy.integrate import odeint
from scipy.stats import beta
from functools import partial
from itertools import dropwhile, takewhile

def mm_rate(k,z):
  """The Michealis-Menten reaction rate a concentration z and rate parameter k"""
  s,e,c,j = z
  k_bind,k_unbind,k_cat = k
  return np.array([-k_bind*s*e+k_unbind*c,
                   -k_bind*s*e+k_unbind*c+k_cat*c,
                   k_bind*s*e-k_unbind*c-k_cat*c,
                   k_cat*c])

def reaction_time_series(rate_fun, z0, T):
    """Returns a reaction time series given the rate function, initial condition, and range of times"""
    if np.size(T)==1:
        t = np.linspace(0,T,1000).T
    else:
        t = T
    return odeint(lambda zed,tau: rate_fun(zed),z0,t)

def basic_susceptibility(phi,z0,z):
    """Returns a simple estimate of the susceptibiltiy based on the chemical state"""
    fraction = z[:,3]/z0[0]
    return beta.cdf(fraction,phi[0],phi[1])

def turbidity_from(reaction_state,susceptibility,calibration):
    """Returns the turbidity given the reaction sate and susceptibilty functions"""    
    return reaction_state[0,0]*(calibration[0]+(calibration[1]-calibration[0])*susceptibility(reaction_state))

def infer_ml_parameters_given(sigma,calib,initial_conditions,time,turbidity,phi0, k_bounds = None,k0 = None):
    """Return maximum likelihood solution the basic instance of the inference problem"""
    if k0 is None:
        k0 = np.mean(k_bounds,1) 
    def model(k1,k2,k3,a,b):
        k = [k1,k2,k3]
        Z = [reaction_time_series(partial(mm_rate,k), z0, time) for z0 in initial_conditions]
        susceptibility = lambda z0,z: basic_susceptibility((a,b),z0,z)
        return np.array([turbidity_from(z,partial(susceptibility,z[0,:]),calib) for z in Z])
    x0 = np.concatenate((k0,phi0))
    f = lambda x: np.ndarray.flatten(model(*x)-turbidity)/sigma
    params_est, params_est_cov, infodict, message,flag = optimize.leastsq(f,x0,full_output=1)
    return (params_est, params_est_cov)

def parse_turbidity_data(filename):
    """Parses standard turbidity datafile to obtain metadata,calibration,initial conditions and time series"""
    with open(filename) as f:
      meta,sigma,calib,initial_conditions, time_series = _segment_data(f.readlines())
    sigma = float(sigma[0])
    calib = np.fromstring(calib[0],sep=' ')
    initial_conditions = zip(np.fromstring(initial_conditions[0],sep=' '),
                             np.fromstring(initial_conditions[1],sep=' '),
                             np.zeros(np.shape(np.fromstring(initial_conditions[0],sep=' '))),
                             np.zeros(np.shape(np.fromstring(initial_conditions[0],sep=' '))))
    time_series = np.array([np.fromstring(row,sep=' ') for row in time_series]).T
    t = time_series[0]
    turbidity = time_series[1:]
    return (meta,calib,sigma,initial_conditions,t,turbidity)
    
def _segment_data(data):
    data = iter(d[:-1]+' ' for d in data)
    is_brak_line = lambda d : d[0] == '>'
    return [list(_drop_then_take_until(is_brak_line,data)) for i in range(5)]

def _drop_then_take_until(pred,seq):
    return takewhile(lambda x: not pred(x),dropwhile(pred,seq))

