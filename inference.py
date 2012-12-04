import numpy as np
import scipy.optimize as optimize
import pylab as plt
from collections import namedtuple

RV = namedtuple('RV',['interval','likelihood','pdf'])

def normalize(X):
    """Returns the random variable with properply normalized pdf"""
    pass

def maximum_likelihood(X,k0 = None):
    if k0 is None:
        interval = X.interval
        if isinstance(interval,list):
            k0 = [np.mean(i) for i in interval]
        else:
            k0 = np.mean(interval)
    f = lambda k: -np.log(X.likelihood(k))
    params_est, params_est_cov, infodict, message,flag = optimize.leastsq(f,k0,full_output=1)
    return (params_est,params_est_cov)

def is_within(interval,x):
    """Returns True if the argument is strictly within the specified interval"""
    if isinstance(interval,list):
        return all(is_within(i,x) for i in interval)
    else:
        return interval[0] < x < interval[1]

def interval_intersection(i1,i2):
    """Returns the intersection of two intervals"""
    if isinstance(i1,list):
        return [interval_intersection(x,y) for x,y in zip(i1,i2)]
    else:
        return (np.max(i1[0],i2[0]),np.min(i1[1],i2[1]))

def uniform(interval):
    """Returns a uniformly disitributed random variable over the interval"""
    if isinstance(interval,list):
        u = 1.0/product(i[1]-i[0] for i in interval)
        uniform_pdf = lambda k: u if is_within(interval,k) else 0
    else:
        uniform_pdf = lambda k: 1.0/(interval[1]-interval[0]) if is_within(interval,k) else 0
    return RV(interval,uniform_pdf,uniform_pdf)

def normal(mean,covariance):
    """Returns a normally distributed random variable min specified mean and covariance"""
    N = len(data)
    if np.size(covariance) == 1:
        interval = [(-np.Inf,np.Inf) for i in range(N)]
        normal_pdf = lambda x : np.exp(-0.5*np.sum((x-data)**2)/covariance)/np.sqrt(2*np.pi*covariance)
        return RV(interval,normal_pdf,normal_pdf)
    else:
        raise NotImplementedError("Will implement if this error is ever thrown")

def posterior_params(prior_data,prior_params,model):
    """Forms RV fro posterior parameters from priors and model"""
    interval = prior_params.interval    
    posterior_likelihood = lambda k: prior_params.likelihood(k)*prior_data.likelihood(model(k))
    return RV(interval,posterior_likelihood,None)







    


