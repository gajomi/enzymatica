import numpy as np
import scipy.optimize as optimize
import pylab as plt
from collections import namedtuple

class RV(object):
    def __init__(self,space,Pfun):
        self.space = space
        self.Pfun = Pfun

class Normal(RV):
    def __init__(self,mean,variance):
        self.mean = mean
        self.variance = variance
        self.space = "Reals"
    def E(self,f = None):
        if f is None:
            return self.mean
        else:
            raise NotImplementedError
    def L(self,x):
        return norm.pdf(x,self.mean,np.sqrt(self.variance))
    def P(self,event):
        if isinstance(event,RealInterval):
            return norm.cdf(event.upper,self.mean,np.sqrt(self.variance))-norm.cdf(event.lower,self.mean,np.sqrt(self.variance))
        else:
            return 0.
    def maximum_likelihood(self):
        return self.E()

class Uniform(RV):
    def __init__(self,interval):
        self.space = interval
    def E(self,f = None):
        if f is None:
            return [.5*(x[0]+x[1]) for x in space]
        else:
            raise NotImplementedError
    def L(self,x):
        return 1.

class ModelRV(RV):
    def __init__(self,input_space,output_space,model,is_deterministic = True):
        self.space = [input_space,output_space]#logically, the cartesian product
        self.model = model
        self.is_deterministic = is_deterministic

class Posterior(RV):
    def __init__(self,prior,model,data):
        self.prior = prior
        self.model = model
        self.data = data
    def E(self,f = None):
        raise NotImplementedError
    def L(self,x):
        if self.model.is_deterministic:
            d = self.model(x)
            return self.prior.L(x)*self.data.L(d)
        else:
            raise NotImplementedError
    def maximum_likelihood(self,x0=None):
        if x0 is None:
            if isinstance(self.prior,UniformRV):
                x0 = [np.mean(y) for y in self.prior.space]
            else:
                x0 = self.prior.maximum_likelihood()
        f = lambda x: -np.log(self.L(x))
        params_est, params_est_cov, infodict, message,flag = optimize.leastsq(f,x0,full_output=1)
        return (params_est,params_est_cov)


    


