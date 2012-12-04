import numpy as np
import scipy.optimize as optimize
import pylab as plt

def linear_model(args):
    """A simple linear model"""
    def f(x):
        return args[0]*x + args[1]
    return f

N = 10
x = np.linspace(0,1,N)

m_exact = 1.5
b_exact = 0.5
params_exact = (m_exact,b_exact)
f_exact = linear_model(params_exact)
y_exact = f_exact(x)

sigma = .005
y_data = y_exact + sigma*np.random.randn(N)

f = lambda args: (linear_model(args)(x) - y_data)/sigma
params_0 = (1.8,.4)

params_est, params_est_cov, infodict, message,flag = optimize.leastsq(f,params_0,full_output=1)

f_est = linear_model(params_est)
y_est = f_est(x)


m_range = np.linspace(0,3,200)
b_range = np.linspace(0,1,200)
mm,bb = np.meshgrid(m_range,b_range)

f_squared = lambda m,b: sum((linear_model((m,b))(x) - y_data)**2/sigma**2)
f_squared = np.vectorize(f_squared)

error = f_squared(mm,bb)
