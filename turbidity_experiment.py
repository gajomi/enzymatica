import numpy as np
import scipy.optimize as optimize
import pylab as plt

from functools import partial
from itertools import dropwhile, takewhile
from collections import namedtuple

TurbiditySetup = namedtuple('TurbiditySetup', 
                            ['meta', 'calibration','sigma','initial_conditions','t'])

class TurbidityExperiment(object):
  """
  A turbidity experiment. This is an informational class that will define the attributes of an
  experiment that is to be conducted. A turbidity experiment is contructed from a TturbiditySetup
  and either a specification of the turbidity data or a simulation

  --------------------------------------------------------------------------------------------------
  Superclass and Subtypes
  --------------------------------------------------------------------------------------------------
  This is going to be used as a super class for the sub types of experimentation. There will be 2
  specific subsets of experimentation, which will be simulation and analysis. The analysis will be
  working on real data, and will be provided turbidity information. The simulation will be given
  known chemistry and curve parameters (k, phi), and will simulate the turbidity provided a model.
  """

  def __init__(self,turbiditysetup, **config):
    self.turbiditysetup = turbiditysetup
    self.time_series = None

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

  def _drop_then_take_until(self, pred,seq):
    return takewhile(lambda x: not pred(x),dropwhile(pred,seq))

  def _segment_data(self, data):
    data = iter(d[:-1]+' ' for d in data)
    is_brak_line = lambda d : d[0] == '>'
    return [list(self._drop_then_take_until(is_brak_line,data)) for i in range(5)]

