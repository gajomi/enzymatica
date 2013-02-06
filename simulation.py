from enzymatica import TurbidityExperiment
from functools import partial
from itertools import dropwhile, takewhile
import numpy as np

class Simulation(TurbidityExperiment):
  """
  TODO add class level doc.
  """

  def __init__(self, k, phi, **config):
    TurbidityExperiment.__init__(self, **config)
    self.k = k
    self.phi = phi

