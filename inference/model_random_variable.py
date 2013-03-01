import numpy as np
import scipy.optimize as optimize
import pylab as plt
from collections import namedtuple
from inference import RandomVariable

class ModelRandomVariable(RandomVariable):
  def __init__(self,input_space,output_space,model,is_deterministic = True):
    self.space = [input_space,output_space]#logically, the cartesian product
    self.model = model
    self.is_deterministic = is_deterministic


