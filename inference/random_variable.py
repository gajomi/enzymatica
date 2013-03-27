class RandomVariable(object):
  """
  A representation of a random variable, that defines the probability distribution over some space.
  The space is the bounds of which the random variable exists, and the Pfun is the function provided
  to the random variable that defines the probability density at a poin thtat is provided.
  """

  def __init__(self, pdf, space = [], mean = None, variance = None, stddev = None, covar = None):
    """ 
    Initial conditions include a probability density function, and a space. The space cannot be 
    null.

    Keyword Args:
    space -- A set of points within the space that will restrict the range.
    """
    self._pdf = pdf
    self._space = space
    self._mean = mean
    self._stddev = stddev
    # If the covariance is a scalar, then the appropriate diagonal matrix can be made.
    # Assuming we know the dimensions from the space and the mean
    self._covar = covar # Will be a diagonal matrix.
    
    if len(self._space) > 0:
      self.__verify_sample_space() # At this time, assume the sample space passed in is correct.

  def add_sample_space_point(self, point):
    """
    Add a point that defines a boundary value.
    """
    self.space += point
    self.__verify_sample_space()

  def cdf(self, x = None, interval = None):
    """ 
    The CDF of either a single point, or an interval. If neither the point x or interval are
    supplied, then this returns a function that can take in a single value. This function will treat
    a single point as the maximum value that it can take, and will sample all values less than that
    point in all dimensions.

    Keyword Args:
    x -- A point in the sample space.
    interval -- A range to in the sample space to find the cumulative probability of.
    """
    raise NotImplementedError

  def expectation(self, interval = None):
    """
    The expected value of the distribution. If an interval is supplied, then it returns
    the expected value of that range.
    
    Keyword Args:
    interval -- The interval to get the expected value of.
    """
    raise NotImplementedError 

  def likelihood(self, x):
    """
    The likelihood of a point x in this random variable space.
    
    Keyword Args:
    x -- The point to test the likelihood of.
    """
    raise NotImplementedError

  def log_likelihood(self, x):
    """
    The log likelihood of a point x in this random variable space.
    
    Keyword Args:
    x -- The point to test the likelihood of.
    """
    raise NotImplementedError

  def pdf(self, x = None, interval = None):
    """
    The PDF of the random variable at a single point or an interval. If neither the point x or
    the interval are supplied, then this returns a function that takes in a point.

    Keyword Args:
    x -- The point to get the probability of.
    interval -- A 2-tuple range to calculate the probability of.
    """
    raise NotImplementedError

  # private ----------------------------------------------------------------------------------------

  def __verify_sample_space(self):
    """ 
    The sample space needs to maintain a certain number of bounds in the n dimensional space.
    Assuming that the space is defined by closing in certain points, which will make it a convex
    polygon, then a definition of which points form an intersection with other points is not
    necessary.
    """
    space = self._space

    # Check that space exists.
    if space == None or (isinstance(space, type([])) and len(space) == 0):
      raise 'The space for this random variable cannot be empty'

    # Check that the dimensionality of each tuple in the space is the same.
    n = len(space[0])
    for boundary in space:
      if len(boundary) != n:
        raise 'All points must have the same dimensionality.'

    return True

