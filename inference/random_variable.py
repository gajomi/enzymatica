class RandomVariable(object):
  """
  A representation of a random variable, that defines the probability distribution over some space.
  The space is the bounds of which the random variable exists, and the Pfun is the function provided
  to the random variable that defines the probability density at a poin thtat is provided.
  """

  def __init__(self, pdf, space=[], mean=None, variance=None, stddev=None, covar=None):
    """ 
    Initial conditions include a probability density function, and a space. The space cannot be 
    null. 
    """
    self._pdf = pdf
    self._space = space
    self._mean = mean
    self._stddev = stddev
    # If the covariance is a scalar, then the appropriate diagonal matrix can be made.
    # Assuming we know the dimensions from the space and the mean
    self._covar = covar # Will be a diagonal matrix.

    if len(self.space) > 0):
      self.__verify_sample_space() # At this time, assume the sample space passed in is correct.

  def add_sample_space_point(self, point):
    """
    Add a point that defines a boundary value.
    """
    self.space += point
    self.__verify_sample_space()

  def __verify_sample_space(self):
    """ 
    The sample space needs to maintain a certain number of bounds in the n dimensional space.
    Assuming that the space is defined by closing in certain points, which will make it a convex
    polygon, then a definition of which points form an intersection with other points is not
    necessary.
    """
    space = self.space

    # Check that space exists.
    if space == None || (not isinstance(space, type([])) || len(space) == 0:
      raise 'The space for this random variable cannot be empty'

    # Check that the dimensionality of each tuple in the space is the same.
    n = len(space[0])
    for boundary in space:
      if len(boundary) != n:
        raise 'All points must have the same dimensionality.'

    return True

