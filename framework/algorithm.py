# -*- encoding: utf-8 -*-

def gradient_desc(X,y):
  """
  Gradient descent is a first-order optimization algorithm.
  http://en.wikipedia.org/wiki/Gradient_descent
  This is written in the Matrix form:
  :param X: is a m-x-(n+1) matrix, with m trainingsets (each a row) and n params. 
  There is an additional param leading the others, which is always set to an.
  :param y: is a m-x-1 vector, containing the solutions to the training sets
  .T is the transitiv
  .I is the inverse
  all multiplications are matrix multiplications
  :return: a n-x-1 vector containing the parameters that will, applied to a
  """
  return (X.T * X).I * X.T * y
