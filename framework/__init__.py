# -*- encoding: utf-8 -*-

import re
from numpy import matrix
from framework.io import read_file
from framework.algorithm import gradient_desc

def normalizer(X):
    """
    generates a matrix of normalizing parameters which will be applied to
    getX() and gradient_desc()
    :param X: the raw data that should be normalized
    :return: normalizing vector
    """
    return matrix([[(sum(X.T[i].flat) if i > 0 else 1.0) for i in xrange(X.shape[1])]])

def getX(raw,norm=None):
    """
    takes the raw data and generates a matrix from it, with each row's
    first element replaced by 1. That's nessesary for gradient_desc.
    :param raw: a numpy.matrix with float() elements
    :param norm: a vector to normalize getX
    :return: a numpy.matrix
    """
    p_norm = norm if norm != None else matrix([[1,1,1]])
    return matrix([[(raw[i,j] if j > 0 else 1) for j in xrange(raw.shape[1])] for i in xrange(raw.shape[0])]) / p_norm

def getY(raw):
    """
    makes a vector with just the first column of a matrix
    :param raw: a numpy.matrix with float() elements
    :return: a numpy.matrix
    """
    return matrix([[raw[i,0]] for i in xrange(raw.shape[0])])

def prepData(data):
    """
    takes a list of data which is unformatted for learning usage
    and formats it correctly.
    [2,3,4] -> numpy.matrix([[1,2,3,4]])
    :param data: a list of float() elements
    :return: a numpy.matrix
    """
    return matrix([[1] + data])

def learn_data(data,algo=None):
    """
    creates a trained function which predicts results
    according to the given input
    :param data: the raw trainingset as m-x-(n+1) numpy.matrix
    :param algo: the algorithm chosen for learning,
                 default:algorithm.gradient_desc
    :return: a function that maps a 1-x-n numpy.matrix to a float()
             which is a prediction. Keep in mind that the n here and
             in the param data must be the same number!
    """
    algorithm = algo if algo != None else gradient_desc
    norm = normalizer(data)
    return (lambda theta: lambda inp: (prepData(inp) * theta)[0,0])(algorithm(getX(data,norm),getY(data)) / norm.T)

def learn_from_io(data_descr,algo=None,io=None):
    """
    applies learn_data() to data gotten from an external source.
    :param data_descr: a unique data description used by the param io,
                       i.e. a filename
    :param algo: the function chosen for learning,
                 default: algorithm.gradient_desc
    :param io: a function that reads data from external sources,
               uses the param data_descr to find the source
               default: read_file(*filename*)
    :return: a function, see learn_data for details
    """
    algorithm = algo if algo != None else gradient_desc
    dreader = io if io != None else read_file
    return learn_data(matrix(dreader(data_descr)),algorithm)
