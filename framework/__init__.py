# -*- encoding: utf-8 -*-

import re
from numpy import matrix
from framework.io import read_file
from framework.algorithm import gradient_desc

def getX(raw):
    """
    takes the raw data and generates a matrix from it, with each row's
    first element replaced by 1. That's nessesary for gradient_desc.
    :param raw: a numpy.matrix with float() elements
    :return: a numpy.matrix
    """
    return matrix([[(raw[i,j] if j > 0 else 1) for j in range(raw.shape[1])] for i in range(raw.shape[0])])

def getY(raw):
    """
    makes a vector with just the first column of a matrix
    :param raw: a numpy.matrix with float() elements
    :return: a numpy.matrix
    """
    return matrix([[raw[i,0]] for i in range(raw.shape[0])])


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
    return (lambda theta: lambda inp: (prepData(inp) * theta)[0,0])(algorithm(getX(data),getY(data)))

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
