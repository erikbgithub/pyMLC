# -*- encoding: utf-8 -*-

import re

def read_file(name):
  """
  read cvs data from a file.
  Whitespaces is ignored, as well as lines that contain a '#' sign somewhere.
  All elements must be legal python float() numbers.
  :param name: the filename.
  :result: is a list containing a list for each line. The inner list
  contains float values for each entry
  """
  with open(name,"r") as file:
    return [[float(x) for x in re.sub(r'\s','',line).split(",")] for line in file if "#" not in line]
