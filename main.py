#!/usr/bin/env python2
# -*- coding: utf-8 -*-

VERSION = "1.0"

import argparse

from framework import read_file, learn_from_io

def main():
    parser = argparse.ArgumentParser(description="simple Python Machine Learning Algorithms v"+VERSION)
    parser.add_argument("trainingset",metavar="T", type=str, help="filename of the csv trainingset")
    parser.add_argument("input",metavar="I", type=str, help="filename of the csv input data")
    parser.add_argument("-v","--version",action="version",version="%%(prog)s v%s"%VERSION)
    args = parser.parse_args()
    in_data = read_file(args.input)
    h = learn_from_io(args.trainingset)
    for i in in_data:
        print h(i)

if __name__ == "__main__":
    main()
