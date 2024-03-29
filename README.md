Description
===========

pyMLC is a python framework based on what I learned from [ML class](http://www.ml-class.com). It contains basic functionality for machine learning purposes.

Usage
============

To use this framework you need Python2.7 and Numpy installed.
Simply download pyMLC and test if everything works with `./main.py -h`

Now I will guide you through an example. People familiar with the Google
classification service shouldn't be too unfamiliar with the process.

Example:

    $ ./main.py ~/data.csv ~/input.csv
    16.0
    28.0
    20.0
    20.1142857143

for data.csv:

    16,89,4
    20,124,9
    20,103,5

and input.csv:

    89,4
    124,9
    103,5
    100,6

__What happens here?__

data.csv contains csv data with 3 datasets which begin with the result and continue with 2 features each.
input.csv contains 4 observations, which also have 2 features like the training data.

Based on the training data pyMLC learns to predict results according to these 2 features. The learning result is a function itself, which then will be applied to each row of the input.csv seperately, yielding a predicted result.
Because the first 3 oberservations (datasets in input.csv) are the same as in our trainingset, you can check that the predicted values really are the results it should predict.
