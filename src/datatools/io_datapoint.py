# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np

class InputOutputData:
    '''Contains the input and output data to work with the neural network package nntools
    the input data should be a 2D numpy array and the output data a single character'''
    def __init__(self,inputData=None,outputData=None):
        self.input = inputData
        self.output = outputData
    def __eq__(self, other):
        return self.output == other.output and np.array_equal(self.input,other.input)
    def __ne__(self, other):
        return self.output != other.output or (not np.array_equal(self.input,other.input))
    def __gt__(self,other):
        return self.output > other.output
    def __ge__(self,other):
        return self.output >= other.output
    def __hash__(self):
        return hash((str(self.input.flatten()),self.output))
    def copy(self):
        return InputOutputData(self.input.copy(),self.output)
    def toTuple(self):
        return (self.input,self.output)
