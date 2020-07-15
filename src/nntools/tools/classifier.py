# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np

class NeuralNetworkClassifier:
  '''Provides the tools to evaluate and obtain the output of the linked neural network with
  a certain input'''
  def __init__(self, neuralNetwork):
    self.network = neuralNetwork

  def _mapVectorToOutput(self,outputVector):
    '''Returns the output symbol out of the maximum value in the output vector'''
    return self.network.outputMap[np.argmax(outputVector)]

  def _vectorizeInput(self,inputData):
    '''Reshapes an input to the correct dimensions to work with the neural network'''
    return inputData.flatten().reshape((-1,1))

  def evaluate(self,inputData):
    '''Returns the result from the classification, i.e. the symbol with the highest
    likelyhood out of the neural network's output'''
    self.network.loadInput(self._vectorizeInput(inputData))
    self.network.activate()
    return self._mapVectorToOutput(self.network.getOutput())

