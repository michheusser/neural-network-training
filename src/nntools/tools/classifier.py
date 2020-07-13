# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np

class NeuralNetworkClassifier:
  def __init__(self, neuralNetwork):
    self.network = neuralNetwork

  def mapVectorToOutput(self,outputVector):
      return self.network.outputMap[np.argmax(outputVector)]

  def vectorizeInput(self,inputData):
    return inputData.flatten().reshape((-1,1))

  def evaluate(self,inputData):
    self.network.loadInput(self.vectorizeInput(inputData))
    self.network.activate()
    return self.mapVectorToOutput(self.network.getOutput())

