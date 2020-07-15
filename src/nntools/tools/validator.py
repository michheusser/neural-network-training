# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np

class NeuralNetworkValidator:
  '''Contains the network, the validation dataset, and the method to validate the neural network
  over said specific validation dataset'''
  def __init__(self, neuralNetwork):
    self.network = neuralNetwork
    self.dataSet = []

  def loadDataFile(self, sourcePath):
    '''Loads the validation dataset to the class'''
    self.dataSet = np.load(sourcePath)
    return self.network

  def validate(self):
    '''Validates the network over the validation dataset and returns two lists, containing
    the correctly predicted outputs of the output map and the amount of outputs
    that were evaluated of each entry in the output map'''
    correctPredictionsCounter = [0]*len(self.network.outputMap)
    dataSetLengths = [0]*len(self.network.outputMap)
    for dataPoint in self.dataSet:
      dataSetLengths[self.network.outputMap.index(dataPoint.output)] += 1
      if self.network.manipulator.classifier.evaluate(dataPoint.input) == dataPoint.output:
        correctPredictionsCounter[self.network.outputMap.index(dataPoint.output)] += 1
    return correctPredictionsCounter, dataSetLengths
  
  
  

