# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np

class NeuralNetworkValidator:
  def __init__(self, neuralNetwork):
    self.network = neuralNetwork
    self.dataSet = []

  def loadDataFile(self, sourcePath):
    self.dataSet = np.load(sourcePath)
    return self.network

  def validate(self):
    correctPredictionsCounter = [0]*len(self.network.outputMap)
    dataSetLengths = [0]*len(self.network.outputMap)
    for dataPoint in self.dataSet:
      dataSetLengths[self.network.outputMap.index(dataPoint.output)] += 1
      if self.network.manipulator.classifier.evaluate(dataPoint.input) == dataPoint.output:
        correctPredictionsCounter[self.network.outputMap.index(dataPoint.output)] += 1
    return correctPredictionsCounter, dataSetLengths
  
  
  

