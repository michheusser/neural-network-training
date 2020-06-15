import numpy as np

class NeuralNetworkValidator:
  def __init__(self, neuralNetwork):
    self.network = neuralNetwork
    self.dataSet = []

  def loadDataFile(self, sourcePath):
    self.dataSet = np.load(sourcePath)
    return self.network

  def validate(self):
    correctPredictionsCounter = 0
    for dataPoint in self.dataSet:
      if self.network.manipulator.classifier.evaluate(dataPoint.input) == dataPoint.output:
        correctPredictionsCounter += 1
    return correctPredictionsCounter, len(self.dataSet)
  
  
  

