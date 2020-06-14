import numpy as np
from .NeuralNetworkTrainer import NeuralNetworkTrainer
from .NeuralNetworkValidator import NeuralNetworkValidator
from .NeuralNetworkClassifier import NeuralNetworkClassifier
from .NeuralNetwork import NeuralNetwork

class NeuralNetworkManipulator:
  def __init__(self, neuralNetwork = None):
    self.network = neuralNetwork
    self.trainer = None
    self.validator = None
    self.classifier = None

  def create(self, layers, outputMap):
    self.network = NeuralNetwork(manipulator=self, layers = layers, value='randn', outputMap = outputMap) 
    self.classifier = NeuralNetworkClassifier(self.network)
    return self.network
  
  def test(self, input):
    return self.classifier.evaluate(input)
  
  def train(self, trainingDataPath,epochs,miniBatchSize,eta):
    self.trainer = NeuralNetworkTrainer(self.network)
    self.trainer.loadDataFile(trainingDataPath)
    self.trainer.train(epochs,miniBatchSize,eta)
    return self.network
  
  def validate(self, validationDataPath):
    self.validator = NeuralNetworkValidator(self.network)
    self.validator.loadDataFile(validationDataPath)
    return self.validator.validate()
  
  
  

