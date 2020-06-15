import numpy as np
from .NeuralNetworkTrainer import NeuralNetworkTrainer
from .NeuralNetworkValidator import NeuralNetworkValidator
from .NeuralNetworkClassifier import NeuralNetworkClassifier
from .NeuralNetwork import NeuralNetwork

class NeuralNetworkManipulator:
  def __init__(self, neuralNetwork = None):
    self.network = neuralNetwork
    self.classifier = None if self.network == None else NeuralNetworkClassifier(self.network)
    self.validator = None if self.network == None else NeuralNetworkValidator(self.network)
    self.trainer = None if self.network == None else NeuralNetworkTrainer(self.network, self.validator)
    
  def create(self, layers, outputMap):
    self.network = NeuralNetwork(layers = layers, value='randn', outputMap = outputMap, manipulator=self) 
    self.classifier = NeuralNetworkClassifier(self.network)
    self.validator = NeuralNetworkValidator(self.network)
    self.trainer = NeuralNetworkTrainer(self.network, self.validator)
    return self.network
  
  def test(self, input):
    return self.classifier.evaluate(input)
  
  def train(self, trainingDataPath,epochs,miniBatchSize,eta, validationDataPath):
    self.validator.loadDataFile(validationDataPath)
    self.trainer.loadDataFile(trainingDataPath)
    self.trainer.train(epochs,miniBatchSize,eta)
    return self.network
  
  def validate(self, validationDataPath):
    self.validator.loadDataFile(validationDataPath)
    return self.validator.validate()
  
  
  

