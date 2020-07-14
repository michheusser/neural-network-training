# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np
import pickle
import json
from .tools.trainer import NeuralNetworkTrainer
from .tools.validator import NeuralNetworkValidator
from .tools.classifier import NeuralNetworkClassifier
from .tools.neural_network import NeuralNetwork

class NeuralNetworkManipulator:
  def __init__(self, neuralNetwork = None):
    self.network = neuralNetwork
    self.classifier = None if self.network == None else NeuralNetworkClassifier(self.network)
    self.validator = None if self.network == None else NeuralNetworkValidator(self.network)
    self.trainer = None if self.network == None else NeuralNetworkTrainer(self.network, self.validator)
    
  def create(self, layers, outputMap, activation):
    self.network = NeuralNetwork(layers = layers, value='randn', outputMap = outputMap, manipulator=self,activation=activation) 
    self.classifier = NeuralNetworkClassifier(self.network)
    self.validator = NeuralNetworkValidator(self.network)
    self.trainer = NeuralNetworkTrainer(self.network, self.validator)
    return self.network
  
  def test(self, input):
    return self.classifier.evaluate(input)
  
  def train(self, trainingDataPath,epochs,miniBatchSize,eta, validationDataPath,func,calculateCost,gamma):
    self.validator.loadDataFile(validationDataPath)
    self.trainer.loadDataFile(trainingDataPath)
    self.trainer.train(epochs,miniBatchSize,eta,func,calculateCost,gamma)
    return self.network
  
  def validate(self, validationDataPath):
    self.validator.loadDataFile(validationDataPath)
    return self.validator.validate()
  
  def importFiles(self,sourcePath,activation):
    layers, outputMap, weights, bias, accuracy, costs, singleAccuracies= self.loadFiles(sourcePath)
    self.create(layers,outputMap,activation)
    self.network.weights = weights
    self.network.bias = bias
    self.trainer.validationAccuracy = accuracy
    self.trainer.costs = costs
    self.trainer.singleValidationAccuracies = singleAccuracies
    return self.network

  def loadFiles(self,sourcePath):
    layers = pickle.load(open(sourcePath + "/layers.p","rb"))
    outputMap = pickle.load(open(sourcePath + "/outputMap.p","rb"))
    weights = pickle.load(open(sourcePath + "/weights.p","rb"))
    bias = pickle.load(open(sourcePath + "/bias.p","rb"))
    accuracy = pickle.load(open(sourcePath + "/accuracy.p","rb"))
    costs = pickle.load(open(sourcePath + "/costs.p","rb"))
    singleAccuracies = pickle.load(open(sourcePath + "/singleAccuracies.p","rb"))
    return layers, outputMap, weights, bias, accuracy, costs, singleAccuracies

  def exportFiles(self, destinationPath):
    print("Saving neural network...")
    pickle.dump(self.network.layers, open(destinationPath + "/layers.p", 'wb'))
    pickle.dump(self.network.outputMap, open(destinationPath + "/outputMap.p", 'wb'))
    pickle.dump(self.network.weights, open(destinationPath + "/weights.p", 'wb'))
    pickle.dump(self.network.bias, open(destinationPath + "/bias.p", 'wb'))
    print("Neural Network saved to: " + destinationPath)
    print("Saving training information...")
    pickle.dump(self.trainer.validationAccuracy, open(destinationPath + "/accuracy.p", 'wb'))
    pickle.dump(self.trainer.singleValidationAccuracies, open(destinationPath + "/singleAccuracies.p", 'wb'))
    pickle.dump(self.trainer.costs, open(destinationPath + "/costs.p", 'wb'))
    print("Training information saved to" + destinationPath)
    return self

  def numpyToList(self,numpyArray):
    pass

  def exportToJSON(self,destinationPath):
    print("Exporting neural network to JSON")



  
  

