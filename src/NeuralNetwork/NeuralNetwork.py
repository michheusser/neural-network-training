import numpy as np
from .NeuralNetworkTrainer import NeuralNetworkTrainer 


class NeuralNetwork:
  def __init__(self, layers, value=0):
    self.layers = layers
    self.initialize(value)
    self.trainer = NeuralNetworkTrainer(self)
  
  @property
  def layers(self):
    return self._layers

  @layers.setter
  def layers(self,layers):
    if type(layers)== tuple and len(layers) > 1:
      self._layers = layers
    else:
      raise Exception("layers must be a tuple with at least 2 entries")

  def initialize(self,value=0):
    if(value == 'randn'):
      self.weights = tuple([None] + [np.random.randn(self.layers[i],self.layers[i-1]) for i in range(1,len(self.layers))])
      self.bias = tuple([None] + [np.random.randn(self.layers[i],1) for i in range(1,len(self.layers))])
    else:
      self.weights = tuple([None] + [np.full((self.layers[i],self.layers[i-1]),value) for i in range(1,len(self.layers))])
      self.bias = tuple([None] + [np.full((self.layers[i],1),value) for i in range(1,len(self.layers))])
    
    self.activations = tuple([np.zeros((self.layers[i],1)) for i in range(0,len(self.layers))])
    return self

  def activationFunction(self,x,prime=False):
    sigma = 1/(1+np.exp(-x)) 
    return sigma if not prime else sigma-sigma**2

  def loadInput(self,input):
    if type(input) != np.ndarray or input.shape[1] != 1 or input.shape[0] != self.layers[0]:
      raise Exception("Input must be a numpy column vector of length " + str(self.layers[0]))
    for i in range(0,len(self.activations[0])):
      self.activations[0][i][0] = input[i][0]

  def activate(self):
    for i, activation in enumerate(self.activations):
      if i:
        activation = self.activationFunction(np.dot(self.weights[i],self.activations[i-1])+self.bias[i])
