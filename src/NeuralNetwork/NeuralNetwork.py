import numpy as np
#from .NeuralNetworkManipulator import NeuralNetworkManipulator

class NeuralNetwork:
  def __init__(self,layers, value=0, outputMap = '', manipulator = None, activation = 'sigmoid'):
    self.layers = layers
    self.outputMap = outputMap
    self.initialize(value)
    self.manipulator = manipulator
    self.activation = activation
  
  @property
  def layers(self):
    return self._layers

  @layers.setter
  def layers(self,layers):
    if type(layers)== tuple and len(layers) > 1:
      self._layers = layers
    else:
      raise Exception("layers must be a tuple with at least 2 entries")

  def addManipulator(self):
    #self.manipulator = NeuralNetworkManipulator(self)
    return self

  def initialize(self,value=0):
    if(value == 'randn'):
      self.weights = [None] + [np.random.randn(self.layers[i],self.layers[i-1]) for i in range(1,len(self.layers))]
      self.bias = [None] + [np.random.randn(self.layers[i],1) for i in range(1,len(self.layers))]
      print("")
    else:
      self.weights = [None] + [np.full((self.layers[i],self.layers[i-1]),value) for i in range(1,len(self.layers))]
      self.bias = [None] + [np.full((self.layers[i],1),value) for i in range(1,len(self.layers))]
    
    self.activations = [np.zeros((self.layers[i],1)) for i in range(0,len(self.layers))]
    self.weightedInputs = [None] + [np.zeros((self.layers[i],1)) for i in range(1,len(self.layers))]
    return self

  def loadOutputMap(self, outputMap):
    self.outputMap = outputMap
    return self
    
  def activationFunction(self,x,prime=False):
    if self.activation == 'sigmoid':
      sigma = 1/(1+np.exp(-x))
    elif self.activation == 'softmax':
      sigma = np.exp(x)/np.sum(np.exp(x))
    return sigma if not prime else sigma-np.square(sigma)

  def loadInput(self,input):
    if type(input) != np.ndarray or input.shape[1] != 1 or input.shape[0] != self.layers[0]:
      raise Exception("Input must be a numpy column vector of length " + str(self.layers[0]))
    for i in range(0,len(self.activations[0])):
      self.activations[0][i][0] = input[i][0]
    return self

  def getOutput(self):
    return self.activations[-1]

  def activate(self):
    for i in range(1,len(self.activations)):
        self.weightedInputs[i] = np.dot(self.weights[i],self.activations[i-1])+self.bias[i]
        self.activations[i] = self.activationFunction(self.weightedInputs[i])
    return self
