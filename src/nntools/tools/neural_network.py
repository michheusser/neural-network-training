# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np

class NeuralNetwork:
  '''The main neural network object that contains the information about the layer sizes, activations,
  activation function, outputMap (the symbol corresponding to every output neuron), its neuron bias,
  and the neuron connection weights. It also has a manipulator object appended that contains all the
  methods and objects to handle it'''
  def __init__(self,layers, value=0, outputMap = '', manipulator = None, activation = 'sigmoid'):
    self.layers = layers
    self.outputMap = outputMap
    self.initialize(value)
    self.manipulator = manipulator
    self.activation = activation
  
  @property
  def layers(self):
    '''Getter for the layers property'''
    return self._layers

  @layers.setter
  def layers(self,layers):
    '''Setter for the layers property'''
    if type(layers)== tuple and len(layers) > 1:
      self._layers = layers
    else:
      raise Exception("layers must be a tuple with at least 2 entries")

  def initialize(self,value=0):
    '''Initializes weights, bias and activations'''
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
    '''Loads the output map of the neural network'''
    self.outputMap = outputMap
    return self
    
  def activationFunction(self,x,prime=False):
    '''Defines the activation function of the neural network. Set prime=True to get the derivative'''
    if self.activation == 'sigmoid':
      sigma = 1/(1+np.exp(-x))
    elif self.activation == 'softmax':
      xExp = np.exp(x-np.max(x))
      sigma = xExp/np.sum(xExp)
    return sigma if not prime else sigma-np.square(sigma)

  def loadInput(self,input):
    '''Sets the passed input as the activation of the input layer'''
    if type(input) != np.ndarray or input.shape[1] != 1 or input.shape[0] != self.layers[0]:
      raise Exception("Input must be a numpy column vector of length " + str(self.layers[0]))
    for i in range(0,len(self.activations[0])):
      self.activations[0][i][0] = input[i][0]
    return self

  def getOutput(self):
    '''Gets the output vector of the neural network (i.e. the output layer)'''
    return self.activations[-1]

  def activate(self):
    '''Evaluates the neural network progressively starting from the input layer feeding
    forward towards the output layer'''
    for i in range(1,len(self.activations)):
        self.weightedInputs[i] = np.dot(self.weights[i],self.activations[i-1])+self.bias[i]
        self.activations[i] = self.activationFunction(self.weightedInputs[i])
    return self
