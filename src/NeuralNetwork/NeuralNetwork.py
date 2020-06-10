import numpy as np

class NeuralNetwork:
  def __init__(self, layers):
    self.weights = [np.zeros((layers[i],layers[i-1])) for i in range(1,len(layers))]
    self.bias = [np.zeros((layers[i],1)) for i in range(1,len(layers))]


