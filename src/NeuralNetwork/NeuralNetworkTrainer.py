import numpy as np


class NeuralNetworkTrainer:
  def __init__(self, neuralNetwork):
    self.network = neuralNetwork
    self.gradientToBias = [None]*len(self.network.layers)
    self.gradientToWeights = [None]*len(self.network.layers)
    self.eta = 1
    self.dataSet

  def backPropagation(self,input,output):
    self.network.loadInput(input)
    self.network.activate()

    self.gradientToBias[-1] = np.multiply((self.network.activations[-1]-output),(self.network.activations[-1]-np.square(self.network.activations[-1])))
    for i in range(2,len(self.network.layers)):
        self.gradientToBias[-i] = np.dot(self.network.weights[-i+1].transpose(),self.gradientToBias[-i+1])*(self.network.activations[-i]-self.network.activations[-i]**2)
    self.gradientToBias = self.gradientToBias 
    for i in range(1,len(self.network.layers)):
      self.gradientToWeights[i] = np.dot(self.gradientToBias[i],self.network.activations[i-1].transpose())
    return self.network
  
  def update(self):
    for i in range(1,len(self.network.layers)):
      self.network.weights[i] -= self.eta*self.gradientToWeights[i]
      self.network.bias[i] -= self.eta*self.gradientToBias[i]
    return self.network

  def train(self,epochs,miniBatchSize):
    pass

  def loadDataFile(self, sourcePath):
    self.dataSet = np.load(sourcePath)
    return self.network

