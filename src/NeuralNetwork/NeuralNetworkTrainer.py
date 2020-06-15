import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt

class NeuralNetworkTrainer:
  def __init__(self, neuralNetwork,validator):
    self.network = neuralNetwork
    self.eta = 0
    self.dataSet = []
    self.initializeWeightsBias()
    self.validator = validator
    
  def initializeWeightsBias(self):
    self.gradientToBias = [None]*len(self.network.layers)
    self.gradientToWeights = [None]*len(self.network.layers)

  def batchBackPropagation(self,inputOutputBatch):
    self.initializeWeightsBias()

    activations = [None]*len(self.network.activations)
    for i in range(0,len(activations)): #Initialize
      activations[i] = np.empty((len(inputOutputBatch),self.network.activations[i].shape[0],self.network.activations[i].shape[1]))
    
    output = np.empty((len(inputOutputBatch),self.network.activations[-1].shape[0],self.network.activations[-1].shape[1]))
    for i in range(0,len(inputOutputBatch)):
      inputVector, outputVector = self.vectorizeInputOuput(inputOutputBatch[i])
      self.network.loadInput(inputVector)
      self.network.activate()
      output[i] = outputVector
      for l in range(1,len(activations)):
        activations[l][i] = self.network.activations[l]
    
    self.gradientToBias[-1] =(activations[-1]-output)*(activations[-1]-np.square(activations[-1]))
    for i in range(2,len(self.network.layers)):
      self.gradientToBias[-i] = np.tensordot(self.gradientToBias[-i+1],self.network.weights[-i+1],axes= ((1),(0))).transpose(0,2,1)*(activations[-i]-np.square(activations[-i]))
    for i in range(1,len(self.network.layers)):
      self.gradientToWeights[i] = np.einsum('ijk,ilm->ijl',self.gradientToBias[i],activations[i-1])
    return self.network

  def update(self):
    for i in range(1,len(self.network.layers)):
      self.network.weights[i] -= self.eta*np.sum(self.gradientToWeights[i],axis =0)
      self.network.bias[i] -= self.eta*np.sum(self.gradientToBias[i], axis = 0)
    return self.network

  def shuffleData(self):
    rd.shuffle(self.dataSet)
    return self.network

  def createMiniBatch(self, miniBatchSize, index):
    return self.dataSet[index*miniBatchSize:(index+1)*miniBatchSize] 

  def mapOutputToVector(self,output):
      outputVector = np.zeros((len(self.network.outputMap),1))
      outputVector[self.network.outputMap.index(output)] = 1
      return outputVector

  def vectorizeInputOuput(self,inputOutputData):
    return inputOutputData.input.flatten().reshape((-1,1)), self.mapOutputToVector(inputOutputData.output)

  def train(self,epochs,miniBatchSize,eta):
    validationAccuracy = [None]*epochs
    self.eta = eta
    print("Training data: ", str(len(self.dataSet)), " datapoints")
    print("Training starting...")
    startTime = time.time()
    for i in range(0,epochs):
      self.shuffleData()
      for j in range(0,len(self.dataSet)//miniBatchSize):
        self.batchBackPropagation(self.createMiniBatch(miniBatchSize,j))
        self.update()
        print("Epoch:", str(i), ", batch:", str(j), "of" , str(len(self.dataSet)//miniBatchSize))
      print("Validating...")
      correctOutputs, dataSetLength = self.validator.validate()
      print("Finished Validation.")
      validationAccuracy[i] = round(correctOutputs/dataSetLength,4)
    endTime = time.time()
    print("Training finished:", round(endTime - startTime), "seconds")
    self.displayResults(validationAccuracy)
    return self.network

  def loadDataFile(self, sourcePath):
    self.dataSet = np.load(sourcePath)
    return self.network

  def displayResults(self,validationAccuracy):
    plt.plot(validationAccuracy)
    plt.show()
    return validationAccuracy
  

#def backPropagation(self,input,output):
    # self.initializeWeightsBias()
    # self.network.loadInput(input)
    # self.network.activate()

    # self.gradientToBias[-1] = np.multiply((self.network.activations[-1]-output),(self.network.activations[-1]-np.square(self.network.activations[-1])))
    # for i in range(2,len(self.network.layers)):
    #     self.gradientToBias[-i] = np.dot(self.network.weights[-i+1].transpose(),self.gradientToBias[-i+1])*(self.network.activations[-i]-self.network.activations[-i]**2)
    # for i in range(1,len(self.network.layers)):
    #   self.gradientToWeights[i] = np.dot(self.gradientToBias[i],self.network.activations[i-1].transpose())
    # return self.network