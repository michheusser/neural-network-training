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
    self.validationAccuracy = []
    self.costs = []
    
  def initializeWeightsBias(self):
    self.gradientToBias = [None] +[np.zeros(self.network.bias[i].shape) for i in range(1,len(self.network.layers))]
    self.gradientToWeights = [None] + [np.zeros(self.network.weights[i].shape) for i in range(1,len(self.network.layers))]
    return self

  def evaluateCostFunction(self,func):
      C = 0
      for dataPoint in self.dataSet: 
        input, output = self.vectorizeInputOuput(dataPoint)
        self.network.loadInput(input)
        self.network.activate()
        C += self.costFunction(self.network.getOutput(),output,func)
      return C/len(self.dataSet)

  def costFunction(self, prediction, output, func, prime=False):
    if func == 'MSE':
      if prime:
        return prediction-output
      else:
        return 0.5*np.sum(np.square(prediction-output))
    if func == 'CE':
      if prime:
        return (prediction-output)
      else:
        return -np.sum(np.multiply(output,np.log(prediction)))
        #-np.sum(output*np.nan_to_num(np.log(prediction)) + (1-output)*np.nan_to_num(np.log(1-prediction)))

  def batchBackPropagation(self, inputOutputBatch,func):
    self.initializeWeightsBias()
    for i in range(0,len(inputOutputBatch)):
      gradientToWeights, gradientToBias = self.backPropagation(inputOutputBatch[i],func)
      for l in range(1,len(self.network.layers)):
        self.gradientToWeights[l] += gradientToWeights[l]
        self.gradientToBias[l] += gradientToBias[l]
    return self.network

  def backPropagation(self,inputOutputDatapoint,func):
    gradientToBias = [None]*len(self.network.layers)
    gradientToWeights = [None]*len(self.network.layers)
    input, output = self.vectorizeInputOuput(inputOutputDatapoint)
    self.network.loadInput(input)
    self.network.activate()
    gradientToBias[-1] = np.multiply(self.costFunction(self.network.getOutput(),output,func,True),(self.network.getOutput()-np.square(self.network.getOutput())))
    for i in range(2,len(self.network.layers)):
      gradientToBias[-i] = np.multiply(np.dot(self.network.weights[-i+1].transpose(),gradientToBias[-i+1]),(self.network.activations[-i]-np.square(self.network.activations[-i])))
    for i in range(1,len(self.network.layers)):
      gradientToWeights[i] = np.dot(gradientToBias[i],self.network.activations[i-1].transpose())
    return gradientToWeights, gradientToBias

  def update(self, miniBatchSize):
    for i in range(1,len(self.network.layers)):
      self.network.weights[i] -= (self.eta/miniBatchSize)*self.gradientToWeights[i]
      self.network.bias[i] -= (self.eta/miniBatchSize)*self.gradientToBias[i]
      #self.network.weights[i] -= (self.eta/miniBatchSize)*np.sum(self.gradientToWeights[i],axis =0)
      #self.network.bias[i] -= (self.eta/miniBatchSize)*np.sum(self.gradientToBias[i], axis = 0)
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

  def train(self,epochs,miniBatchSize,eta,func,calculateCost):
    self.eta = eta
    print("Training data: ", str(len(self.dataSet)), " datapoints")
    print("Training starting...")
    startTime = time.time()
    for i in range(0,epochs):
      self.shuffleData()
      for j in range(0,len(self.dataSet)//miniBatchSize):
        self.batchBackPropagation(self.createMiniBatch(miniBatchSize,j),func)
        self.update(miniBatchSize)
        print("Epoch:", str(len(self.validationAccuracy)+1), ", batch:", str(j+1), "of" , str(len(self.dataSet)//miniBatchSize))
      print("Validating...")
      correctOutputs, dataSetLength = self.validator.validate()
      print("Finished Validation with " + str(round(correctOutputs*100/dataSetLength,2)) + " accuracy.")
      self.validationAccuracy.append(round(correctOutputs/dataSetLength,4))
      if calculateCost:
        print("Calculating current cost...")
        cost = self.evaluateCostFunction(func)
        print("Current cost: " + str(cost))
        self.costs.append(cost)
    endTime = time.time()
    print("Training finished:", round(endTime - startTime), "seconds")
    self.displayResults()
    return self.network

  def loadDataFile(self, sourcePath):
    self.dataSet = np.load(sourcePath)
    return self.network

  def displayResults(self):
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy on validation set')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost on training set')
    ax1.plot(self.validationAccuracy)
    ax2.plot(self.costs)
    plt.show()
    return self.network
  
# def batchBackPropagation2(self,inputOutputBatch):
#     self.initializeWeightsBias()

#     activations = [None]*len(self.network.activations)
#     for i in range(0,len(activations)): #Initialize
#       activations[i] = np.zeros((len(inputOutputBatch),self.network.activations[i].shape[0],self.network.activations[i].shape[1]))
    
#     output = np.zeros((len(inputOutputBatch),self.network.activations[-1].shape[0],self.network.activations[-1].shape[1]))
#     for i in range(0,len(inputOutputBatch)):
#       inputVector, outputVector = self.vectorizeInputOuput(inputOutputBatch[i])
#       self.network.loadInput(inputVector)
#       output[i] = outputVector
#       self.network.activate()
#       for l in range(1,len(activations)):
#         activations[l][i] = self.network.activations[l]
#     self.gradientToBias[-1] =(activations[-1]-output)*(activations[-1]-np.square(activations[-1]))
#     for i in range(2,len(self.network.layers)):
#       self.gradientToBias[-i] = np.tensordot(self.gradientToBias[-i+1],self.network.weights[-i+1],axes= ((1),(0))).transpose(0,2,1)*(activations[-i]-np.square(activations[-i]))
#     for i in range(1,len(self.network.layers)):
#       self.gradientToWeights[i] = np.einsum('ijk,ilm->ijl',self.gradientToBias[i],activations[i-1])
#     return self.network