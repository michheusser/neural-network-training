import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt

class NeuralNetworkTrainer:
  def __init__(self, neuralNetwork,validator):
    self.network = neuralNetwork
    self.eta = 0
    self.gamma = 0
    self.dataSet = []
    self.initializeWeightsBias()
    self.validator = validator
    self.validationAccuracy = []
    self.costs = []
    self.singleValidationAccuracies = [[]]*len(self.network.outputMap)
    
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
        C += self.costFunction(self.network.getOutput(),output,func,False,self.gamma)
      return C/len(self.dataSet)

  def costFunction(self, prediction, output, func, prime=False, gamma=0):
    if func == 'MSE':
      if prime:
        return prediction-output
      else:
        return 0.5*np.sum(np.square(prediction-output)) + 0.5*gamma*self.calculateWeightSum()
    if func == 'CE':
      if prime:
        return (prediction-output)
      else:
        return -1*np.sum(np.multiply(output,np.log(prediction))) + 0.5*gamma*self.calculateWeightSum()

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
    gradientToBias[-1] = np.multiply(self.costFunction(self.network.getOutput(),output,func,True),self.network.activationFunction(self.network.weightedInputs[-1],True))
    for i in range(2,len(self.network.layers)):
      gradientToBias[-i] = np.multiply(np.dot(self.network.weights[-i+1].transpose(),gradientToBias[-i+1]),self.network.activationFunction(self.network.weightedInputs[-i],True))
    for i in range(1,len(self.network.layers)):
      gradientToWeights[i] = np.dot(gradientToBias[i],self.network.activations[i-1].transpose())
    return gradientToWeights, gradientToBias

  def update(self, miniBatchSize):
    for i in range(1,len(self.network.layers)):
      self.network.weights[i] = self.network.weights[i]*(1-self.eta*self.gamma/miniBatchSize) - (self.eta/miniBatchSize)*self.gradientToWeights[i]
      self.network.bias[i] -= (self.eta/miniBatchSize)*self.gradientToBias[i]
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

  def train(self,epochs,miniBatchSize,eta,func,calculateCost, gamma):
    self.eta = eta
    self.gamma = gamma
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
      correctOutputs, dataSetLengths = self.validator.validate()
      print("Finished Validation with " + str(round(sum(correctOutputs)*100/sum(dataSetLengths),2)) + " %" + " accuracy.")
      self.updateAccuracies(correctOutputs,dataSetLengths,display=True)
      if calculateCost:
        print("Calculating current cost...")
        cost = self.evaluateCostFunction(func)
        print("Current cost: " + str(cost))
        self.costs.append(cost)
    endTime = time.time()
    print("Training finished:", round(endTime - startTime), "seconds")
    self.displayResults()
    
    return self.network

  def updateAccuracies(self,correctOutputs,dataSetLengths, display=False):
    self.validationAccuracy.append(round(sum(correctOutputs)/sum(dataSetLengths),4))
    for output in self.network.outputMap:
      index = self.network.outputMap.index(output)
      self.singleValidationAccuracies[index] =self.singleValidationAccuracies[index] +[round(correctOutputs[index]/dataSetLengths[index],4)]
      if display:
        print('Accuracy of ' + output + ": " + str(round(correctOutputs[index]*100/dataSetLengths[index],2))+ "%")
    return self.network
    
  def loadDataFile(self, sourcePath):
    self.dataSet = np.load(sourcePath)
    return self.network

  def displayResults(self):
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Cost')
    ax1.plot(self.validationAccuracy)
    for output in self.network.outputMap:
      index = self.network.outputMap.index(output)
      ax2.plot(self.singleValidationAccuracies[index],label=output)
    ax2.legend(fontsize='small', ncol=len(self.network.outputMap),loc = 'lower left')
    ax3.plot(self.costs)
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax2.autoscale(enable=True, axis='x', tight=True)
    ax3.autoscale(enable=True, axis='x', tight=True)
    plt.show()
    return self.network
  
  def calculateWeightSum(self):
    sumW = 0
    for i in range(1,len(self.network.layers)):
      sumW += np.sum(self.network.weights[i],axis=(0,1))
    return sumW
  