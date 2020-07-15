# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np
import random as rd
import time
import matplotlib.pyplot as plt

class NeuralNetworkTrainer:
  '''Provides the methods to train the linked neural network and contains the parameters of training
  as well as the training dataset and a validation object to perform the validations on the trained
  network'''
  def __init__(self, neuralNetwork,validator):
    self.network = neuralNetwork
    self.eta = 0
    self.gamma = 0
    self.dataSet = []
    self._initializeWeightsBias()
    self.validator = validator
    self.validationAccuracy = []
    self.costs = []
    self.singleValidationAccuracies = [[]]*len(self.network.outputMap)
    
  def _initializeWeightsBias(self):
    '''Initializes the gradients of the weights and bias of the stochastic gradient descent'''
    self.gradientToBias = [None] +[np.zeros(self.network.bias[i].shape) for i in range(1,len(self.network.layers))]
    self.gradientToWeights = [None] + [np.zeros(self.network.weights[i].shape) for i in range(1,len(self.network.layers))]
    return self

  def _evaluateCostFunction(self,func):
    '''Calculates the total cost function over the whole training dataset'''
    C = 0
    for dataPoint in self.dataSet: 
      input, output = self._vectorizeInputOuput(dataPoint)
      self.network.loadInput(input)
      self.network.activate()
      C += self.costFunction(self.network.getOutput(),output,func,False,self.gamma)
    return C/len(self.dataSet)

  def costFunction(self, prediction, output, func, prime=False, gamma=0):
    '''Defines the mathematical shape of the cost function: either Mean-squared-error or cross-entropy'''
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

  def _batchBackPropagation(self, inputOutputBatch,func):
    '''Calculates the gradients of weights and bias over a dataset batch (minibatch)'''
    self._initializeWeightsBias()
    for i in range(0,len(inputOutputBatch)):
      gradientToWeights, gradientToBias = self._backPropagation(inputOutputBatch[i],func)
      for l in range(1,len(self.network.layers)):
        self.gradientToWeights[l] += gradientToWeights[l]
        self.gradientToBias[l] += gradientToBias[l]
    return self.network

  def _backPropagation(self,inputOutputDatapoint,func):
    '''Returns the gradient of weights and bias over one single datapoint'''
    gradientToBias = [None]*len(self.network.layers)
    gradientToWeights = [None]*len(self.network.layers)
    input, output = self._vectorizeInputOuput(inputOutputDatapoint)
    self.network.loadInput(input)
    self.network.activate()
    gradientToBias[-1] = np.multiply(self.costFunction(self.network.getOutput(),output,func,True),self.network.activationFunction(self.network.weightedInputs[-1],True))
    for i in range(2,len(self.network.layers)):
      gradientToBias[-i] = np.multiply(np.dot(self.network.weights[-i+1].transpose(),gradientToBias[-i+1]),self.network.activationFunction(self.network.weightedInputs[-i],True))
    for i in range(1,len(self.network.layers)):
      gradientToWeights[i] = np.dot(gradientToBias[i],self.network.activations[i-1].transpose())
    return gradientToWeights, gradientToBias

  def _update(self, miniBatchSize):
    '''Updates the networks weights and bias with the current gradients of weights and bias'''
    for i in range(1,len(self.network.layers)):
      self.network.weights[i] = self.network.weights[i]*(1-self.eta*self.gamma/miniBatchSize) - (self.eta/miniBatchSize)*self.gradientToWeights[i]
      self.network.bias[i] -= (self.eta/miniBatchSize)*self.gradientToBias[i]
    return self.network

  def _shuffleData(self):
    '''Schuffles the training dataset for the creation of minibatches in the stochastic gradient descent'''
    rd.shuffle(self.dataSet)
    return self.network 

  def _createMiniBatch(self, miniBatchSize, index):
    '''Returns a specific minibatch of a specific length out of the training dataset'''
    return self.dataSet[index*miniBatchSize:(index+1)*miniBatchSize] 

  def _mapOutputToVector(self,output):
    '''Returns a zero vector with the corresponding output set to 1'''
    outputVector = np.zeros((len(self.network.outputMap),1))
    outputVector[self.network.outputMap.index(output)] = 1
    return outputVector

  def _vectorizeInputOuput(self,inputOutputData):
    '''Takes the input and output of a datapoint and vectorizes to match the dimensions of 
    the input and output layer of the neural network'''
    return inputOutputData.input.flatten().reshape((-1,1)), self._mapOutputToVector(inputOutputData.output)

  def train(self,epochs,miniBatchSize,eta,func,calculateCost, gamma):
    '''Using backpropagation it trains the neural network with the training dataset for the given epochs
    minibatch size, cost function, learning rate (eta) and regularization parameter (gamma)'''
    self.eta = eta
    self.gamma = gamma
    print("Training data: ", str(len(self.dataSet)), " datapoints")
    print("Training starting...")
    startTime = time.time()
    for i in range(0,epochs):
      self._shuffleData()
      for j in range(0,len(self.dataSet)//miniBatchSize):
        self._batchBackPropagation(self._createMiniBatch(miniBatchSize,j),func)
        self._update(miniBatchSize)
        print("Epoch:", str(len(self.validationAccuracy)+1), ", batch:", str(j+1), "of" , str(len(self.dataSet)//miniBatchSize))
      print("Validating...")
      correctOutputs, dataSetLengths = self.validator.validate()
      print("Finished Validation with " + str(round(sum(correctOutputs)*100/sum(dataSetLengths),2)) + " %" + " accuracy.")
      self._updateAccuracies(correctOutputs,dataSetLengths,display=True)
      if calculateCost:
        print("Calculating current cost...")
        cost = self._evaluateCostFunction(func)
        print("Current cost: " + str(cost))
        self.costs.append(cost)
    endTime = time.time()
    print("Training finished:", round(endTime - startTime), "seconds")
    self.displayResults()
    return self.network

  def _updateAccuracies(self,correctOutputs,dataSetLengths, display=False):
    '''Adds the training accuracy of every single possible output of the current epoch to the list
    of accuracies, to be plotted at the end of training'''
    self.validationAccuracy.append(round(sum(correctOutputs)/sum(dataSetLengths),4))
    for output in self.network.outputMap:
      index = self.network.outputMap.index(output)
      self.singleValidationAccuracies[index] =self.singleValidationAccuracies[index] +[round(correctOutputs[index]/dataSetLengths[index],4)]
      if display:
        print('Accuracy of ' + output + ": " + str(round(correctOutputs[index]*100/dataSetLengths[index],2))+ "%")
    return self.network
    
  def loadDataFile(self, sourcePath):
    '''Loads the training dataset from the specified path'''
    self.dataSet = np.load(sourcePath)
    return self.network

  def displayResults(self):
    '''Displays graphically the results of the training, which includes, overall accuracy per epoch,
    single output accuracy per epoch, and cost function per epoch'''
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
    '''Calculates the sum of all weights in the neural network, as a part of the cost function with 
    regularization'''
    sumW = 0
    for i in range(1,len(self.network.layers)):
      sumW += np.sum(self.network.weights[i],axis=(0,1))
    return sumW
  