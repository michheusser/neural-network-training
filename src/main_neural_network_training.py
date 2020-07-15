# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

from nntools.manipulator import NeuralNetworkManipulator

# DATASET PATHS
trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_training.npy"
validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_validation.npy"
testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_testing.npy"
neuralNetworkFile = "/Users/michelsmacbookpro/Desktop/Projects/neural-network-training/src/Saved Networks/Current"

# TRAINING PARAMETERS
outputMap = '0123456789+-*%[]'
layers = (784,64,32,16)
eta = 0.02
gamma = 0
learningType = [('sigmoid','MSE'),('softmax','CE')]
a = 0

# NEURAL NETWORK TRAINING
neuralNetwork = NeuralNetworkManipulator().create(layers, outputMap,learningType[a][0]) # Only run when training for the first time
#neuralNetwork = NeuralNetworkManipulator().importFiles(neuralNetworkFile,learningType[a][0]) # Run when training cycles have already happened
neuralNetwork.manipulator.train(trainingDataPath=trainingSetPath, epochs=10, miniBatchSize=20,eta=eta, validationDataPath=validationSetPath,func=learningType[a][1], calculateCost=True, gamma=gamma)
neuralNetwork.manipulator.exportFiles(neuralNetworkFile)
