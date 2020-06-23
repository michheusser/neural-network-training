from NeuralNetwork.NeuralNetworkManipulator import NeuralNetworkManipulator

trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_training.npy"
validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_validation.npy"
testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_testing.npy"

outputMap = '0123456789+-*:[]'
layers = (784,45,16)
neuralNetwork = NeuralNetworkManipulator().create(layers, outputMap)
neuralNetwork.manipulator.train(trainingDataPath=trainingSetPath,epochs=10,miniBatchSize=40,eta=2, validationDataPath = validationSetPath)