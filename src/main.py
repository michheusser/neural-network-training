from NeuralNetwork.NeuralNetworkManipulator import NeuralNetworkManipulator

trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_training.npy"
validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_validation.npy"
testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_testing.npy"
neuralNetworkFile = "/Users/michelsmacbookpro/Desktop/Projects/neural-network-training/src/Saved Network"

outputMap = '0123456789+-*%[]'
layers = (784,45,16)
neuralNetwork = NeuralNetworkManipulator().create(layers, outputMap)
#neuralNetwork = NeuralNetworkManipulator().importFiles(neuralNetworkFile)
neuralNetwork.manipulator.train(trainingDataPath=trainingSetPath,epochs=2,miniBatchSize=2000,eta=3, validationDataPath = validationSetPath)
#neuralNetwork.manipulator.exportFiles(neuralNetworkFile)
