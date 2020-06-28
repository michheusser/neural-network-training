from NeuralNetwork.NeuralNetworkManipulator import NeuralNetworkManipulator

trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_training.npy"
validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_validation.npy"
testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_testing.npy"
neuralNetworkFile = "/Users/michelsmacbookpro/Desktop/Projects/neural-network-training/src/Saved Network"

outputMap = '0123456789+-*%[]'
layers = (784,45,16)
#neuralNetwork = NeuralNetworkManipulator().create(layers, outputMap)
neuralNetwork = NeuralNetworkManipulator().importFiles(neuralNetworkFile)
neuralNetwork.manipulator.train(trainingDataPath=trainingSetPath,epochs=5,miniBatchSize=20,eta=3, validationDataPath = validationSetPath,func = 'MSE')
neuralNetwork.manipulator.exportFiles(neuralNetworkFile)