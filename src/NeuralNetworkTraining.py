from NeuralNetwork.NeuralNetworkManipulator import NeuralNetworkManipulator

# trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_training.npy"
# validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_validation.npy"
# testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_testing.npy"
# neuralNetworkFile = "/Users/michelsmacbookpro/Desktop/Projects/neural-network-training/src/Saved Network"

trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_training.npy"
validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_validation.npy"
testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_testing.npy"
neuralNetworkFile = "/Users/michelsmacbookpro/Desktop/Projects/neural-network-training/src/Saved Network"
outputMap = '0123456789+-*%[]'
layers = (784,45,16)
lmbda = 5.0
eta = 0.1

learningType = [('sigmoid','MSE'),('softmax','CE')]
neuralNetwork = NeuralNetworkManipulator().create(layers, outputMap,learningType[1][0])
#neuralNetwork = NeuralNetworkManipulator().importFiles(neuralNetworkFile,'sigmoid')
neuralNetwork.manipulator.train(trainingDataPath=trainingSetPath, epochs=20, miniBatchSize=20,eta=0.1, validationDataPath=validationSetPath,func=learningType[1][1], calculateCost=False)
#neuralNetwork.manipulator.exportFiles(neuralNetworkFile)