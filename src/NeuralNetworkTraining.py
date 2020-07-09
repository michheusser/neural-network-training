from NeuralNetwork.NeuralNetworkManipulator import NeuralNetworkManipulator

trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_training.npy"
validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_validation.npy"
testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_testing.npy"
neuralNetworkFile = "/Users/michelsmacbookpro/Desktop/Projects/neural-network-training/src/Saved Network"

# trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_extended_training.npy"
# validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_extended_validation.npy"
# testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_extended_testing.npy"
# neuralNetworkFile = "/Users/michelsmacbookpro/Desktop/Projects/neural-network-training/src/Saved Network"

#trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_training.npy"
#validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_validation.npy"
#testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_testing.npy"
#neuralNetworkFile = "/Users/michelsmacbookpro/Desktop/Projects/neural-network-training/src/Saved Network (Small)"

outputMap = '0123456789+-*%[]'
layers = (784,64,32,16)
#layers = (784,45,16)
eta = 0.02
#gamma = 0.000001
#gamma = 0.0000001
gamma = 0

learningType = [('sigmoid','MSE'),('softmax','CE')]
a = 0
#neuralNetwork = NeuralNetworkManipulator().create(layers, outputMap,learningType[a][0])
neuralNetwork = NeuralNetworkManipulator().importFiles(neuralNetworkFile,learningType[a][0])
neuralNetwork.manipulator.train(trainingDataPath=trainingSetPath, epochs=10, miniBatchSize=20,eta=eta, validationDataPath=validationSetPath,func=learningType[a][1], calculateCost=False, gamma = gamma)
neuralNetwork.manipulator.exportFiles(neuralNetworkFile)
