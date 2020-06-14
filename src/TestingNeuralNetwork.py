from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.NeuralNetworkManipulator import NeuralNetworkManipulator
import numpy as np
import math

# neuralNetwork = NeuralNetwork(layers = (7,6,5),value = 'randn')
# input1 = np.array([[1],[0.5],[0.25],[0.5],[1],[0.25],[0.5]])
# output1 = np.array([[0.5],[1],[0.5],[1],[0.5]])
# inputOutput1 = (input1,output1)
# input2 = np.array([[0.5],[0.3],[0.1],[0.4],[0.2],[0.4],[0.2]])
# output2 = np.array([[1],[0.5],[0.5],[1],[1]])
# inputOutput2 = (input2,output2)
# input3 = np.array([[0.3],[1],[0.25],[0.4],[0.5],[0.3],[0.1]])
# output3 = np.array([[0.5],[0.3],[0.1],[0.3],[0.2]])
# inputOutput3 = (input3,output3)

# inputOutputBatch = [inputOutput1,inputOutput2,inputOutput3]

#neuralNetwork.trainer.backPropagation(input1,output1).trainer.update()
#print(neuralNetwork.trainer.gradientToWeights)
#print(neuralNetwork.trainer.gradientToBias)


#neuralNetwork.trainer.batchBackPropagation(inputOutputBatch)
#neuralNetwork.trainer.update()

filePath = "/Users/michelsmacbookpro/Desktop/InputOutputDatapoints.npy"
outputMap = '0123456789+-*:[]'
layers = (784,45,16)
neuralNetwork = NeuralNetworkManipulator().create(layers, outputMap)

#print(neuralNetwork.trainer.loadOutputMap(outputMap).trainer.mapOutputToVector('0'))

