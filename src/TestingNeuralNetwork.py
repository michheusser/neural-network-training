from NeuralNetwork.NeuralNetwork import NeuralNetwork
import numpy as np
import math

neuralNetwork = NeuralNetwork(layers = (5,4,3),value = 'randn')
input = np.array([[1],[0.5],[0.25],[0.5],[1]])
output = np.array([[0.5],[1],[0.5]])

neuralNetwork.trainer.backPropagation(input,output).trainer.update()

print(neuralNetwork.trainer.gradientToWeights)
print(neuralNetwork.trainer.gradientToBias)