import numpy as np
from ImageProcessing.ImageData import ImageData
from ImageProcessing.ImageProcessor import ImageProcessor
from NeuralNetwork.NeuralNetworkManipulator import NeuralNetworkManipulator

# sourcePath = "/Users/michelsmacbookpro/Desktop/Symbol Images Selected"
# filePath = "/Users/michelsmacbookpro/Desktop/InputOutputDatapoints.npy"
# extendedFilePath = "/Users/michelsmacbookpro/Desktop/InputOutputDatapoints_Extended.npy"
# imageProcessor = ImageProcessor().createDataSet(sourcePath,False,True,200).exportDataSet(filePath)
# imageProcessor = ImageProcessor().importDataSet(filePath) 
# imageProcessor.getDataSummary()
# imageProcessor.generateArtificialData(symbol = '0', xScaleList = [1.2, 1], yScaleList = [1.2], rotationList = [-10,0,10], display = False)
# imageProcessor.generateArtificialData(symbol = '1', xScaleList = [1.1, 1], yScaleList = [], rotationList = [0], display = False)
# imageProcessor.generateArtificialData(symbol = '2', xScaleList = [1.1,1], yScaleList = [], rotationList = [-10,0,10], display = False)
# imageProcessor.generateArtificialData(symbol = '3', xScaleList = [1.1], yScaleList = [1.1,1], rotationList = [-10,0,10], display = False)
# imageProcessor.generateArtificialData(symbol = '4', xScaleList = [1.3, 1.1], yScaleList = [1.1,1], rotationList = [-10,0,10], display = False)
# imageProcessor.generateArtificialData(symbol = '5', xScaleList = [1.3 ,1.15, 1], yScaleList = [1.3, 1.1], rotationList = [-20,-10,0,10,-20], display = False)
# imageProcessor.generateArtificialData(symbol = '6', xScaleList = [1.3 ,1.15, 1], yScaleList = [1.5, 1.3, 1.1], rotationList = [-20,-10,0,10,-20], display = False)
# imageProcessor.generateArtificialData(symbol = '7', xScaleList = [1.3 ,1.15, 1], yScaleList = [1.5, 1.3, 1.1], rotationList = [-20,-10,0,10,20], display = False)
# imageProcessor.generateArtificialData(symbol = '8', xScaleList = [1.3 ,1.15, 1], yScaleList = [1.5, 1.3, 1.1], rotationList = [-20,-10,0,10,20], display = False)
# imageProcessor.generateArtificialData(symbol = '9', xScaleList = [1.3 ,1.15, 1], yScaleList = [1.5, 1.3, 1.1], rotationList = [-20,-10,0,10,20], display = False)
# imageProcessor.generateArtificialData(symbol = '+', xScaleList = [1.1], yScaleList = [1.1,1], rotationList = [-10,0,10], display = False)
# imageProcessor.generateArtificialData(symbol = '-', xScaleList = [1.1], yScaleList = [1.1,1], rotationList = [-10,0,10], display = False)
# imageProcessor.generateArtificialData(symbol = '*', xScaleList = [1.45, 1.3 ,1.15, 1], yScaleList = [1.7, 1.5, 1.3, 1.1], rotationList = [-20,-10,0,10,20], display = False)
# imageProcessor.generateArtificialData(symbol = ':', xScaleList = [1.75, 1.6, 1.45, 1.3 ,1.15, 1], yScaleList = [1.7, 1.5, 1.3, 1.1], rotationList = [-20,-10,0,10,20], display = False)
# imageProcessor.generateArtificialData(symbol = ']', xScaleList = [1.75, 1.6, 1.45, 1.3 ,1.15, 1], yScaleList = [1.7, 1.5, 1.3, 1.1], rotationList = [-20,-10,0,10,20], display = False)
# imageProcessor.generateArtificialData(symbol = '[', xScaleList = [1.75, 1.6, 1.45, 1.3 ,1.15, 1], yScaleList = [1.7, 1.5, 1.3, 1.1], rotationList = [-20,-10,0,10,20], display = False)
# imageProcessor.getDataSummary()
# imageProcessor.exportDataSet(extendedFilePath)
# imageProcessor.displayDataGroup(start=0, end=100,symbol = "]", gridWidth = 30)

sourcePath = "/Users/michelsmacbookpro/Desktop/Symbol Images Selected"
filePath = "/Users/michelsmacbookpro/Desktop/InputOutputDatapoints.npy"
extendedFilePath = "/Users/michelsmacbookpro/Desktop/InputOutputDatapoints_Extended.npy"
#imageProcessor = ImageProcessor().createDataSet(sourcePath,False,True).exportDataSet(filePath)
imageProcessor = ImageProcessor().importDataSet(filePath) 
imageProcessor.getDataSummary()
imageProcessor.reduceDatasets(200)
imageProcessor.getDataSummary()
outputMap = '0123456789+-*:[]'

for symbol in outputMap:
  imageProcessor.generateArtificialData(symbol = symbol, xScaleList = [1.6, 1,45, 1.3, 1.15, 1], yScaleList = [1.6, 1.45, 1.3, 1.15], rotationList = [-20 -10,0,10,20], display = False)
imageProcessor.getDataSummary()
imageProcessor.exportDataSet(extendedFilePath)







# filePath = "/Users/michelsmacbookpro/Desktop/InputOutputDatapoints.npy"
# outputMap = '0123456789+-*:[]'
# layers = (784,45,16)
# neuralNetwork = NeuralNetworkManipulator().create(layers, outputMap)
# neuralNetwork.manipulator.train(trainingDataPath=filePath,epochs=10,miniBatchSize=40,eta=2, validationDataPath = filePath)

