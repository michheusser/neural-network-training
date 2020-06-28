from ImageProcessing.ImageProcessor import ImageProcessor
import sys

sys.setrecursionlimit(2000)

outputMap = '0123456789+-*%[]'

#sourcePath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/Bulk Processed/"
#destinationPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/Single Files/"

#for char in outputMap:
#  imageProcessor = ImageProcessor().segmentBatch(sourcePath+char, destinationPath+char, '.png',False)

#sourcePath = "/Users/michelsmacbookpro/Desktop/Projects/Single Files/"
# destinationPath = "/Users/michelsmacbookpro/Desktop/Projects/Scaled Images/"
# xScaleList = [1,1.15,1.3]
# yScaleList = [1,1.15,1.3]
# for char in outputMap:
#   imageProcessor = ImageProcessor().extendDataSet(sourcePath+char, destinationPath+char,[1,1.2,1.3], [1,1.2,1.3], '.png')

sourcePath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/All data"
destinationPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small"
imageProcessor = ImageProcessor().createDataSet(sourcePath).exportDataSet(destinationPath)

datasetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small.npy"
imageProcessor = ImageProcessor().importDataSet(datasetPath)
imageProcessor.getDataSummary()
trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_training.npy"
trainingSetLength = 0.6
validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_validation.npy"
validationSetLength = 0.2
testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_small_testing.npy"
imageProcessor.createLearningSets(trainingSetPath, trainingSetLength, validationSetPath, validationSetLength, testingSetPath)
imageProcessor.importDataSet(trainingSetPath).getDataSummary()
imageProcessor.importDataSet(validationSetPath).getDataSummary()
imageProcessor.importDataSet(testingSetPath).getDataSummary()