# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

from datatools.dataset_processor import DatasetProcessor
import sys

# Agglomeration of symbols in an image works recursively. When recursion limit is achieved,
# the algorithm breaks, and goes to the next symbol. Use carefully to avoid a stack overflow.
sys.setrecursionlimit(2000)


# SEGMENTATION OF SCANNED IMAGES INTO INDIVIDUAL IMAGES WITH SYMBOLS
sourcePath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/Bulk Processed/"
destinationPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/Single Files/"

outputMap = '0123456789+-*%[]' 
for char in outputMap:
  datasetProcessor = DatasetProcessor().segmentBatch(sourcePath+char, destinationPath+char, '.png',False)

# DATA AUGMENTATION BY SCALING IMAGES TO NEW INDIVIDUAL IMAGES
sourcePath = "/Users/michelsmacbookpro/Desktop/Projects/Single Files/"
destinationPath = "/Users/michelsmacbookpro/Desktop/Projects/Scaled Images/"
xScaleList = [1,1.15,1.3]
yScaleList = [1,1.15,1.3]
for char in outputMap:
  datasetProcessor = DatasetProcessor().extendDataSet(sourcePath+char, destinationPath+char,[1,1.2,1.3], [1,1.2,1.3], '.png')

# MAIN DATASET CREATION (INPUT-OUTPUT)
sourcePath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/All data"
destinationPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet"
datasetProcessor = DatasetProcessor().createDataSet(sourcePath).exportDataSet(destinationPath,asTuple=False)

# TRAINING, VALIDATION, AND TEST DATASETS CREATION OUT OF MAIN DATASET
datasetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet.npy"
datasetProcessor = DatasetProcessor().importDataSet(datasetPath)
datasetProcessor.getDataSummary()
trainingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_training_tuples.npy"
trainingSetLength = 0.60
validationSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_validation_tuples.npy"
validationSetLength = 0.20
testingSetPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images/CompleteDataSet_testing_tuples.npy"
datasetProcessor.createLearningSets(trainingSetPath, trainingSetLength, validationSetPath, validationSetLength, testingSetPath, asTuples=True)
datasetProcessor.importDataSet(trainingSetPath).getDataSummary()
datasetProcessor.importDataSet(validationSetPath).getDataSummary()
datasetProcessor.importDataSet(testingSetPath).getDataSummary()