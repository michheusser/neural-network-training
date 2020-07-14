# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import functools
from .image_processing.image_data import ImageData
from .io_datapoint import InputOutputData
from .image_processing.segmentator import ImageSegmentator

class DatasetProcessor:
    '''
    Provides tools to generate datasets out of images, which includes:
    - Importing and segmenting images with more than one symbol and save as individual images
    - Generate new artificial data through scaling and rotation
    - Importing or exporting created datasets
    - Create training, validation and test sets out of an existing dataset
    - Removing duplicates
    - Removing images with the amount of filled pixels smaller than a threshold value
    '''
    def __init__(self):
        self.dataSet = []
        
    def _createDataPoint(self, imageData,symbol):
        '''Returns a datapoint with given arguments of type InputOutputData'''
        return InputOutputData(imageData,symbol)

    def _addData(self, imageData,symbol):
        '''Creates a datapoint with given arguments of type InputOutputData
        and adds it to the class dataset'''
        self.dataSet.append(self._createDataPoint(imageData,symbol))
        return self

    def _areEqual(self,indexI, indexJ):
        '''Compares if two dataset points contain the same image (as matrix) as an input'''
        if self.dataSet[indexI].output == self.dataSet[indexJ].output:
            return np.array_equal(self.dataSet[indexI].input, self.dataSet[indexJ].input)
        return False

    def _removeDuplicates(self):
        '''Removes duplicate datapoints (inputs and outputs are both equal) in the class dataset'''
        dataPointsBefore = len(self.dataSet)
        print("Removing duplicates...")
        self.dataSet = list(set(self.dataSet))
        dataPointsNow = len(self.dataSet)
        print(str(dataPointsBefore - dataPointsNow) + " duplicates found and removed")
        print("Total Datapoints: " + str(dataPointsNow))
        return self

    def _processImageData(self,imageData):
        '''Processes an image in the defined way (e.g. 28x28 pixels and no margins) way. 
        Normally used to standarize a batch of pictures to be consistent in size and dimensions'''
        return imageData.manipulator.fit(xFields=28, yFields=28, xMargin = 0, yMargin = 0, keepRatio = True,scaleStroke=True)

    def createDataSet(self,sourcePath,display=False, removeDuplicates = False):
        '''Goes through all '.png' images within a root folder (sourcePath), including images in nested folders,
        and creates datasets using the folder name of each image as the output value in the datapoint. The images
        are pre-processed according to the method _processImageData(imageData)'''
        for (root,dirs,files) in os.walk(sourcePath, topdown = True): 
            counter = 0
            print("Folder name: " + root.rpartition("/")[2] + ", files: " + str(len(files)) )
            for file in files:
                path = root + "/" + file
                if path.endswith(".png"):
                    imageData = self.processImageData(ImageData().loadImage(path))
                    if display:
                        imageData.display()
                    self._addData(imageData.data,root.rpartition("/")[2])
                    counter += 1
                    print("Folder name: " + root.rpartition("/")[2] + ", files Processed: " + str(counter) + " of " + str(len(files)))
        print("Total Datapoints: " + str(len(self.dataSet)))
        if removeDuplicates:
            self._removeDuplicates()
        return self

    def _dataToTuple(self,dataSet=None):
        '''Returns the dataset in the argument as a list of tuples of the following form (input, output).
        If no argument is passed returns the class dataset'''
        if dataSet:
            return [dataPoint.toTuple() for dataPoint in dataSet]
        else:
            return [dataPoint.toTuple() for dataPoint in self.dataSet]

    def exportDataSet(self, destinationPath, dataSet=None, asTuple=False):
        '''Exports the dataset in the argument as a .npy file in the specified destination path,
        either as a list of InputOutputData objects or as a list of tuples. If no dataset is
        passed in the arguments, the class dataset is exported'''
        if dataSet:
            if asTuple:
                np.save(destinationPath,self._dataToTuple(dataSet))
            else:
                np.save(destinationPath,dataSet)
        else:
            if asTuple:
                np.save(destinationPath,self._dataToTuple)
            else:
                np.save(destinationPath,self.dataSet)
        return self

    def importDataSet(self,sourcePath):
        '''Imports a .npy dataset from the specified path previously exported using the exportDataSet method. The file
        has to contain a list of InputOutputData datapoints (no tuples!)'''
        self.dataSet = list(np.load(sourcePath))
        return self

    def generateArtificialData(self,symbol,xScaleList,yScaleList,rotationList,display = False, export = False):
        '''Extends the class dataset to contain variations of every datapoint according to all permutations
        of the given scalings in x,y and rotations. If a scaling/rotation does not change an image, the image
        is ommited to avoid duplication (e.g. scalingx = 1, scalingy = 1, rotation = 0)'''
        newArtificialData = []
        counter = 0
        for dataPoint in self.dataSet:
            if(dataPoint.output == symbol):
                print("Generating new data for '" + str(symbol)+ "'. Points processed: ", str(counter))
                counter += 1
                shapeX = dataPoint.input.shape[1]
                shapeY = dataPoint.input.shape[0]
                for rotation in rotationList:
                    for xScale in xScaleList:
                        if(rotation!=0 or xScale !=1):
                            newImageData = ImageData(dataPoint.copy().input)
                            scaledX = math.floor(shapeX*xScale)
                            scaledY = shapeY
                            newImageData.manipulator.scale(scaledX,scaledY,True).manipulator.rotate(rotation,True,False).manipulator.fit(shapeX,shapeY,0,0,True,True)
                            if display:
                                newImageData.display()
                            newArtificialData.append(self._createDataPoint(newImageData.data,symbol))
                    for yScale in yScaleList:
                        if(rotation!=0 or yScale !=1):
                            newImageData = ImageData(dataPoint.copy().input)
                            scaledX = shapeX
                            scaledY = math.floor(shapeY*yScale)
                            newImageData.manipulator.scale(scaledX,scaledY,True).manipulator.rotate(rotation,True,False).manipulator.fit(shapeX,shapeY,0,0,True,True)
                            if display:
                                newImageData.display()
                            newArtificialData.append(self._createDataPoint(newImageData.data,symbol))
        self.dataSet.extend(newArtificialData)
        return self

    def getDataSummary(self):
        '''Summarizes datapoints according to their outputs'''
        symbols = []
        for dataPoint in self.dataSet:
            if not (dataPoint.output in symbols):
                symbols.append(dataPoint.output)
        
        counter = [0]*len(symbols)
        for dataPoint in self.dataSet:
            counter[symbols.index(dataPoint.output)]+=1
        print("Total Datapoints: " + str(len(self.dataSet)))
        for i in range(0,len(symbols)):
            print("Symbol: " + str(symbols[i]) + ", entries: " + str(counter[i]))
        return [(symbol, count) for symbol, count in zip(symbols,counter)]

    def getOutputs(self):
        '''Returns a list of all possible outputs in the class dataset'''
        outputs = []
        for dataPoint in self.dataSet:
            if not (dataPoint.output in outputs):
                outputs.append(dataPoint.output)
        return outputs

    def reduceDatasets(self,size):
        '''Reduces the class dataset to contain at most a certain amount of datapoints 
        of every possible output (specified by the argument size)'''
        reducedDataSet = []
        print("Outputs: " + str(self.getOutputs))
        for output in self.getOutputs():
            counter = 0
            print("Reducing set for '" + str(output) + "'")
            for dataPoint in self.dataSet:
                if output == dataPoint.output:
                    reducedDataSet.append(dataPoint)
                    counter += 1
                if counter == size:
                    break
        self.dataSet = reducedDataSet

    def _scalePermutation(self, xScaleList, yScaleList):
        '''Generates all permutations of x and y scalings passed as arguments'''
        permutations = []
        for xScale in xScaleList:
            for yScale in yScaleList:
                permutations.append((xScale,yScale))
        return permutations

    def _segmentImage(self, sourcePath, destinationPath, display = False):
        '''Segments an image in the sourcePath with several symbols and exports the segmented
        symbols into individual images in the destinationPath
        '''
        if display:
            segments = ImageSegmentator(ImageData().loadImage(sourcePath).display()).createSegments()
        else:
            segments = ImageSegmentator(ImageData().loadImage(sourcePath)).createSegments()
        for i,segment in enumerate(segments):
            print("Processing image: " + str(i+1) + " of " + str(len(segments)))
            self._processImageData(segment)
            segment.exportImage(destinationPath,os.path.basename(sourcePath).rpartition(".")[0]+"_"+str(i))
        return self

    def _createScaledCopies(self, sourcePath, destinationPath, xScaleList, yScaleList):
        '''Takes an image from the source path and creates scaled copies out of all the 
        permutations of the passed scaling lists in the speficied destination path'''
        image = ImageData().loadImage(sourcePath)
        print("Creating scaled images...")
        counter = 0
        for scaling in self._scalePermutation(xScaleList,yScaleList):
            if not scaling == (1,1):
                image.manipulator.scale(math.ceil(image.data.shape[0]*scaling[0]),math.ceil(image.data.shape[1]*scaling[1]),True)
                self._processImageData(image)
                image.exportImage(destinationPath,os.path.basename(sourcePath).rpartition(".")[0]+"_scaled_"+"x"+str(scaling[0]).replace('.','_')+"y"+str(scaling[1]).replace('.','_'))
                counter += 1
        print("Created " + str(counter) + " scaled images")
        return self

    def segmentBatch(self, sourcePath, destinationPath, extension = '.png',display=False):
        '''Segments all images within a certain folder (including nested folders) and exports
        the segmented images to the destination path'''
        print("*****************")
        print("Batch segmentation starting...")
        print("Source path: " + str(sourcePath))
        print("Destination path: " + str(destinationPath))
        for (root,dirs,files) in os.walk(sourcePath, topdown = True): 
                counter = 0
                print("Root folder: " + str(root) + ", files: " +str(len(files)))
                for file in files:
                    print("File to process: " + str(file))
                    path = root + "/" + file
                    if path.endswith(extension):
                        self._segmentImage(path,destinationPath,display)
                        counter += 1
                        print("Folder name: " + root.rpartition("/")[2] + ", files Processed: " + str(counter) + " of " + str(len(files)))

    def extendDataSet(self, sourcePath, destinationPath, xScaleList, yScaleList, extension = '.png'):
        '''Extends the class dataset to contain variations of every datapoint according to all permutations
        of the given scalings in x,y. If a scaling does not change an image, the image
        is ommited to avoid duplication (e.g. scalingx = 1, scalingy = 1)'''
        print("*****************")
        print("Dataset extension starting...")
        print("Source path: " + str(sourcePath))
        print("Destination path: " + str(destinationPath))
        for (root,dirs,files) in os.walk(sourcePath, topdown = True): 
                counter = 0
                print("Root folder: " + str(root) + ", files: " +str(len(files)))
                #print("Folder name: " + root.rpartition("/")[2] + ", files: " + str(len(files)) )
                for file in files:
                    print("File to process: " + str(file))
                    path = root + "/" + file
                    if path.endswith(extension):
                        self._createScaledCopies(path, destinationPath, xScaleList, yScaleList)
                        counter += 1
                        print("Folder name: " + root.rpartition("/")[2] + ", files Processed: " + str(counter) + " of " + str(len(files)))
    
    def removeSmallElements(self, pixelThreshold, displayDeleted = False):
        '''Removes datapoints in the class dataset, where the image has less than the specified
        amount of filled pixels'''
        newDataSet = []
        for imageData in self.dataSet:
            pixels = functools.reduce(lambda x,y: x+1 if y else x,imageData.data.flatten(), 0)
            if pixels > pixelThreshold:
                newDataSet.append(imageData)
            elif displayDeleted:
                imageData.display()
        self.dataSet = newDataSet
        return self

    def createLearningSets(self, trainingSetPath, trainingSetLength, validationSetPath, validationSetLength, testingSetPath, shuffle=True, asTuples=False):
        '''Generates training, validation, and test sets out of the class datasets according
        to the specified relative length of the original dataset'''
        if shuffle:
            print("Shuffling dataset...")
            random.shuffle(self.dataSet)
            print("Dataset shuffled.")
        self.exportDataSet(trainingSetPath,self.dataSet[0:math.floor(len(self.dataSet)*trainingSetLength)],asTuple=asTuples)
        self.exportDataSet(validationSetPath,self.dataSet[math.floor(len(self.dataSet)*trainingSetLength):math.floor(len(self.dataSet)*(trainingSetLength+validationSetLength))],asTuple=asTuples)
        self.exportDataSet(testingSetPath,self.dataSet[math.floor(len(self.dataSet)*(trainingSetLength+validationSetLength)):len(self.dataSet)],asTuple=asTuples)
        return self
    
    
    



    
