import os
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import functools
from .ImageData import ImageData
from .InputOutputData import InputOutputData
from .ImageSegmentator import ImageSegmentator

class ImageProcessor:
    def __init__(self):
        self.dataSet = []
        
    def createDataPoint(self, imageData,symbol):
        return InputOutputData(imageData,symbol)

    def addData(self, imageData,symbol):
        self.dataSet.append(self.createDataPoint(imageData,symbol))
        return self
    def areEqual(self,indexI, indexJ):
        if self.dataSet[indexI].output == self.dataSet[indexJ].output:
            return np.array_equal(self.dataSet[indexI].input, self.dataSet[indexJ].input)
        return False

    def removeDuplicates(self):
        dataPointsBefore = len(self.dataSet)
        print("Removing duplicates...")
        self.dataSet = list(set(self.dataSet))
        dataPointsNow = len(self.dataSet)
        print(str(dataPointsBefore - dataPointsNow) + " duplicates found and removed")
        print("Total Datapoints: " + str(dataPointsNow))
        return self

    def processImageData(self,imageData):
        return imageData.manipulator.fit(xFields=28, yFields=28, xMargin = 0, yMargin = 0, keepRatio = True,scaleStroke=True)

    def createDataSet(self,sourcePath,display=False, removeDuplicates = False):
        for (root,dirs,files) in os.walk(sourcePath, topdown = True): 
            counter = 0
            print("Folder name: " + root.rpartition("/")[2] + ", files: " + str(len(files)) )
            for file in files:
                path = root + "/" + file
                if path.endswith(".png"):
                    imageData = self.processImageData(ImageData().loadImage(path))
                    if display:
                        imageData.display()
                    self.addData(imageData.data,root.rpartition("/")[2])
                    counter += 1
                    print("Folder name: " + root.rpartition("/")[2] + ", files Processed: " + str(counter) + " of " + str(len(files)))
        return self
        
        print("Total Datapoints: " + str(len(self.dataSet)))
        if removeDuplicates:
            self.removeDuplicates()
        return self

    def exportDataSet(self, destinationPath, dataSet=None):
        if dataSet:
            np.save(destinationPath,dataSet)
        else:
            np.save(destinationPath,self.dataSet)
        return self

    def generateArtificialData(self,symbol,xScaleList,yScaleList,rotationList,display = False, export = False):
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
                            newArtificialData.append(self.createDataPoint(newImageData.data,symbol))
                    for yScale in yScaleList:
                        if(rotation!=0 or yScale !=1):
                            newImageData = ImageData(dataPoint.copy().input)
                            scaledX = shapeX
                            scaledY = math.floor(shapeY*yScale)
                            newImageData.manipulator.scale(scaledX,scaledY,True).manipulator.rotate(rotation,True,False).manipulator.fit(shapeX,shapeY,0,0,True,True)
                            if display:
                                newImageData.display()
                            newArtificialData.append(self.createDataPoint(newImageData.data,symbol))
        self.dataSet.extend(newArtificialData)
        return self

    def importDataSet(self,sourcePath):
        self.dataSet = list(np.load(sourcePath))
        return self

    def getDataSummary(self):
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
        outputs = []
        for dataPoint in self.dataSet:
            if not (dataPoint.output in outputs):
                outputs.append(dataPoint.output)
        return outputs

    def reduceDatasets(self,size):
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

    def scalePermutation(self, xScaleList, yScaleList):
        permutations = []
        for xScale in xScaleList:
            for yScale in yScaleList:
                permutations.append((xScale,yScale))
        return permutations

    def segmentImage(self, sourcePath, destinationPath, display = False):
        if display:
            segments = ImageSegmentator(ImageData().loadImage(sourcePath).display()).createSegments()
        else:
            segments = ImageSegmentator(ImageData().loadImage(sourcePath)).createSegments()
        for i,segment in enumerate(segments):
            print("Processing image: " + str(i+1) + " of " + str(len(segments)))
            self.processImageData(segment)
            segment.exportImage(destinationPath,os.path.basename(sourcePath).rpartition(".")[0]+"_"+str(i))
        return self

    def createScaledCopies(self, sourcePath, destinationPath, xScaleList, yScaleList):
        image = ImageData().loadImage(sourcePath)
        print("Creating scaled images...")
        counter = 0
        for scaling in self.scalePermutation(xScaleList,yScaleList):
            if not scaling == (1,1):
                image.manipulator.scale(math.ceil(image.data.shape[0]*scaling[0]),math.ceil(image.data.shape[1]*scaling[1]),True)
                self.processImageData(image)
                image.exportImage(destinationPath,os.path.basename(sourcePath).rpartition(".")[0]+"_scaled_"+"x"+str(scaling[0]).replace('.','_')+"y"+str(scaling[1]).replace('.','_'))
                counter += 1
        print("Created " + str(counter) + " scaled images")
        return self

    def segmentBatch(self, sourcePath, destinationPath, extension = '.png',display=False):
        print("*****************")
        print("Batch segmentation starting...")
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
                        self.segmentImage(path,destinationPath,display)
                        counter += 1
                        print("Folder name: " + root.rpartition("/")[2] + ", files Processed: " + str(counter) + " of " + str(len(files)))

    def extendDataSet(self, sourcePath, destinationPath, xScaleList, yScaleList, extension = '.png'):
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
                        self.createScaledCopies(path, destinationPath, xScaleList, yScaleList)
                        counter += 1
                        print("Folder name: " + root.rpartition("/")[2] + ", files Processed: " + str(counter) + " of " + str(len(files)))
    
    def removeSmallElements(self, pixelThreshold, displayDeleted = False):
        newDataSet = []
        for imageData in self.dataSet:
            pixels = functools.reduce(lambda x,y: x+1 if y else x,imageData.data.flatten(), 0)
            if pixels > pixelThreshold:
                newDataSet.append(imageData)
            elif displayDeleted:
                imageData.display()
        self.dataSet = newDataSet
        return self

    def createLearningSets(self, trainingSetPath, trainingSetLength, validationSetPath, validationSetLength, testingSetPath, shuffle = True):
        if shuffle:
            print("Shuffling dataset...")
            random.shuffle(self.dataSet)
            print("Dataset shuffled.")
        self.exportDataSet(trainingSetPath,self.dataSet[0:math.floor(len(self.dataSet)*trainingSetLength)])
        self.exportDataSet(validationSetPath,self.dataSet[math.floor(len(self.dataSet)*trainingSetLength):math.floor(len(self.dataSet)*(trainingSetLength+validationSetLength))])
        self.exportDataSet(testingSetPath,self.dataSet[math.floor(len(self.dataSet)*(trainingSetLength+validationSetLength)):len(self.dataSet)])
        return self
    
    
    



    
