import os
import numpy as np
import math
import matplotlib.pyplot as plt
from .ImageData import ImageData
from .InputOutputData import InputOutputData

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
                if path.endswith(".jpg"):
                    imageData = self.processImageData(ImageData().loadImage(path))
                    if display:
                        imageData.display()
                    self.addData(imageData.data,root.rpartition("/")[2])
                    counter += 1
                    print("Folder name: " + root.rpartition("/")[2] + ", files Processed: " + str(counter) + " of " + str(len(files)))
        
        print("Total Datapoints: " + str(len(self.dataSet)))
        if removeDuplicates:
            self.removeDuplicates()
        return self

    def exportDataSet(self, destinationPath):
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

    def displayDataGroup(self,start, end,symbol, gridWidth):
        listDataPoints = [dataPoint for dataPoint in self.dataSet if dataPoint.output == symbol]
        x = gridWidth
        y = math.ceil((end-start+1)/x)
        fig, axs = plt.subplots(x, y)
        for i in range(start,end):
            axs[i%x,math.floor(i/x)].imshow(listDataPoints[i].input)
            axs[i%x,math.floor(i/x)].get_xaxis().set_visible(False)
            axs[i%x,math.floor(i/x)].get_yaxis().set_visible(False)
        plt.show()
    



    
