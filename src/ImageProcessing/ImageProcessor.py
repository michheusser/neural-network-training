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
                    print("Folder name: " + root.rpartition("/")[2] + " files Processed: " + str(counter) + " of " + str(len(files)))
        
        print("Total Datapoints: " + str(len(self.dataSet)))
        if removeDuplicates:
            self.removeDuplicates()
        return self

    def exportDataSet(self, destinationPath):
        np.save(destinationPath,self.dataSet)
        return self

    def importDataSet(self,sourcePath):
        self.dataSet = np.load(sourcePath)
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
        return (symbols, counter)

    def displayDataGroup(self,indexArray,symbol):
        listDataPoints = []
        for dataPoint in self.dataSet:
            if(dataPoint.output == symbol):
                listDataPoints.append(dataPoint)
        indexArray
        x = 10
        y = math.ceil(len(indexArray)/x)
        fig, axs = plt.subplots(x, y)
        for i in range(0,len(indexArray)):
            axs[i%x,math.floor(i/x)].imshow(listDataPoints[indexArray[i]].input)
            axs[i%x,math.floor(i/x)].get_xaxis().set_visible(False)
            axs[i%x,math.floor(i/x)].get_yaxis().set_visible(False)
        plt.show()
    
    

    
