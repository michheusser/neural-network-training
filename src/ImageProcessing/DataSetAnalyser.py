import os
import numpy as np
import math
import matplotlib.pyplot as plt
from .ImageData import ImageData
from .InputOutputData import InputOutputData

class DataSetAnalyser:
    def __init__(self):
        self.dataSet = []

    def displayDataGroup(self,indexArray,symbol):
        listDataPoints = []
        for dataPoint in self.dataSet:
            if(dataPoint.output == symbol):
                listDataPoints.append(dataPoint)
        x = 10
        y = math.ceil(len(indexArray)/x)
        fig, axs = plt.subplots(x, y)
        for i in range(0,len(indexArray)):
            axs[i%x,math.floor(i/x)].imshow(listDataPoints[indexArray[i]].input)
            axs[i%x,math.floor(i/x)].get_xaxis().set_visible(False)
            axs[i%x,math.floor(i/x)].get_yaxis().set_visible(False)
        plt.show()
    



    
