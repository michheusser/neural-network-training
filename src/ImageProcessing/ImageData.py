import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from .ImageManipulator import ImageManipulator 

class ImageData:
    def __init__(self, dataArray = None):
        self.data = dataArray
        self.manipulator = ImageManipulator(self)

    def loadImage(self, path):
        self.data = np.array(ImageOps.invert(Image.open(path)))
        self.normalizeData()
        self.cleanData()
        return self

    def normalizeData(self):
        maxValue = np.amax(self.data)
        self.data = self.data/maxValue
        return self

    def cleanData(self):
        self.data = np.round(self.data).astype(int)
        return self

    def printData(self):
        print(self.data)
        return self

    def display(self):
        plt.imshow(self.data)
        plt.show()
        return self

