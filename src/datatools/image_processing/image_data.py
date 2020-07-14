# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np
import math
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from .manipulator import ImageManipulator 

class ImageData:
    '''Contains the data of a black/white image as a 2d numpy array and an image manipulator
    object which contains the tools for image processing'''
    def __init__(self, dataArray = None):
        self.data = dataArray
        self.manipulator = ImageManipulator(self)

    def copy(self):
        '''Deep copy of the object'''
        return ImageData(np.copy(self.data))

    def loadImage(self, path, greyScale = False):
        '''Loads an image, normalizes the values of its pixels to have 1 as the darkest pixel,
        and rounds the pixel values to ensure all pixels are either filled or unfilled.
        The value 0 represents a white/unfilled pixel'''
        self.data = np.array(ImageOps.invert(Image.open(path).convert('L')))
        self.normalizeData()
        self.cleanData()
        return self

    def exportImage(self, path, name):
        '''Exports image to the specified location'''
        ImageOps.invert(Image.fromarray((self.data*255).astype('uint8'), 'L')).save(path+'/'+name+'.png')
        return self

    def normalizeData(self):
        '''Normalizes the values to be between 0 and 1'''
        maxValue = np.amax(self.data)
        self.data = self.data/maxValue
        return self

    def cleanData(self):
        '''Rounds up pixel values to be either 0 or 1'''
        self.data = np.round(self.data).astype(int)
        return self

    def printData(self):
        '''Prints the matrix values of the image'''
        print(self.data)
        return self

    def display(self):
        '''Displays the image'''
        plt.imshow(self.data)
        plt.show()
        return self

