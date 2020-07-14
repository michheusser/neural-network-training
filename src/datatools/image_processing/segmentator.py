# Copyright 2020, Michel Heusser
# ALl rights reserved
# https://github.com/michheusser

import numpy as np
import math
from .datapoint import ImageData

class ImageSegmentator:
    '''Contains the tools to segment images that contain more than one symbol or
    agglomeration of pixels'''
    def __init__(self,imageData = None):
        self.imageData = imageData
        self.imageSegments = []
    
    def loadImageData(self, imageData):
        '''Loads the image (as an ImageData object) to the manipulator'''
        self.imageData = imageData
        return self 

    def _agglomerate(self,x,y,bufferImage, imageSegment = None):
        '''Recursive algorithm to agglomerate filled pixels that are next to each other'''
        if not bufferImage.data[y][x]:
            return imageSegment
        if not imageSegment:
            imageSegment = ImageData(np.zeros(self.imageData.data.shape))
        imageSegment.data[y][x] = bufferImage.data[y][x]
        bufferImage.data[y][x] = 0
        positions = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0, 1),(1, -1),(1, 0),(1, 1)]
        for position in positions:
            if 0<= x+position[1] < self.imageData.data.shape[1] and y+position[0] < self.imageData.data.shape[0]: 
                self._agglomerate(x+position[1],y+position[0],bufferImage,imageSegment)
        return imageSegment

    def createSegments(self):
        '''Fills the class segments list with the segmented images (of type ImageData) of the 
        loaded image
        '''
        self.clearSegments()
        imageCopy = self.imageData.copy()
        counter = 0
        for y in range(0,self.imageData.data.shape[0]):
            for x in range(0,self.imageData.data.shape[1]): 
                try:
                    imageSegment = self._agglomerate(x,y,imageCopy)
                except RecursionError:
                    print("Maximum depth reached. Segment not added")
                if imageSegment:
                    counter += 1
                    self.imageSegments.append(imageSegment)
        print("Segmentation finished: " + str(counter) + " segments created")
        return self.imageSegments

    def clearSegments(self):
        '''Deletes segments previously created'''
        self.imageSegments = []

