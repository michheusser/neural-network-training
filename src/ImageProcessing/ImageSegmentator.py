import numpy as np
import math
from .ImageData import ImageData

class ImageSegmentator:
    def __init__(self,imageData = None):
        self.imageData = imageData
        self.imageSegments = []
    
    #def agglomerate(self, bufferImage, imageSegment, x, y):
        #imageSegment.data[y][x] = bufferImage[y][x]
        #bufferImage[y][x] = 0
        #positions = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0, 1),(1, -1),(1, 0),(1, 1)]
        #for position in positions:
        #    this.agglomerate(grid, gridSegment, x + position[0], y + position[1])
    def loadImageData(self, imageData):
        self.imageData = imageData
        return self 

    def agglomerate(self,x,y,bufferImage, imageSegment = None):
        if not bufferImage.data[y][x]:
            return imageSegment
        if not imageSegment:
            print("Segmentating body...")
            imageSegment = ImageData(np.zeros(self.imageData.data.shape))
        imageSegment.data[y][x] = bufferImage.data[y][x]
        bufferImage.data[y][x] = 0
        positions = [(-1, -1),(-1, 0),(-1, 1),(0, -1),(0, 1),(1, -1),(1, 0),(1, 1)]
        for position in positions:
            if 0<= x+position[1] < self.imageData.data.shape[1] and y+position[0] < self.imageData.data.shape[0]: 
                self.agglomerate(x+position[1],y+position[0],bufferImage,imageSegment)
        return imageSegment

    def createSegments(self):
        self.clearSegments()
        imageCopy = self.imageData.copy()
        counter = 0
        for y in range(0,self.imageData.data.shape[0]):
            for x in range(0,self.imageData.data.shape[1]): 
                #if self.imageData.data[y][x]:
                #    imageSegment = ImageData(np.zeros(self.imageData.data.shape))
                #    self.agglomerate(imageCopy, imageSegment, x, y)
                #    self.imageSegments.push(imageSegment)
                imageSegment = self.agglomerate(x,y,imageCopy)
                if imageSegment:
                    counter += 1
                    print("Segment Created. Segments: "+ str(counter))
                    self.imageSegments.append(imageSegment)
        return self.imageSegments

    def clearSegments(self):
        self.imageSegments = []

