from ImageProcessing.ImageProcessor import ImageProcessor
import sys

sys.setrecursionlimit(2000)

outputMap = '0123456789+-*%[]'

sourcePath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images (Own Dataset)/Bulk Processed/"
destinationPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images (Own Dataset)/Single Files/"

for char in outputMap:
  imageProcessor = ImageProcessor().segmentBatch(sourcePath+char, destinationPath+char, '.png',False)


