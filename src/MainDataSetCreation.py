from ImageProcessing.ImageProcessor import ImageProcessor
import sys

sys.setrecursionlimit(2000)
sourcePath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images (Own Dataset)/Bulk Processed/0"
destinationPath = "/Users/michelsmacbookpro/Desktop/Projects/Symbol Images (Own Dataset)/Single Files"
imageProcessor = ImageProcessor().segmentBatch(sourcePath, destinationPath)

