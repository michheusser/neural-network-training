from ImageProcessing.ImageData import ImageData
from ImageProcessing.ImageProcessor import ImageProcessor


sourcePath = "/Users/michelsmacbookpro/Desktop/Symbol Images Selected"
filePath = "/Users/michelsmacbookpro/Desktop/InputOutputDatapoints.npy"
#imageProcessor = ImageProcessor().createDataSet(sourcePath,False,True).exportDataSet(filePath)
imageProcessor = ImageProcessor().importDataSet(filePath)
imageProcessor.getDataSummary()
imageProcessor.generateArtificialData(symbol = ']', xScaleList = [1.7, 1.5, 1.3 ,1.15, 1], yScaleList = [1.7, 1.5, 1.3, 1.15], rotationList = [-20,-10,0,10,20], display = True)
#imageProcessor.displayDataGroup(list(range(0,100)),"]")

