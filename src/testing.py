from ImageProcessing.ImageData import ImageData
from ImageProcessing.ImageProcessor import ImageProcessor
import numpy as np



path = "/Users/michelsmacbookpro/Library/Mobile Documents/com~apple~CloudDocs/Symbol-Calculator/src/test.jpg"
path2 = "/Users/michelsmacbookpro/Library/Mobile Documents/com~apple~CloudDocs/Symbol-Calculator/src/test2.jpg"
#image = ImageData().loadImage(path).display().manipulator.shift(-10,-10).display().manipulator.align().display()
#image = ImageData().loadImage(path).display().manipulator.addMargins(20,20).display().manipulator.wrap(3,3).display().manipulator.scale(50,50).display()
#image = ImageData().loadImage(path).display().manipulator.addMargins(0,0).display().manipulator.scale(10,10).display()
#image = ImageData().loadImage(path).manipulator.shift(-27,0).manipulator.wrap(0,1).manipulator.crop(2,2).display().manipulator.scale(6,2).display()
#image = ImageData().loadImage(path).display().manipulator.shift(-15,0).display().manipulator.addMargins(3,3).display().manipulator.fit(50,50,keepRatio = True).display()
#image = ImageData().loadImage(path).manipulator.addMargins(3,3).display().manipulator.fitStroke(5,10,0,0,True).display()
#image = ImageData().loadImage(path).display().manipulator.wrap().manipulator.addMargins(0,0).display().manipulator.ScaleStroke(20,90).display()
#image = ImageData().loadImage(path).manipulator.wrap().manipulator.shift(-8,0).manipulator.crop(3,2).display().manipulator.scale(6,4,True).display()
#image = ImageData().loadImage(path).display().manipulator.fit(45,45,0,0,False,True).display()
#image = ImageData().loadImage(path).manipulator.wrap().manipulator.shift(-8,0).manipulator.crop(3,4).display().manipulator.fit(100,2,0,0,False,True).display().printData()

#image = ImageData().loadImage(path2).manipulator.wrap().manipulator.shift(-8,-41).manipulator.wrap().manipulator.crop(5,2).manipulator.wrap().display().manipulator.fit(2,4,0,0,False,True).display()

#image = ImageData().loadImage(path2).manipulator.wrap().manipulator.shift(-9,-41).display().manipulator.crop(2,1).display().manipulator.fit(30,3,0,0,False,True).display()

#for i in range(5,100,5):
#  for j in range(5,100,5):
#    print(i,j)
#    image = ImageData().loadImage(path).manipulator.addMargins(10,10).manipulator.fit(xFields=i,yFields=j,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).display()
#image = ImageData().loadImage(path).display().manipulator.wrap().display().manipulator.fit(xFields=90,yFields=90,xMargin=0,yMargin=0,keepRatio=False,scaleStroke=True).display()

#for i in range(3,100,12):
#  for j in range(3,100,12):
#    print(i,j)
#    image = ImageData().loadImage(path).manipulator.wrap().manipulator.fit(xFields=i,yFields=j,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).display().printData(rangeY = (320,370), rangeX=(0,100))

#image = ImageData().loadImage(path2).manipulator.wrap().manipulator.shift(-9,-41).manipulator.crop(2,2).manipulator.shift(0,1).display().manipulator.rotate(90,True).display()

#ERROR!
#image = ImageData().loadImage(path2).manipulator.wrap().display().manipulator.rotate(45,True).display()

# ERROR! Fit is not really making the image from the given size
#image = ImageData().loadImage(path2).manipulator.wrap().display().manipulator.fit(xFields=200,yFields=200,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).display().manipulator.rotate(30,True).display().manipulator.fit(xFields=100,yFields=200,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).display()

# ERROR!
#image = ImageData().loadImage(path2).manipulator.wrap().display().manipulator.fit(xFields=200,yFields=200,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).manipulator.wrap().display().manipulator.rotate(50,True).display().manipulator.fit(xFields=500,yFields=500,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).display()

# ERROR Spaces after fitting to 500x500
#image = ImageData().loadImage(path2).manipulator.wrap().display().manipulator.fit(xFields=45,yFields=45,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).manipulator.wrap().display().manipulator.rotate(90,True).display().manipulator.fit(xFields=500,yFields=500,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).display()

# ERROR
#image = ImageData().loadImage(path2).manipulator.wrap().display().manipulator.rotate3(40,True).display()

#for i in range(0,95,5):
#  image = ImageData(np.array([[1,1,1],[1,0,1],[1,0,1],[1,0,1],[1,0,1],[1,1,1]])).manipulator.addMargins(0,0).display()
#  print(i)
#  image.manipulator.rotate3(i,True).display()

#image = ImageData(np.array([[1,1,1,0,1,1,1]])).manipulator.addMargins(0,0).display().manipulator.rotate4(120).display()
#image = ImageData(np.array([[0,1]])).manipulator.addMargins(0,0).display().manipulator.rotate4(-225).display()

image = ImageData().loadImage(path2).manipulator.wrap().manipulator.fit(xFields=50,yFields=200,xMargin=0,yMargin=0,keepRatio=True,scaleStroke=True).manipulator.wrap().display().manipulator.rotate4(45).display()