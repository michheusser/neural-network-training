import numpy as np
import math

class ImageManipulator:
    def __init__(self,imageData):
        self.imageData = imageData

    def limits(self):
        xMin = xMax = yMin = yMax = None
        for y in range(0,self.imageData.data.shape[0]):
            for x in range(0,self.imageData.data.shape[1]):
                if self.imageData.data[y][x]:
                    #print("x = " + str(x) + ", y = " + str(y) + " => " + str(self.data[y][x]))
                    xMax = x if xMax == None else (x if x>xMax else xMax)
                    yMax = y if yMax == None else (y if y>yMax else yMax)
                    xMin = x if xMin == None else (x if x<xMin else xMin)
                    yMin = y if yMin == None else (y if y<yMin else yMin)
        return xMin, xMax, yMin, yMax

    def shift(self,xOffset, yOffset):
        if xOffset == 0 and yOffset == 0:
            return self.imageData
    
        newData = np.zeros(self.imageData.data.shape)
        for y in range(0,self.imageData.data.shape[0]):
            for x in range(0,self.imageData.data.shape[1]):
                xNew = x-xOffset
                yNew = y-yOffset
                if (0 <= xNew < self.imageData.data.shape[1] and (0 <= yNew < self.imageData.data.shape[0])):
                    newData[y][x] = self.imageData.data[yNew][xNew]
        self.imageData.data = newData
        return self.imageData

    def align(self):
        xMin, xMax, yMin, yMax = self.limits()
        xMargin = math.ceil((self.imageData.data.shape[1] - (xMax - xMin + 1)) / 2)
        yMargin = math.ceil((self.imageData.data.shape[0] - (yMax - yMin + 1)) / 2)
        self.shift(xMargin-xMin,yMargin-yMin)
        return self.imageData
  
    def crop(self,xFields, yFields, align = False):
        newData = np.zeros((yFields,xFields))
        for y in range(0,newData.shape[0]):
            for x in range(0,newData.shape[1]):
                if ((0 <= x < self.imageData.data.shape[1]) and (0 <= y < self.imageData.data.shape[0])):
                    self.imageData.data[y][x]
                    newData[y][x] = self.imageData.data[y][x]
        self.imageData.data = newData.astype(int)
        if align:
            self.align()
        self.imageData.data
        return self.imageData
  
    def addMargins(self,xMargin = 0, yMargin = 0):
        self.crop(self.imageData.data.shape[1] + 2 * xMargin, self.imageData.data.shape[0] + 2 * yMargin)
        self.shift(xMargin, yMargin)
        return self.imageData
    
    def wrap(self,xMargin = 0, yMargin = 0):
        xMin, xMax, yMin, yMax = self.limits()
        height = yMax - yMin + 1
        width = xMax - xMin + 1
        self.shift(0-xMin, 0-yMin)
        self.crop(width + 2 * xMargin, height + 2 * yMargin)
        self.shift(xMargin, yMargin)
        return self.imageData

    def isCorner(self,x,y,position):
        positionC =  [int((position[0]-position[1])/2),int((position[0]+position[1])/2)]
        positionCC = [int((position[1]+position[0])/2),int((position[1]-position[0])/2)]
        xNextC = x+positionC[1] 
        yNextC = y+positionC[0]
        xNextCC = x+positionCC[1]
        yNextCC = y+positionCC[0]

        if  (abs(position[0])==1 and abs(position[1])==1) and (0 <= xNextC < self.imageData.data.shape[1] and 0 <= yNextC < self.imageData.data.shape[0] and xNextCC < self.imageData.data.shape[1] and 0 <= yNextCC < self.imageData.data.shape[0]):
            if self.imageData.data[y][x] and (self.imageData.data[yNextC][xNextC] or self.imageData.data[yNextCC][xNextCC]):
                return True
        return False


    def scale(self,xFields,yFields,scaleStroke=False):
        scaledData = np.zeros((yFields,xFields))
        if(scaleStroke):
            shapeX = self.imageData.data.shape[1]
            xFieldsAugmented = math.ceil((xFields - 1)*shapeX/(shapeX-1)) if shapeX != 1 else 0
            shapeY = self.imageData.data.shape[0]
            yFieldsAugmented = math.ceil((yFields - 1)*shapeY/(shapeY-1)) if shapeY != 1 else 0

            scalingX = xFieldsAugmented / self.imageData.data.shape[1]
            scalingY = yFieldsAugmented / self.imageData.data.shape[0]
            for y in range(0,self.imageData.data.shape[0]):
                for x in range(0,self.imageData.data.shape[1]):
                    if(self.imageData.data[y][x]):
                        xScaled = math.floor(x * scalingX)
                        yScaled = math.floor(y * scalingY)
                        scaledData[yScaled][xScaled] = self.imageData.data[y][x]
                        positions = [[-1,1],[0,1],[1,1],[1,0],[-1,-1]]
                        for position in positions:
                            xNext = x+position[1] if 0 <= x+position[1] < self.imageData.data.shape[1] else x
                            yNext = y+position[0] if 0 <= y+position[0] < self.imageData.data.shape[0] else y
                            if(self.imageData.data[yNext][xNext] and (not self.isCorner(x,y,position))):
                                xScaledNext = math.floor(xNext * scalingX)
                                yScaledNext = math.floor(yNext * scalingY)
                                tMax = max(abs(xScaledNext-xScaled),abs(yScaledNext-yScaled))
                                for t in range(1,tMax):
                                    xP = math.floor(xScaled+(t/tMax)*(xScaledNext-xScaled))
                                    yP = math.floor(yScaled+(t/tMax)*(yScaledNext-yScaled))
                                    scaledData[yP][xP] = self.imageData.data[y][x]
        else:
            scalingX = xFields / self.imageData.data.shape[1]
            scalingY = yFields / self.imageData.data.shape[0]
            for y in range(0,self.imageData.data.shape[0]):
                for x in range(0,self.imageData.data.shape[1]):
                    xScaled = math.floor(x * scalingX)
                    xScaledNext = math.floor((x + 1) * scalingX)
                    yScaled = math.floor(y * scalingY)
                    yScaledNext = math.floor((y + 1) * scalingY)
                    xScaledNext = xScaled+1 if xScaledNext == xScaled else xScaledNext
                    yScaledNext = yScaled+1 if yScaledNext == yScaled else yScaledNext
                    if(self.imageData.data[y][x]):
                        for yP in range(yScaled,yScaledNext):
                            for xP in range(xScaled,xScaledNext):
                                scaledData[yP][xP] = self.imageData.data[y][x]

        self.imageData.data = scaledData
        return self.imageData

    def fit(self,xFields, yFields, xMargin = 0, yMargin = 0, keepRatio = False,scaleStroke=False): 
        xFieldsNetto = xFields - 2*xMargin
        yFieldsNetto = yFields - 2*yMargin
        [xMin, xMax, yMin, yMax] = self.limits()
        height = yMax - yMin + 1
        width = xMax - xMin + 1
        scaleRatio = height / width
        if not keepRatio:
            xFieldsScaled = xFieldsNetto
            yFieldsScaled = yFieldsNetto
        elif scaleRatio > yFieldsNetto / xFieldsNetto: 
            xFieldsScaled = 1 if math.floor(yFieldsNetto/scaleRatio) == 0 else math.floor(yFieldsNetto/scaleRatio)
            yFieldsScaled = yFieldsNetto
        else:
            xFieldsScaled = xFieldsNetto
            yFieldsScaled = 1 if math.floor(xFieldsNetto*scaleRatio)==0 else math.floor(xFieldsNetto * scaleRatio)
        self.wrap()
        self.scale(xFieldsScaled,yFieldsScaled,scaleStroke)
        self.crop(xFieldsNetto,yFieldsNetto,True)
        self.addMargins(xMargin, yMargin)
        return self.imageData

    def _round(self,x,n):
        return int(x*10**n + 0.5 * math.copysign(1,x))*10**(-n)
        
    def rotate(self, angle, rotateStroke = False,fit = False,roundPoints=True):
        angle = angle*2*math.pi/360
        shapeX = self.imageData.data.shape[1]
        shapeY = self.imageData.data.shape[0]
        cosA = self._round(math.cos(angle),10)
        sinA = self._round(math.sin(angle),10)
        xRotatedList = []
        yRotatedList = []
        for y in range(0,shapeY):
            for x in range(0,shapeX):
                if roundPoints:                    
                    xRotated = round(x*cosA - y*sinA)
                    yRotated = round(x*sinA + y*cosA) 
                else:
                    xRotated = math.ceil(x*cosA - y*sinA) if x*cosA - y*sinA > 0 else math.floor(x*cosA - y*sinA)
                    yRotated = math.ceil(x*sinA + y*cosA) if x*sinA + y*cosA > 0 else math.floor(x*sinA + y*cosA) 
                xRotatedList.append(xRotated)
                yRotatedList.append(yRotated)
        rotatedData = np.zeros((max(yRotatedList)-min(yRotatedList)+1,max(xRotatedList)-min(xRotatedList)+1))
        for y in range(0,shapeY):
            for x in range(0,shapeX):
                if(not rotateStroke or self.imageData.data[y][x]):
                    if roundPoints:
                        xRotated = round(x*cosA - y*sinA) -min(xRotatedList)
                        yRotated = round(x*sinA + y*cosA) -min(yRotatedList)
                    else:
                        xRotated = math.ceil(x*cosA - y*sinA) -min(xRotatedList) if x*cosA - y*sinA > 0 else math.floor(x*cosA - y*sinA) -min(xRotatedList)
                        yRotated = math.ceil(x*sinA + y*cosA) -min(yRotatedList) if x*sinA + y*cosA > 0 else math.floor(x*sinA + y*cosA) -min(yRotatedList)
                    rotatedData[yRotated][xRotated] = self.imageData.data[y][x]
        self.imageData.data = rotatedData
        if fit:
            self.fit(xFields = shapeX,yFields = shapeY,xMargin = 0, yMargin = 0,keepRatio=True,scaleStroke = rotateStroke)
        return self.imageData

    def filledPixels(self,array2D):
        counter=0
        for y in range(0,array2D.shape[0]):
            for x in range(0,array2D.shape[1]):
                counter = counter+1 if array2D[y][x] else counter
        print("Filled pixels = " + str(counter))
        return counter
