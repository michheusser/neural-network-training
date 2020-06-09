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
            # xFieldsNetto = xFields / (1 + 1/self.imageData.data.shape[1])
            # strokeMarginX = math.floor(xFieldsNetto / self.imageData.data.shape[1])
            
            # xFields + strokeMargin = xFieldsAugmented
            # xFieldsAugmented = xFields*(1 + 1/self.image.data.shape[1])
            # strokeMarginX = xFieldsAugmented / self.imageData.data.shape[1] # -1 potentially
            # xFields = xFieldsAugmented - strokeMarginX 
            # xFields = xFieldsAugmented - xFieldsAugmented/shape + 1
            # strokeMarginX = xFieldsAugmented / self.imageData.data.shape[1] - 1
            # xFieldsAugmented = (xFields-1) / (1 - 1 / self.imageData.data.shape[1]) 

            #IMPORTANT: 
            # 
            # xFields + strokeMarginX = xFieldsAugmented*
            # map: (x) => x * xFieldsAugmented / shapeX      
            # xLastShape = shapeX -1
            # xLastAugmented = (shapeX - 1) * xFieldsAugmented / shapeX
            # xLastAugmented = (xFields - 1)
            # => (xFields - 1) = (shapeX - 1) * xFieldsAugmented / shapeX
            # => xFieldsAugmented = (xFields - 1)*shapeX/(shapeX-1)
            # => strokeMarginX = xFieldsAugmented - xFields
            # strokeMarginX = (xFields - 1)*shapeX/(shapeX-1) - xFields
            # ----
            # strokeMarginX = xFields*(shapeX/(shapeX-1)-1)-shapeX/(shapeX-1)
            # xFieldsAugmented = (xFields - 1)*shapeX/(shapeX-1)
            # => xFieldsAugmented = math.ceil(xFields-1)*shapeX/(shapeX-1) worked with math.ceil since there were some
            # jumps when iterating through different scaling ratios

            shapeX = self.imageData.data.shape[1]
            #strokeMarginX = math.ceil(xFields*(shapeX/(shapeX-1)-1)-shapeX/(shapeX-1))
            xFieldsAugmented = math.ceil((xFields - 1)*shapeX/(shapeX-1)) if shapeX != 1 else 0
            shapeY = self.imageData.data.shape[0]
            #strokeMarginY = math.ceil(yFields*(shapeY/(shapeY-1)-1)-shapeY/(shapeY-1))
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
                                #xScaledNext = xScaled+1 if xScaledNext == xScaled else xScaledNext
                                #yScaledNext = yScaled+1 if yScaledNext == yScaled else yScaledNext
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
        #print("self.imageData.data.shape[1] = "+str(self.imageData.data.shape[1]) + ", self.imageData.data.shape[0] = "+str(self.imageData.data.shape[0]))
        return self.imageData

    def rotate(self, angle, rotateStroke = False):
        angleDeg = angle
        angle = angle*2*math.pi/360
        rotatedData = np.zeros(self.imageData.data.shape)
        shapeX = self.imageData.data.shape[1]
        shapeY = self.imageData.data.shape[0]
        # IMPORTANT:
        # 2*xMax = shapeX >= D*abs(cos(angle+angle0)) = D*abs(cos(angle)*cos(angle0)-sin(angle)*sin(angle0)) = D*abs(cos(angle)*shapeX/sqrt(shapeX**2 + shapeY**2)-sin(angle)*shapeY/sqrt(shapeX**2 + shapeY**2))
        # 2*yMax = shapeY >= D*abs(sin(angle+angle0)) = D*abs(sin(angle)*cos(angle0)+cos(angle)*sin(angle0)) = D*abs(sin(angle)*shapeX/sqrt(shapeX**2 + shapeY**2)+cos(angle)*shapeY/sqrt(shapeX**2 + shapeY**2))
        # ==> R <= shapeX/(2*abs(cos(angle)*shapeX/sqrt(shapeX**2 + shapeY**2)-sin(angle)*shapeY/sqrt(shapeX**2 + shapeY**2)))
        # ==> R <= shapeY/(2*abs(sin(angle)*shapeX/sqrt(shapeX**2 + shapeY**2)+cos(angle)*shapeY/sqrt(shapeX**2 + shapeY**2)))
        # D <= shapeX / abs(cosA*cosA0-sinA*sinA0)
        # D <= shapeY / abs(sinA*cosA0+sinA0*cosA)
        cosA0 = shapeX/(math.sqrt(shapeX**2+shapeY**2))
        sinA0 = shapeY/(math.sqrt(shapeX**2+shapeY**2))
        cosA = round(math.cos(angle),5)
        sinA = round(math.sin(angle),5)

        D = min(shapeX/abs(cosA*cosA0-sinA*sinA0),shapeY/abs(sinA*cosA0+sinA0*cosA))

        #if (cosA*shapeX-cosA*shapeY==0):
        #    R = shapeY/(2*abs(sinA*shapeX/math.sqrt(shapeX**2 + shapeY**2)+cosA*shapeY/math.sqrt(shapeX**2 + shapeY**2)))
        #elif(sinA*shapeX+cosA*shapeY==0):
        #    R = shapeX/(2*abs(cosA*shapeX/math.sqrt(shapeX**2 + shapeY**2)-sinA*shapeY/math.sqrt(shapeX**2 + shapeY**2)))
        #else: 
        #    R = min(shapeX/(2*abs(cosA*shapeX/math.sqrt(shapeX**2 + shapeY**2)-sinA*shapeY/math.sqrt(shapeX**2 + shapeY**2))),shapeY/(2*abs(sinA*shapeX/math.sqrt(shapeX**2 + shapeY**2)+cosA*shapeY/math.sqrt(shapeX**2 + shapeY**2))))

        #R = min(shapeX/(2*abs(cosA*shapeX/math.sqrt(shapeX**2 + shapeY**2)-sinA*shapeY/math.sqrt(shapeX**2 + shapeY**2))),shapeY/(2*abs(sinA*shapeX/math.sqrt(shapeX**2 + shapeY**2)+cosA*shapeY/math.sqrt(shapeX**2 + shapeY**2))))

        print("*********")
        print("shapeX = " + str(shapeX) + ", shapeY = " + str(shapeY) + ", angleDeg = "+ str(angleDeg) + ", D = " + str(D))
        print("*********")
    
        rotatedShapeX = math.floor(D*cosA0)
        rotatedShapeY = math.floor(D*sinA0)

        
        self.scale(rotatedShapeX,rotatedShapeY,scaleStroke=rotateStroke)
        print("rotatedShapeX = " + str(rotatedShapeX) + ", rotatedShapeY = " + str(rotatedShapeY) + ", angleDeg = "+ str(angleDeg) + ", D = " + str(D))
        print("*********")
        self.imageData.display()


        #xCentre = math.floor((shapeX-1)/2)
        #yCentre = math.floor((shapeY-1)/2)
        #xCentreRotated = math.floor((rotatedShapeX-1)/2)
        #yCentreRotated = math.floor((rotatedShapeY-1)/2)
        xCentre = (shapeX-1)/2
        yCentre = (shapeY-1)/2
        xCentreRotated = (D*cosA0-1)/2
        yCentreRotated = (D*sinA0-1)/2

        for y in range(0,self.imageData.data.shape[0]):
                for x in range(0,self.imageData.data.shape[1]):
                    if(self.imageData.data[y][x]):
                        # xRotated = math.floor((x-xCentreRotated)*cosA - (y-yCentreRotated)*sinA) + xCentre
                        # yRotated = math.floor((x-xCentreRotated)*sinA + (y-yCentreRotated)*cosA) + yCentre
                        #xRotated = math.floor((x-xCentreRotated)*cosA - (y-yCentreRotated)*sinA + xCentre)
                        #yRotated = math.floor((x-xCentreRotated)*sinA + (y-yCentreRotated)*cosA + yCentre)
                        
                        xRotated = (x-xCentreRotated)*cosA - (y-yCentreRotated)*sinA + xCentre
                        yRotated = (x-xCentreRotated)*sinA + (y-yCentreRotated)*cosA + yCentre
                        #why not using any math.floor is this not working?
                        if(xRotated-xCentre > 0):
                            xRotated = math.floor(xRotated) + math.floor(xCentre)
                        else:
                            xRotated = math.ceil(xRotated) + math.floor(xCentre)
                        if(yRotated-yCentre > 0):
                            yRotated = math.floor(yRotated) + math.floor(yCentre)
                        else:
                            yRotated = math.ceil(yRotated) + math.floor(yCentre)
                        

                        rotatedData[yRotated][xRotated] = self.imageData.data[y][x]
                        if yRotated < 0 or xRotated< 0:
                            print("found")
        self.imageData.data = rotatedData
        return self.imageData

    def rotate2(self, angle, rotateStroke = False):
        angleDeg = angle
        angle = angle*2*math.pi/360
        rotatedData = np.zeros(self.imageData.data.shape)
        shapeX = self.imageData.data.shape[1]
        shapeY = self.imageData.data.shape[0]
        
        cosA0 = shapeX/(math.sqrt(shapeX**2+shapeY**2))
        sinA0 = shapeY/(math.sqrt(shapeX**2+shapeY**2))
        cosA1 = -cosA0
        sinA1 = sinA0
        cosA = round(math.cos(angle),5)
        sinA = round(math.sin(angle),5)
    
        D = min(shapeX/abs(cosA*cosA0-sinA*sinA0),shapeY/abs(sinA*cosA0+sinA0*cosA), shapeX/abs(cosA*cosA1-sinA*sinA1),shapeY/abs(sinA*cosA1+sinA1*cosA))
        rotatedShapeX = math.floor(D*cosA0)
        rotatedShapeY = math.floor(D*sinA0)
        self.scale(rotatedShapeX,rotatedShapeY,scaleStroke=rotateStroke)
        self.imageData.display()

        xCentre = (shapeX-1)/2
        yCentre = (shapeY-1)/2
        xCentreRotated = (rotatedShapeX-1)/2
        yCentreRotated = (rotatedShapeY-1)/2

        print("*********")
        print("angleDeg = "+ str(angleDeg) + ", D = " + str(D))
        print("shapeX = " + str(shapeX) + ", shapeY = " + str(shapeY))
        print("rotatedShapeX = " + str(rotatedShapeX) + ", rotatedShapeY = " + str(rotatedShapeY))
        print("*********")

        for y in range(0,self.imageData.data.shape[0]):
            for x in range(0,self.imageData.data.shape[1]):
                if(self.imageData.data[y][x]):
                    xRotated = math.floor((x-xCentreRotated)*cosA - (y-yCentreRotated)*sinA + xCentre)
                    yRotated = math.floor((x-xCentreRotated)*sinA + (y-yCentreRotated)*cosA + yCentre)

                    rotatedData[yRotated][xRotated] = self.imageData.data[y][x]
                    if yRotated < 0 or xRotated< 0:
                        print("found")
        self.imageData.data = rotatedData
        return self.imageData

    def rotate3(self, angle, rotateStroke = False):
        angleDeg = angle
        angle = angle*2*math.pi/360
        
        shapeX = self.imageData.data.shape[1]
        shapeY = self.imageData.data.shape[0]
        
        cosA0 = (shapeX-1)/(math.sqrt((shapeX-1)**2+(shapeY-1)**2))
        angle0deg = math.acos(cosA0)*360/(2*math.pi)
        sinA0 = (shapeY-1)/(math.sqrt((shapeX-1)**2+(shapeY-1)**2))
        cosA1 = -cosA0
        angle1deg = math.acos(cosA1)*360/(2*math.pi)
        sinA1 = sinA0
        cosA = round(math.cos(angle),5)
        sinA = round(math.sin(angle),5)
        D = math.sqrt((shapeX-1)**2+(shapeY-1)**2)
    
        rotatedMaxX = max(D*abs(cosA*cosA0-sinA*sinA0),D*abs(cosA*cosA1-sinA*sinA1))
        rotatedMaxY = max(D*abs(sinA*cosA0+sinA0*cosA),D*abs(sinA*cosA1+sinA1*cosA))

        rotatedCornerX = D*(cosA*cosA0-sinA*sinA0)
        rotatedCornerY = D*(sinA*cosA0+sinA0*cosA)

        rotatedShapeX = math.floor(maxX)
        rotatedShapeY = math.floor(maxY)
        rotatedData = np.zeros((rotatedShapeY,rotatedShapeX))

        xCentre = (shapeX-1)/2
        yCentre = (shapeY-1)/2
        xCentreRotated = (maxX-1)/2
        yCentreRotated = (maxY-1)/2

        for y in range(0,shapeY):
            for x in range(0,shapeX):
                if(self.imageData.data[y][x]):
                    
                    xRotated = math.floor((x-xCentre)*cosA - (y-yCentre)*sinA + xCentreRotated)
                    yRotated = math.floor((x-xCentre)*sinA + (y-yCentre)*cosA + yCentreRotated)
                    rotatedData[yRotated][xRotated] = self.imageData.data[y][x]
        self.imageData.data = rotatedData
        return self.imageData

    def rotate4(self, angle, rotateStroke = False):
        angleDeg = angle
        angle = angle*2*math.pi/360
        
        shapeX = self.imageData.data.shape[1]
        shapeY = self.imageData.data.shape[0]
        
        cosA0 = (shapeX-1)/(math.sqrt((shapeX-1)**2+(shapeY-1)**2))
        sinA0 = (shapeY-1)/(math.sqrt((shapeX-1)**2+(shapeY-1)**2))
        cosA = round(math.cos(angle),5)
        sinA = round(math.sin(angle),5)
        #angle0deg = math.acos(cosA0)*360/(2*math.pi)
        #D = math.sqrt((shapeX-1)**2+(shapeY-1)**2)

        #rotatedCornerX = math.ceil(D*(cosA*cosA0-sinA*sinA0)) if D*(cosA*cosA0-sinA*sinA0) > 0 else math.floor(D*(cosA*cosA0-sinA*sinA0))
        #rotatedCornerY = math.ceil(D*(sinA*cosA0+sinA0*cosA)) if D*(sinA*cosA0+sinA0*cosA) > 0 else math.floor(D*(sinA*cosA0+sinA0*cosA))

        xRotatedList = []
        yRotatedList = []
        for y in range(0,shapeY):
            for x in range(0,shapeX):
                    xRotated = math.ceil(x*cosA - y*sinA) if x*cosA - y*sinA > 0 else math.floor(x*cosA - y*sinA)
                    yRotated = math.ceil(x*sinA + y*cosA) if x*sinA + y*cosA > 0 else math.floor(x*sinA + y*cosA) 
                    xRotatedList.append(xRotated)
                    yRotatedList.append(yRotated)
        xRotatedMin = min(xRotatedList)
        yRotatedMin = min(yRotatedList)
        xRotatedMax = max(xRotatedList)
        yRotatedMax = max(yRotatedList)
        rotatedData = np.zeros((yRotatedMax-yRotatedMin+1,xRotatedMax-xRotatedMin+1))
        
        for y in range(0,shapeY):
            for x in range(0,shapeX):
                #if(self.imageData.data[y][x]):
                    xRotated = math.ceil(x*cosA - y*sinA) if x*cosA - y*sinA > 0 else math.floor(x*cosA - y*sinA)
                    yRotated = math.ceil(x*sinA + y*cosA) if x*sinA + y*cosA > 0 else math.floor(x*sinA + y*cosA) 
                    xRotated -= xRotatedMin
                    yRotated -= yRotatedMin
                    rotatedData[yRotated][xRotated] = self.imageData.data[y][x]
        self.imageData.data = rotatedData
        return self.imageData