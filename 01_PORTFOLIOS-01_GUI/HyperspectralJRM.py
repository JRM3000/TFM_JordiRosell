#pip install opencv-python

import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

class HypSpImage:
    ''' 
        __init__(self,inputCUB,resolution,slices,cameraBandsFile)
        info(self)
        getWidth(self)
        getHeight(self)
        getNumberBands(self)        
        getBWimage(self,band=-1.0,idx=-1)
        getRGBimage(self)
        getCUBimage(self)
        saveBWimage(self, outFilePath, band=-1.0, idx=-1)
        saveRGBimage(self, outFilePath)
        getPixelBands(self, x, y)
        getIntervalBands
        getPixelBand(self, x, y, band=-1.0,idx=-1)
    ''' 
    
    def __init__(self,inputCUB,resolution,slices,cameraBandsFile):
        """
        HypSpImage class constructor. Inspect the file and create the CUB structure.         
        ------------------------------------------------------------------------------------------
        INPUT:
          - inputCUB[string]: Path of the RAW file containing the CUB.
          - resolution[int]: X width of the image.
          - slices[int]: Y height of the image.
          - cameraBandsFile[string]: Path of the file where the wavelengths returned by the camera are specified
        """
        self.dicc_sourceCode= {"NULL":-1, "color":0, "hsc":1, "ir":2}
        self.dicc_sourceDesc = {v: k for k, v in self.dicc_sourceCode.items()}
        
        self.initialized = True
        
        # Split the path of the input file
        self.fullPathFile = inputCUB
        self.pathFile, self.nameFile  = os.path.split(inputCUB)
        self.name, self.extension = self.nameFile.split(os.extsep, 1)
        '''
        Example: "C:\GARBAGE\COEN\ir_A_00027(FX17).tiff"
        ----------------------------------------------------------
        self.fullPathFile:  C:\GARBAGE\COEN\ir_A_00027(FX17).tiff
        self.pathFile:  C:\GARBAGE\COEN
        self.nameFile:  ir_A_00027(FX17).tiff
        self.name:  ir_A_00027(FX17)
        self.extension:  tiff
        '''
        
        # Inspect type of image depending to file name:
        # color -> 'color_A_00562.tiff'
        # hsc ---> 'hsc_A_00562.tiff'
        # ir  ---> 'ir_A_00562.tiff'
        if (self.name[:2]=='co'): #"color"
            self.source = self.dicc_sourceCode['color']
        elif (self.name[:2]=='hs'): #"hsc"
            self.source = self.dicc_sourceCode['hsc']
        elif (self.name[:2]=='ir'): #"ir"
            self.source = self.dicc_sourceCode['ir']
        else:
            self.source = self.dicc_sourceCode['NULL']
            self.initialized = False
            #exit -1
        
        # Read Resolution and slices fields
        self.resolution = resolution
        self.slices = slices

        # Create data CUB
        # If the 'slice' and 'resolution' relationship is incorrect it causes an exception
        try:
            self.orgImg = cv2.imread(inputCUB,cv2.IMREAD_ANYDEPTH)
            self.orgImg = self.orgImg.reshape((slices,resolution,-1), order='F')
        except:
            #exit -1
            print("ERROR: Dimensional image conversion error")
            self.initialized = False
            pass
        
        # Calculates the number of bands inspecting the structure of the CUB
        if self.orgImg.ndim == 3:
            self.nBands = self.orgImg.shape[2]  
        else:
            self.initialized = False
            #exit -1
            pass
        
        # Read bands from the Camera bands file.
        self.bands = np.array([])
        with open(cameraBandsFile, "r") as file1:
            for line in file1.readlines():
                #values = (str.split(line))
                values = (line.split())
                self.bands = np.append(self.bands, float(values[0]))
                
        # Check 
        if self.nBands != len(self.bands):
            print(f"ERROR: the number of bands in the image does not match those in the camera file {cameraBandsFile}")
            self.initialized = False
            #exit -1
            pass
        
        # Equalizes and write the image (sum of all bands)
        # img_sum = np.sum(self.orgImg,axis=2,dtype=float)
        # img_final = ((img_sum/np.max(img_sum))*255.0).astype(int)
        # outpath = os.path.join(self.pathFile, self.name+"_reshape.tiff")
        # cv2.imwrite(outpath, img_final)        
        
    def info(self):
        """
        info() --> Show the image info
        ------------------------------------------------------------------------------------------
        INPUT:
        OUTPUT:
        CONSIDERATIONS:
        """         
        print("Image inspected: ",self.nameFile)
        if self.initialized:
            print(f"Type of image: {self.dicc_sourceDesc[self.source]}")            
            print(f"Size image: ({self.resolution}x{self.slices})")
            if (self.source == self.dicc_sourceCode['hsc']) or (self.source == self.dicc_sourceCode['ir']):
                print(f"Number of bands: {self.nBands}")            
                print(f"Bands: from {self.bands[0]}  to {self.bands[self.nBands-1]}")                        
        else:
            print("This image was not initialized correctly")
            
    def getWidth(self):
        """
        getWidth() --> Return Width of the image
        ------------------------------------------------------------------------------------------
        INPUT:
        OUTPUT:
        CONSIDERATIONS:
        """         
        return int(self.resolution)
        
    def getHeight(self):
        """
        getHeight() --> Return Height of the image
        ------------------------------------------------------------------------------------------
        INPUT:
        OUTPUT:
        CONSIDERATIONS:
        """         
        return int(self.slices)
    
    def getNumberBands(self):
        """
        getNumberBands() --> Return number of bands of the image
        ------------------------------------------------------------------------------------------
        INPUT:
        OUTPUT:
        CONSIDERATIONS:
        """         
        return self.nBands
        
    def getBWimage(self,band=-1.0,idx=-1):
        """
        getBWimage() --> Returns the image corresponding to the selected band. The band index can be specified 
                         or the band directly. In this case it would not exactly match the captured bands, it 
                         would return the proportional linear extrapolation of the previous and subsequent images 
                         of the selected band.
        ------------------------------------------------------------------------------------------
        INPUT:
          - band[float]: returns the image corresponding to the selected index.
          - idx[int]: returns the image corresponding to the selected band.
        OUTPUT:
          - retuned[BW image -numpy struc.-]:  Returned image
        CONSIDERATIONS:
          * If HypSpImage is not an 'hsc' or 'ir' image it returns a black image
          * If HypSpImage does not contain the selected band it returns a black image
          * If both are selected it will only consider 'idx'
        """       
           
        blackImg = np.zeros((self.orgImg[:,:,0].shape))
        
        if (band==-1.0) and (idx==-1):
            print(f"ERROR: An index or band must be selected ")
            return blackImg
        else:
            if self.initialized and (self.source == self.dicc_sourceCode['hsc'] or self.source == self.dicc_sourceCode['ir']):
                if idx >= 0:
                    if idx < self.nBands:
                        return np.copy(self.orgImg[:,:,idx])
                    else:
                        return blackImg

                if band > 0:
                    if (band >=self.bands[0] and  band <=self.bands[-1]):
                        idx=self.bands.searchsorted(band)
                        if idx==0:
                            return np.copy(self.orgImg[:,:,idx])
                        else:
                            diffBands= self.bands[idx]-self.bands[idx-1]
                            diffFind = self.bands[idx] - band
                            pond = diffFind / diffBands

                            imgPond = self.orgImg[:,:,idx-1]*pond + self.orgImg[:,:,idx]*(1-pond)
                            return imgPond
                    else:
                        return blackImg
            else:
                return blackImg

        
    def getRGBimage(self):
        """
        getRGBimage() --> Returns the color image
        ------------------------------------------------------------------------------------------
        INPUT:
        OUTPUT:
          - retuned[RGB image -numpy struc.-]:  Returned image
        CONSIDERATIONS:
          * If HypSpImage is not an 'color' or 'hsc' image it returns a black image
        """ 
        if self.source == self.dicc_sourceCode['color']:
            img_RGB = np.copy(self.orgImg)            
        elif self.source == self.dicc_sourceCode['hsc']:
            img_RGB = np.zeros((self.orgImg[:,:,0].shape+ (3,)))
            img_RGB[:,:,0] = self.getBWimage(band=470) # B
            img_RGB[:,:,1] = self.getBWimage(band=525) # G
            img_RGB[:,:,2] = self.getBWimage(band=700)  # R
            img_RGB = np.int32(cv2.cvtColor(np.float32(img_RGB), cv2.COLOR_BGR2RGB))
        else:
            img_RGB = np.zeros((self.orgImg[:,:,0].shape+ (3,)))
            
        img_RGB = ((img_RGB/np.max(img_RGB))*255.0).astype(int)

        return img_RGB
            
    def getCUBimage(self):
        """
        getRGBimage() --> Returns the color image
        ------------------------------------------------------------------------------------------
        INPUT:
        OUTPUT:
          - retuned[CUB image -numpy struc.-]:  Returned CUB image
        CONSIDERATIONS:
        """ 
        return np.copy(self.orgImg)
    
        
    def saveBWimage(self, outFilePath, band=-1.0, idx=-1):
        """
        saveBWimage() -->Save the color image
        ------------------------------------------------------------------------------------------
        INPUT:
          - outFilePath[string]: Name o file to save.
          - band[float]: returns the image corresponding to the selected index.
          - idx[int]: returns the image corresponding to the selected band.
        OUTPUT:
        CONSIDERATIONS:
          * If HypSpImage is not an 'hsc' or 'ir' image it returns a black image
          * If HypSpImage does not contain the selected band it returns a black image
          * If both are selected it will only consider 'idx'
        """ 
        fileOut = os.path.join(outFilePath, self.name+"_BWreshape.tiff")
        cv2.imwrite(fileOut, self.getBWimage(band=band,idx=idx))
            
    
    
    def saveRGBimage(self, outFilePath):
        """
        saveRGBimage() --> Save the color image
        ------------------------------------------------------------------------------------------
        INPUT:
          - outFilePath[string]: Name o file to save.
        OUTPUT:
        CONSIDERATIONS:
          * If HypSpImage is not an 'color' or 'hsc' image it returns a black image
        """ 
        fileOut = os.path.join(outFilePath+'\\'+ self.name+"_RGB.tiff")
        img_RGB = self.getRGBimage()
        b,g,r = cv2.split(img_RGB)
        img_RGB[:,:,0] = r
        img_RGB[:,:,2] = b
        cv2.imwrite(fileOut, img_RGB)
        
        

    def getPixelBands(self, x, y):
        """
        getPixelBands() --> Get pixel value of image (position X,Y)
        ------------------------------------------------------------------------------------------
        INPUT:
          - x[int]: X position.
          - y[int]: Y position.
        OUTPUT:
          - return[vector]: Vector that contains all bands of a pixel
        CONSIDERATIONS:
          * If HypSpImage is not an 'color' or 'hsc' image it returns a black image
        """ 
        if (x < self.resolution) and (y < self.slices):
            strOut = np.copy(self.orgImg[y,x,:])
        else: 
            strOut = np.array([])
    
        return strOut
    
    def getPixelBand(self, x, y, band=-1.0,idx=-1):
        """
        getPixelBand() --> Get pixel value of image (position X,Y)
        ------------------------------------------------------------------------------------------
        INPUT:
          - x[int]: X position.
          - y[int]: Y position.
          - band[float]: returns the image corresponding to the selected index.
          - idx[int]: returns the image corresponding to the selected band.
        OUTPUT:
          - return[int]: Value of a pixel corresponding to specific band
        CONSIDERATIONS:
          * If HypSpImage is not an 'color' or 'hsc' image it returns a black image
        """ 
        valuePixel = -1
        if (x < self.resolution) and (y < self.slices):
            if idx >= 0:
                if idx < self.nBands:
                    valuePixel = self.orgImg[y,x,idx]
            elif band > 0:
                if (band >=self.bands[0] and  band <=self.bands[-1]):
                    idx=self.bands.searchsorted(band)
                    if idx==0:
                        valuePixel = self.orgImg[y,x,idx]
                    else:
                        diffBands= self.bands[idx]-self.bands[idx-1]
                        diffFind = self.bands[idx] - band
                        pond = diffFind / diffBands
                        valuePixel = self.orgImg[y,x,idx-1]*pond + self.orgImg[y,x,idx]*(1-pond)
        return valuePixel
    
    def getAreaBands(self, xIni,yIni, xEnd, yEnd, typeOutInfo):        
        pass 

    def getAreaBand(self, xIni,yIni, xEnd, yEnd, typeOutInfo, band=-1.0,idx=-1):
        pass 
   