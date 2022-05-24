import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import cv2

def extractDescriptorSignal(fileData):

    
    data= pd.read_csv(fileData, sep='\t', header=None)
    # columna 0 --> Nº imatge
    # columna 1 --> Nº blob
    # columna 2 --> Coordinada Y
    # columna 3 --> Coordinada X
    # columna 4:228 --> valor del píxel a conjunt de longituds d'ona (224 bandes)
    
    '''
    # 1.- Número de canvis de signe (+/- i -/+) -imatge normalitzada-
    #        x_IR_01_canviSignePos
    #        x_IR_01_canviSigneNeg
    # 2.- Número de pixels ascendents i en descendents -imatge normalitzada-
    #        x_IR_02_Ascendent
    #        x_IR_02_Descendent
    #        x_IR_02_RelAscentdent  <--- sols fem servir aquest
    # 3.- número valors positius vs total  -imatge normalitzada-
    #        x_IR_03_relacioPos
    # 4.- Posició màxims i mínims 
    #        x_IR_04_Max
    #        x_IR_04_MaxPos
    #        x_IR_04_Min
    #        x_IR_04_MinPos
    #        x_IR_04_RelMaxMin  <--- Relació màxim/minin
    #        x_IR_04_Max_AllG
    #        x_IR_04_MaxPos_AllG
    #        x_IR_04_Min_AllG
    #        x_IR_04_MinPos_AllG
    # 5.- Posició màxims i mínims 1a derivada
    #        x_IR_05_Max_1aDer
    #        x_IR_05_MaxPos_1aDer
    #        x_IR_05_Min_1aDer
    #        x_IR_05_MinPos_1aDer
    # 6.- Desviació estardard respecte 0 -imatge normalitzada i 1a derivada-
    #        x_IR_06_StDev
    #        x_IR_06_StDev_1aDer
    '''
    D= data.to_numpy()
    g_IR = D[:,4:] 
    Npixels= data.shape[0]          # Npixels --> Número de píxels
    Nbandes= len(g_IR[0,4:])     # Nbandes -_> Número de bandes

    g_IR_norm = np.zeros(g_IR.shape)     # g_IR_norm --> Gràfica bandes IR normalitzada (per cada píxel)
    g_IR_deriv = np.zeros(g_IR.shape)    # g_IR_deriv --> Gràfica  1a derivada bandes IR normalitzada (per cada píxel)

    x_IR_01_canviSignePos = np.zeros(Npixels)
    x_IR_01_canviSigneNeg = np.zeros(Npixels)
    x_IR_02_Ascendent = np.zeros(Npixels)
    x_IR_02_Descendent = np.zeros(Npixels)
    x_IR_02_RelAscentdent = np.zeros(Npixels)
    x_IR_03_relacioPos = np.zeros(Npixels) #relació positius/ negatiu
    x_IR_04_Max  = np.zeros(Npixels)
    x_IR_04_MaxPos  = np.zeros(Npixels)
    x_IR_04_Min  = np.zeros(Npixels)
    x_IR_04_MinPos  = np.zeros(Npixels)
    x_IR_04_RelMaxMin  = np.zeros(Npixels)
    x_IR_04_Max_AllG  = np.zeros(Npixels)
    x_IR_04_MaxPos_AllG  = np.zeros(Npixels)
    x_IR_04_Min_AllG  = np.zeros(Npixels)
    x_IR_04_MinPos_AllG  = np.zeros(Npixels)
    x_IR_05_Max_1aDer  = np.zeros(Npixels)
    x_IR_05_MaxPos_1aDer  = np.zeros(Npixels)
    x_IR_05_Min_1aDer  = np.zeros(Npixels)
    x_IR_05_MinPos_1aDer  = np.zeros(Npixels)
    x_IR_06_StDev = np.zeros(Npixels)
    x_IR_06_StDev_1aDer = np.zeros(Npixels)   


    for n in range(Npixels):
    
    #== Normalització de la gràfica (centra la distribució a 0) ===============================================
        mean = sum(g_IR[n,:])/Nbandes
        g_IR_norm[n]= [(v/mean)-1 for v in g_IR[n]]

    #== calcula la primera derivada (centra la distribució a 0) ===============================================
        for idx_X in range(3,Nbandes-3):
            g_IR_deriv[n][idx_X]= g_IR_norm[n,idx_X-3]+g_IR_norm[n][idx_X-2]+g_IR_norm[n][idx_X-1]-g_IR_norm[n][idx_X+1]-g_IR_norm[n][idx_X+2]-g_IR_norm[n][idx_X+3]

    # 1.- Canvis de signe (+/- i -/+) ---------------------------------------------
        pos = 0
        neg = 0
        for idxBanda in range(1,Nbandes):
            if g_IR_norm[n][idxBanda-1]<0 and g_IR_norm[n][idxBanda]>=0:
                pos += 1
            elif g_IR_norm[n][idxBanda-1]>=0 and g_IR_norm[n][idxBanda]<0:
                neg +=1
        x_IR_01_canviSignePos[n] = pos
        x_IR_01_canviSigneNeg[n] = neg


    # 2.- Número de pixels ascendents i en descendents ---------------------------------
        asc = 0
        desc = 0
        for idxBanda in range(1,Nbandes):
            if g_IR_norm[n][idxBanda-1] < g_IR_norm[n][idxBanda]:
                asc += 1
            elif g_IR_norm[n][idxBanda-1] > g_IR_norm[n][idxBanda]:
                desc +=1
        x_IR_02_Ascendent[n] = asc
        x_IR_02_Descendent[n] = desc
        x_IR_02_RelAscentdent[n] = asc/Nbandes 

    # 3.- relació número valors positius vs Nbandes -----------------
        x_IR_03_relacioPos[n]= len(np.where(g_IR_norm[n,:] > 0)[0])/Nbandes


    # 4.- posició max  min  ------------------------------------------------------------
        maxPos = 0
        maxVal = -100000.0
        minPos = 0
        minVal = 100000.0
        maxPosAll = 0
        maxValAll = -100000.0
        minPosAll = 0
        minValAll = 100000.0    

        des = False
        asc = False
        
        for idx in range(1,Nbandes-1):
            if  g_IR_norm[n][idx-1] > g_IR_norm[n][idx]:
                asc = False
                des = True
            elif  g_IR_norm[n][idx-1] < g_IR_norm[n][idx]:
                asc = True
                des = False
                

            if des and (g_IR_norm[n][idx] < g_IR_norm[n][idx+1]):
                if(minVal>g_IR_norm[n][idx]):
                    minVal = g_IR_norm[n][idx]
                    minPos = idx

            elif asc and (g_IR_norm[n][idx] > g_IR_norm[n][idx+1]):
                if(maxVal<g_IR_norm[n][idx]):
                    maxVal = g_IR_norm[n][idx]
                    maxPos = idx

            if(maxValAll<g_IR_norm[n][idx]):
                maxValAll = g_IR_norm[n][idx]
                maxPosAll = idx
            if(minValAll>g_IR_norm[n][idx]):
                minValAll = g_IR_norm[n][idx]
                minPosAll = idx


        x_IR_04_Max[n]=maxVal
        x_IR_04_MaxPos[n]=maxPos
        x_IR_04_Min[n]=minVal
        x_IR_04_MinPos[n]=minPos
        x_IR_04_RelMaxMin[n] = abs(maxVal/minPos)

        x_IR_04_Max_AllG[n]=maxValAll
        x_IR_04_MaxPos_AllG[n]=maxPosAll
        x_IR_04_Min_AllG[n]=minValAll
        x_IR_04_MinPos_AllG[n]=minPosAll  

    # 5.- posicio max - 1a derivada -----------------------------------------------------------
        maxPos = 0
        maxVal = -100000.0
        minPos = 0
        minVal = 100000.0

        for idx in range(1,Nbandes-1):
            if  g_IR_deriv[n][idx-1] > g_IR_deriv[n][idx]:
                asc = False
                des = True
            if  g_IR_deriv[n][idx-1] < g_IR_deriv[n][idx]:
                asc = True
                des = False

            if des and (g_IR_deriv[n][idx] < g_IR_deriv[n][idx+1]):
                if(minVal>g_IR_deriv[n][idx]):
                    minVal = g_IR_deriv[n][idx]
                    minPos = idx

            elif asc and (g_IR_deriv[n][idx] > g_IR_deriv[n][idx+1]):
                if(maxVal<g_IR_deriv[n][idx]):
                    maxVal = g_IR_deriv[n][idx]
                    maxPos = idx


        x_IR_05_Max_1aDer[n]=maxVal
        x_IR_05_MaxPos_1aDer[n]=maxPos
        x_IR_05_Min_1aDer[n]=minVal
        x_IR_05_MinPos_1aDer[n]=minPos

    # 6.- Desviació estardard respecte 0 -imatge normalitzada i 1a derivada--------------------------
        acum = 0
        acum1aDer = 0

        for idx in range(Nbandes):
            acum +=g_IR_norm[n][idx]**2
            acum1aDer +=g_IR_deriv[n][idx]**2

        x_IR_06_StDev[n] = math.sqrt(acum/Nbandes)
        x_IR_06_StDev_1aDer[n] = math.sqrt(acum1aDer/Nbandes)


    #== Dels valors calculats n'elimina dels outliers (segons el percentil)
    # De moment no ho aplico, ja que pot set que dins un mateix conjunt, hi hagi element de diferents categories,
    # primer intentaré discriminar aplicant un algirsme de clustering, així podria identificar diferents materials
    # continguts en un mateix blob (per exemple quan es detecta una empolla amb una etiquera, en el fons està 
    # detectant dos materials amb un signatura diferent).
    # primer intentadé trobar desciptore univalors per tal d'aplicar un clustering que em permeti identificar factors
    # discriminants.
    # --> maxDvSt = ((np.percentile(x_IR_dvStd, 75)-np.percentile(x_IR_dvStd, 25))*1.5)+np.percentile(x_IR_dvStd, 75)
    # --> x_IR_norm = np.delete(x_IR_norm, np.where(x_IR_dvStd>maxDvSt), axis=0)


    #== normalitzo les variables al interval [1..-1]

    x_IR_01_canviSignePos_NOR = x_IR_01_canviSignePos/max(abs(x_IR_01_canviSignePos))
    x_IR_01_canviSigneNeg_NOR = x_IR_01_canviSigneNeg/max(abs(x_IR_01_canviSigneNeg))
    x_IR_02_Ascendent_NOR = x_IR_02_Ascendent/max(abs(x_IR_02_Ascendent))
    x_IR_02_Descendent_NOR = x_IR_02_Descendent/max(abs(x_IR_02_Descendent))
    x_IR_02_RelAscentdent_NOR = x_IR_02_RelAscentdent/max(abs(x_IR_02_RelAscentdent))
    x_IR_03_relacioPos_NOR = x_IR_03_relacioPos/max(abs(x_IR_03_relacioPos))
    x_IR_04_Max_NOR = x_IR_04_Max/max(abs(x_IR_04_Max))
    x_IR_04_MaxPos_NOR = x_IR_04_MaxPos/max(abs(x_IR_04_MaxPos))
    x_IR_04_Min_NOR = x_IR_04_Min/max(abs(x_IR_04_Min))
    x_IR_04_MinPos_NOR = x_IR_04_MinPos/max(abs(x_IR_04_MinPos))
    x_IR_04_RelMaxMin_NOR = x_IR_04_RelMaxMin/max(abs(x_IR_04_RelMaxMin))
    x_IR_04_Max_AllG_NOR = x_IR_04_Max_AllG/max(abs(x_IR_04_Max_AllG))
    x_IR_04_MaxPos_AllG_NOR = x_IR_04_MaxPos_AllG/max(abs(x_IR_04_MaxPos_AllG))
    x_IR_04_Min_AllG_NOR = x_IR_04_Min_AllG/max(abs(x_IR_04_Min_AllG))
    x_IR_04_MinPos_AllG_NOR = x_IR_04_MinPos_AllG/max(abs(x_IR_04_MinPos_AllG))
    x_IR_05_Max_1aDer_NOR = x_IR_05_Max_1aDer/max(abs(x_IR_05_Max_1aDer))
    x_IR_05_MaxPos_1aDer_NOR = x_IR_05_MaxPos_1aDer/max(abs(x_IR_05_MaxPos_1aDer))
    x_IR_05_Min_1aDer_NOR = x_IR_05_Min_1aDer/max(abs(x_IR_05_Min_1aDer))
    x_IR_05_MinPos_1aDer_NOR = x_IR_05_MinPos_1aDer/max(abs(x_IR_05_MinPos_1aDer))
    x_IR_06_StDev_NOR = x_IR_06_StDev/max(abs(x_IR_06_StDev))
    x_IR_06_StDev_1aDer_NOR = x_IR_06_StDev_1aDer/max(abs(x_IR_06_StDev_1aDer))  
    
    
    #pdStruct = pd.DataFrame(np.hstack((x_IR_02_RelAscentdent_NOR, x_IR_04_Max_NOR, x_IR_04_MaxPos_NOR)))
  #  pdStruct = pd.DataFrame(np.hstack([x_IR_02_RelAscentdent_NOR, x_IR_04_Max_NOR, x_IR_04_MaxPos_NOR]))

    dfIdentifiers = pd.DataFrame(D[:,:4])
    dfIdentifiers.columns =['image', 'blob', 'y', 'x']
    dfIdentifiers.astype('int32').dtypes
    
    dfDescriptors = pd.DataFrame([x_IR_01_canviSignePos_NOR,x_IR_01_canviSigneNeg_NOR,x_IR_02_Ascendent_NOR,
                          x_IR_02_Descendent_NOR,x_IR_02_RelAscentdent_NOR,x_IR_03_relacioPos_NOR,
                          x_IR_04_Max_NOR,x_IR_04_MaxPos_NOR,x_IR_04_Min_NOR,x_IR_04_MinPos_NOR,x_IR_04_RelMaxMin_NOR,
                          x_IR_04_Max_AllG_NOR,x_IR_04_MaxPos_AllG_NOR,x_IR_04_Min_AllG_NOR,x_IR_04_MinPos_AllG_NOR,
                          x_IR_05_Max_1aDer_NOR,x_IR_05_MaxPos_1aDer_NOR,x_IR_05_Min_1aDer_NOR,x_IR_05_MinPos_1aDer_NOR,
                          x_IR_06_StDev_NOR,x_IR_06_StDev_1aDer_NOR]).T
    dfDescriptors.columns = ['01_canviSignePos_NOR','02_canviSigneNeg_NOR','03_Ascendent_NOR',
                             '04_Descendent_NOR','05_RelAscentdent_NOR','06_relacioPos_NOR',
                             '07_Max_NOR','08_MaxPos_NOR','09_Min_NOR','10_MinPos_NOR','11_RelMaxMin_NOR',
                             '12_Max_AllG_NOR','13_MaxPos_AllG_NOR','14_Min_AllG_NOR','15_MinPos_AllG_NOR',
                             '16_Max_1aDer_NOR','17_MaxPos_1aDer_NOR','18_Min_1aDer_NOR','19_MinPos_1aDer_NOR',
                             '20_StDev_NOR','21_StDev_1aDer_NOR']
    dfSignal  = pd.DataFrame(g_IR_norm)
    
    dfComplet = pd.concat([dfIdentifiers, dfSignal,dfDescriptors], axis=1)
    
    return dfComplet

def extractPatternsAllBlobs(data,nameMaterial):

    idx0 = data.columns.get_loc(0)
    spectPixels = data.iloc[:,idx0:(idx0+224)]
    sdPixels= np.zeros(len(spectPixels))
    for n in range(len(data)):
        sdPixels[n] = np.std(spectPixels.iloc[n])
    
    first_quartile = np.quantile(sdPixels, 0.25)
    third_quartile = np.quantile(sdPixels, 0.75) 
                            
    pattern = np.zeros(224)
    num=0
    for n in range(len(data)):
        desStPix = np.std(spectPixels.iloc[n])
        if(desStPix>=first_quartile and desStPix<= third_quartile):
            pattern += spectPixels.iloc[n]
            num +=1
    pattern = pattern/num 

    for n in range(len(data)):
         sdPixels[n] = math.sqrt(((spectPixels.iloc[n] - pattern)**2).sum()/224)
    first_quartile = np.quantile(sdPixels, 0.25)
    third_quartile = np.quantile(sdPixels, 0.75)
    sdPattern = third_quartile+(third_quartile-first_quartile)#*1.5                            
                            
    return (nameMaterial,pattern,sdPattern)