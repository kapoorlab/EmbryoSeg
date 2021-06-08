from tifffile import imread, imwrite
from pathlib import Path
from csbdeep.io import load_training_data


class NPZCrops(object):

        def __init__(self, BaseDir, NPZfilename):

               self.BaseDir = BaseDir
               self.NPZfilename = NPZfilename
               self.MakeNPZCrops()

        def MakeNPZCrops(self):


                   Path(self.BaseDir + '/CropRaw/').mkdir(exist_ok=True)
                   Path(self.BaseDir + '/CropRealMask/').mkdir(exist_ok=True)
                   Path(self.BaseDir + '/CropValRaw/').mkdir(exist_ok=True)
                   Path(self.BaseDir + '/CropValRealMask/').mkdir(exist_ok=True) 

                   load_path = self.BaseDir + self.NPZfilename + 'Star' + '.npz'

                   (X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)



                   count = 0
                   countVal = 0
                   for i in range(0,X_val.shape[0]):
                              image = X_val[i]
                              mask = Y_val[i]
                              imwrite(self.BaseDir + '/CropValRaw/' + str(countVal) + '.tif', image[...,0] )
                              imwrite(self.BaseDir + '/CropValRealMask/' + str(countVal) + '.tif', mask[...,0].astype('uint16') )
                              countVal = countVal + 1

 
                   for i in range(0,X.shape[0]):
                              image = X[i]
                              mask = Y[i]
                              imwrite(self.BaseDir + '/CropRaw/' + str(count) + '.tif', image[...,0] )
                              imwrite(self.BaseDir + '/CropRealMask/' + str(count) + '.tif', mask[...,0].astype('uint16') )
                              count = count + 1  

                             
