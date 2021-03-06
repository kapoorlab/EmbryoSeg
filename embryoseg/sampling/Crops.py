import sys
from csbdeep.data import RawData, create_patches
from tifffile import imread, imwrite
from pathlib import Path

class Crops(object):

       def __init__(self, BaseDir, NPZfilename, PatchZ, PatchY, PatchX, n_patches_per_image):

              self.BaseDir = BaseDir
              self.NPZfilename = NPZfilename
              self.PatchZ = PatchZ
              self.PatchY = PatchY
              self.PatchX = PatchX
              self.n_patches_per_image = n_patches_per_image
              self.MakeCrops()  
       def MakeCrops(self):


                    
                      Path(self.BaseDir + '/CropRaw/').mkdir(exist_ok=True)
                      Path(self.BaseDir + '/CropRealMask/').mkdir(exist_ok=True)
                    

                      raw_data = RawData.from_folder (
                      basepath    = self.BaseDir,
                      source_dirs = ['Raw/'],
                      target_dir  = 'RealMask/',
                      axes        = 'ZYX',
                       )

                      X, Y, XY_axes = create_patches (
                      raw_data            = raw_data,
                      patch_size          = (self.PatchZ,self.PatchY,self.PatchX),
                      n_patches_per_image = self.n_patches_per_image,
                      patch_filter  = None,
                      save_file           = self.BaseDir + self.NPZfilename + 'Star' + '.npz',
                      )

                      count = 0
                      for i in range(0,X.shape[0]):
                              image = X[i]
                              mask = Y[i]
                              imwrite(self.BaseDir + '/CropRaw/' + str(count) + '.tif', image[...,0] )
                              imwrite(self.BaseDir + '/CropRealMask/' + str(count) + '.tif', mask[...,0].astype('uint16') )
                              count = count + 1
