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


                      #For training Data of U-Net
                      
                      binary_raw_data = RawData.from_folder (
                      basepath    = self.BaseDir,
                      source_dirs = ['Raw/'],
                      target_dir  = 'BinaryMask/',
                      axes        = 'ZYX',
                       )

                      X, Y, XY_axes = create_patches (
                      raw_data            = binary_raw_data,
                      patch_size          = (self.PatchZ,self.PatchY,self.PatchX),
                      n_patches_per_image = self.n_patches_per_image,
                      save_file           = self.BaseDir + self.NPZfilename + '.npz',
                      )
           
           
           
                      #For training Data of Stardist
                      Path(self.BaseDir + '/BigCropRaw/').mkdir(exist_ok=True)
                      Path(self.BaseDir + '/BigCropRealMask/').mkdir(exist_ok=True)
                      
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
                      normalization = None,
                      save_file           = self.BaseDir + self.NPZfilename + 'BigStar' + '.npz',
                      )
  
                      count = 0
                      for i in range(0,X.shape[0]):
                              image = X[i]
                              mask = Y[i]
                              imwrite(self.BaseDir + '/BigCropRaw/' + str(count) + '.tif', image.astype('float32') )
                              imwrite(self.BaseDir + '/BigCropRealMask/' + str(count) + '.tif', mask.astype('uint16') )
                              count = count + 1
 
                      #For validation Data of Stardist
                      Path(self.BaseDir + '/BigValCropRaw/').mkdir(exist_ok=True)
                      Path(self.BaseDir + '/BigValCropRealMask/').mkdir(exist_ok=True)
                      
                      val_raw_data = RawData.from_folder (
                      basepath    = self.BaseDir,
                      source_dirs = ['ValRaw/'],
                      target_dir  = 'ValRealMask/',
                      axes        = 'ZYX',
                       )

                      X_val, Y_val, XY_axes = create_patches (
                      raw_data            = val_raw_data,
                      patch_size          = (self.PatchZ,self.PatchY,self.PatchX),
                      n_patches_per_image = self.n_patches_per_image,
                      patch_filter  = None,
                      normalization = None,
                      save_file           = self.BaseDir + self.NPZfilename + 'BigStarValidation' + '.npz',
                      )
  
                      count = 0
                      for i in range(0,X_val.shape[0]):
                              image = X_val[i]
                              mask = Y_val[i]
                              imwrite(self.BaseDir + '/BigValCropRaw/' + str(count) + '.tif', image.astype('float32') )
                              imwrite(self.BaseDir + '/BigValCropRealMask/' + str(count) + '.tif', mask.astype('uint16') )
                              count = count + 1                                      