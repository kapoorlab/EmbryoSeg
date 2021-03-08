import sys
from csbdeep.data import RawData, create_patches
from tifffile import imread, imwrite
from pathlib import Path
import glob
from skimage.morphology import label 
import os
import cv2
from scipy.ndimage.filters import minimum_filter, maximum_filter
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

           
                      # Create training data given either labels or binary image was the input
                      
                      Path(self.BaseDir + '/BinaryMask/').mkdir(exist_ok=True)
                      Path(self.BaseDir + '/RealMask/').mkdir(exist_ok=True)
                    
                      Raw = sorted(glob.glob(self.BaseDir + '/Raw/' + '*.tif'))
                      RealMask = sorted(glob.glob(self.BaseDir + '/RealMask/' + '*.tif'))
                      Mask = sorted(glob.glob(self.BaseDir + '/BinaryMask/' + '*.tif'))

                      print('Instance segmentation masks:', len(RealMask))
                      if len(RealMask)== 0:
                        
                        print('Making labels')
                        Mask = sorted(glob.glob(self.BaseDir + '/BinaryMask/' + '*.tif'))
                        
                        for fname in Mask:
                    
                           image = imread(fname)
                    
                           Name = os.path.basename(os.path.splitext(fname)[0])
                    
                           Binaryimage = label(image) 
                    
                           imwrite((self.BaseDir + '/RealMask/' + Name + '.tif'), Binaryimage.astype('uint16'))
                           
                           
                      print('Semantic segmentation masks:', len(Mask))
                      if len(Mask) == 0:
                          
                          print('Generating Binary images')
                          RealfilesMask = sorted(glob.glob(self.BaseDir + '/RealMask/'+ '*tif'))  
                
                
                          for fname in RealfilesMask:
                    
                            image = ReadFloat(fname)
                    
                            Name = os.path.basename(os.path.splitext(fname)[0])
                            
                            image = minimum_filter(image, (1,4,4))
                            image = maximum_filter(image, (1,4,4))
                       
                            Binaryimage = image > 0
                    
                            imwrite((self.BaseDir + '/BinaryMask/' + Name + '.tif'), Binaryimage.astype('uint16'))     


                     
                      #Create some validation images for stardist


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
                              
                              
                              
                              
                              
def ReadFloat(fname):

    return imread(fname).astype('float32')         
         

def ReadInt(fname):

    return imread(fname).astype('uint16')         



         
def DownsampleData(image, DownsampleFactor):
                    


                    scale_percent = int(100/DownsampleFactor) # percent of original size
                    width = int(image.shape[2] * scale_percent / 100)
                    height = int(image.shape[1] * scale_percent / 100)
                    dim = (width, height)
                    smallimage = np.zeros([image.shape[0],  height,width])
                    for i in range(0, image.shape[0]):
                          # resize image
                          smallimage[i,:] = cv2.resize(image[i,:].astype('float32'), dim)         
         
                    return smallimage                              
                              
                              
                              
                              