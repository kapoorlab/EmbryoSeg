from csbdeep.data import RawData, create_patches
from tifffile import imread, imwrite
from pathlib import Path
import glob
from skimage.morphology import label 
import os
import cv2
from random import sample
from scipy.ndimage.filters import minimum_filter, maximum_filter
class Crops(object):

       def __init__(self, BaseDir, NPZfilename, PatchZ, PatchY, PatchX, n_patches_per_image, validation_split = 0.1):

              self.BaseDir = BaseDir
              self.NPZfilename = NPZfilename
              self.PatchZ = PatchZ
              self.PatchY = PatchY
              self.PatchX = PatchX
              self.n_patches_per_image = n_patches_per_image
              self.validation_split = validation_split
              self.MakeCrops()  
              
       def MakeCrops(self):

           
                      # Create training data given either labels or binary image was the input
                      
                      self.LabelDir = self.BaseDir + '/RealMask/'
                      self.BinaryDir = self.BaseDir + '/BinaryMask/'
                      
                      self.CropRawDir = self.BaseDir + '/CropRaw/'
                      self.CropLabelDir = self.BaseDir + '/CropRealMask/'
                      
                      self.CropValRawDir = self.BaseDir + '/CropValRaw/'
                      self.CropValLabelDir = self.BaseDir + '/CropValRealMask/'
                      
                      Path(self.BinaryDir).mkdir(exist_ok=True)
                      Path(self.LabelDir).mkdir(exist_ok=True)
                    
                     
                      RealMask = sorted(glob.glob(self.LabelDir + '*.tif'))
                      Mask = sorted(glob.glob(self.BinaryDir + '*.tif'))

                      print('Instance segmentation masks:', len(RealMask))
                      if len(RealMask)== 0:
                        
                        print('Making labels')
                        Mask = sorted(glob.glob(self.BinaryDir + '*.tif'))
                        
                        for fname in Mask:
                    
                           image = imread(fname)
                    
                           Name = os.path.basename(os.path.splitext(fname)[0])
                    
                           Binaryimage = label(image) 
                    
                           imwrite((self.LabelDir + Name + '.tif'), Binaryimage.astype('uint16'))
                           
                           
                      print('Semantic segmentation masks:', len(Mask))
                      if len(Mask) == 0:
                          
                          print('Generating Binary images')
                          RealfilesMask = sorted(glob.glob(self.LabelDir + '*.tif'))  
                
                
                          for fname in RealfilesMask:
                    
                            image = ReadFloat(fname)
                    
                            Name = os.path.basename(os.path.splitext(fname)[0])
                            
                            image = minimum_filter(image, (1,4,4))
                            image = maximum_filter(image, (1,4,4))
                       
                            Binaryimage = image > 0
                    
                            imwrite((self.BinaryDir + Name + '.tif'), Binaryimage.astype('uint16'))     


                     
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
                      Path(self.CropRawDir).mkdir(exist_ok=True)
                      Path(self.CropLabelDir).mkdir(exist_ok=True)
                      
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
                      save_file           = self.BaseDir + self.NPZfilename + 'Star' + '.npz',
                      )
  
                      count = 0
                      for i in range(0,X.shape[0]):
                              image = X[i]
                              mask = Y[i]
                              imwrite(self.CropRawDir + str(count) + '.tif', image.astype('float32') )
                              imwrite(self.CropLabelDir + str(count) + '.tif', mask.astype('uint16') )
                              count = count + 1
 
                      #For validation Data of Stardist
                      Path(self.CropValRawDir).mkdir(exist_ok=True)
                      Path(self.CropValLabelDir).mkdir(exist_ok=True)
                      
                      p = Path(self.CropRawDir)
                      image_names = [f.name for f in ( p ).glob('*.tif')]
                      shuffled = sample(image_names, len(image_names))
                      vallength = int(self.validation_split * X.shape[0])
                      image_names = shuffled[:vallength]
                      
                      Raw_path = os.path.join(self.CropRawDir, '*.tif')
                      filesRaw = glob.glob(Raw_path)
                      filesRaw.sort
                      
                      Label_path = os.path.join(self.CropLabelDir, '*.tif')
                      filesLabel = glob.glob(Label_path)
                      filesLabel.sort
                      
                      for fname in filesRaw:
                          
                          
                          Name = os.path.basename(os.path.splitext(fname)[0])
                          
                          
                          for secfname in filesLabel:
                              
                              SecName = os.path.basename(os.path.splitext(fname)[0])
                              
                              if Name == SecName and Name in image_names:
                                  
                                  
                                        imwrite(self.CropValRawDir + Name + '.tif', image.astype('float32') )
                                        imwrite(self.CropValLabelDir + Name + '.tif', mask.astype('uint16') )
                                  
                                        #Remove the validation images from the training set
                                        os.remove(self.CropRawDir + Name + '.tif')
                                        os.remove(self.CropLabelDir + Name + '.tif')
                          

                              
                              
                              
                              
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
                              
                              
                              
                              
