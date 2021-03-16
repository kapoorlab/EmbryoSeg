#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 13:58:06 2021

@author: vkapoor
"""

from csbdeep.data import RawData, create_patches
from tifffile import imread, imwrite
from pathlib import Path
import glob
from skimage.morphology import label 
import os
import cv2
import numpy as np
from random import sample
from scipy.ndimage.filters import minimum_filter, maximum_filter
from tqdm import tqdm
from csbdeep.utils import normalize
from csbdeep.io import load_training_data
class Crops2D(object):

       def __init__(self, BaseDir, NPZfilename, PatchY, PatchX, n_patches_per_image, GenerateNPZ = True, validation_split = 0.1):

              self.BaseDir = BaseDir
              self.NPZfilename = NPZfilename
              self.GenerateNPZ = GenerateNPZ
              self.PatchY = PatchY
              self.PatchX = PatchX
              self.n_patches_per_image = n_patches_per_image
              self.validation_split = validation_split
              self.MakeCrops()  
              
       def MakeCrops(self):

           
                      # Create training data given either labels or binary image was the input
                      
                      self.RawDir = self.BaseDir + '/Raw/'
                      self.LabelDir = self.BaseDir + '/RealMask/'
                      self.BinaryDir = self.BaseDir + '/BinaryMask/'
                      

                      Raw = sorted(glob.glob(self.RawDir + '*.tif'))
                      RealMask = sorted(glob.glob(self.LabelDir + '*.tif'))
                      
                      
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
                            
                       
                            Binaryimage = image > 0
                    
                            imwrite((self.BinaryDir + Name + '.tif'), Binaryimage.astype('uint16'))     


                     
                      #Create some validation images for stardist


                      #For training Data of U-Net
                      if self.GenerateNPZ:
                              binary_raw_data = RawData.from_folder (
                              basepath    = self.BaseDir,
                              source_dirs = ['Raw/'],
                              target_dir  = 'BinaryMask/',
                              axes        = 'YX',
                               )
        
                              X, Y, XY_axes = create_patches (
                              raw_data            = binary_raw_data,
                              patch_size          = (self.PatchY,self.PatchX),
                              n_patches_per_image = self.n_patches_per_image,
                              save_file           = self.BaseDir + self.NPZfilename + '.npz',
                              )
                   
                             

                              
                              
                              
                              
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
                              
                              
                              
                              


