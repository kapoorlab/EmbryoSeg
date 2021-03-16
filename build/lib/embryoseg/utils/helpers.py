#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:08:41 2019
@author: aimachine
"""

from __future__ import print_function, unicode_literals, absolute_import, division
#import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import collections
import glob
from stardist.models import StarDist3D
from tifffile import imread, imwrite
import warnings
from skimage.morphology import erosion, dilation, square
from scipy.ndimage.interpolation import zoom
from skimage.morphology import skeletonize
import cv2
import pandas as pd
from skimage.filters import gaussian
from six.moves import reduce
from skimage.feature import canny
from skimage.measure import regionprops_table
from matplotlib import cm
from skimage.filters import threshold_local, threshold_mean, threshold_otsu
from skimage.morphology import remove_small_objects, thin
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from skimage.transform import rescale
from tifffile import imsave, imwrite
from scipy.ndimage.morphology import binary_fill_holes
from skimage.measure import LineModelND, ransac
from skimage.segmentation import watershed,random_walker
from skimage.segmentation import morphological_geodesic_active_contour
from scipy.ndimage import measurements
from btrack.constants import BayesianUpdates
from scipy import ndimage as ndi
from skimage.util import invert
from pathlib import Path

from skimage.segmentation import  relabel_sequential
from skimage import morphology
from skimage import segmentation
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import  binary_dilation, binary_erosion
from skimage.util import invert as invertimage
from skimage import filters
from skimage import measure
from scipy.ndimage.filters import median_filter, gaussian_filter, maximum_filter, minimum_filter
from skimage.filters import sobel
from skimage.measure import label
from scipy import spatial
from csbdeep.utils import normalize
import zipfile
import btrack
from btrack.dataio import  _PyTrackObjectFactory
from csbdeep.utils import plot_some  


def CreateFakeZLabel(filesRaw, FakeZData_dir, FakeZLabel_dir, filesLabel, CommonName, FakeZ,SizeY, SizeX):
    
    StackImage = np.zeros([FakeZ, SizeY, SizeX], dtype='float32')
    StackLabel = np.zeros([FakeZ, SizeY, SizeX], dtype='uint16')
    FileNumber = CommonFiles(filesRaw, CommonName[0]) 
    if FileNumber > FakeZ:   
           count = 0
           for fname in filesRaw:

              Name = os.path.basename(os.path.splitext(fname)[0])
              
              for Labelfname in filesLabel:

                  LabelName = os.path.basename(os.path.splitext(Labelfname)[0])

                  if Name == LabelName and CommonName[0] in Name:
                    

                           
                           image = imread(fname)
                           labelimage = imread(Labelfname)
                           StackImage[count,:] = image
                           StackLabel[count,:] = labelimage
                           count = count + 1
                           
                           if count%FakeZ == 0:
                                count = 0
                                 
                                imwrite(FakeZData_dir + '/' + Name + '.tif', StackImage.astype('float32'))
                                imwrite(FakeZLabel_dir + '/' + Name + '.tif',StackLabel.astype('uint16'))
            

def CommonFiles(filesRaw, CommonName):
    
    count = 0
    for fname in filesRaw:
               if CommonName in fname:
                   count = count  + 1
                     
    return count
def _fill_label_holes(lbl_img, **kwargs):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (set(np.unique(lbl_img)) - set([0])):
        mask = lbl_img==l
        mask_filled = binary_fill_holes(mask,**kwargs)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled
def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""
    # TODO: refactor 'fill_label_holes' and 'edt_prob' to share code
    def grow(sl,interior):
        return tuple(slice(s.start-int(w[0]),s.stop+int(w[1])) for s,w in zip(sl,interior))
    def shrink(interior):
        return tuple(slice(int(w[0]),(-1 if w[1] else None)) for w in interior)
    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i,sl in enumerate(objects,1):
        if sl is None: continue
        interior = [(s.start>0,s.stop<sz) for s,sz in zip(sl,lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl,interior)]==i
        mask_filled = binary_fill_holes(grown_mask,**kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def dilate_label_holes(lbl_img, iterations):
    lbl_img_filled = np.zeros_like(lbl_img)
    for l in (range(np.min(lbl_img), np.max(lbl_img) + 1)):
        mask = lbl_img==l
        mask_filled = binary_dilation(mask,iterations = iterations)
        lbl_img_filled[mask_filled] = l
    return lbl_img_filled


def remove_big_objects(ar, max_size=6400, connectivity=1, in_place=False):
    
    out = ar.copy()
    ccs = out

    try:
        component_sizes = np.bincount(ccs.ravel())
    except ValueError:
        raise ValueError("Negative value labels are not supported. Try "
                         "relabeling the input with `scipy.ndimage.label` or "
                         "`skimage.morphology.label`.")



    too_big = component_sizes > max_size
    too_big_mask = too_big[ccs]
    out[too_big_mask] = 0

    return out

def multiplotline(plotA, plotB, plotC, titleA, titleB, titleC, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].plot(plotA)
    ax[0].set_title(titleA)
   
    ax[1].plot(plotB)
    ax[1].set_title(titleB)
    
    ax[2].plot(plotC)
    ax[2].set_title(titleC)
    
    plt.tight_layout()
    
    if plotTitle is not None:
      Title = plotTitle
    else :
      Title = 'MultiPlot'   
    if targetdir is not None and File is not None:
      plt.savefig(targetdir + Title + File + '.png')
    if targetdir is not None and File is None:
      plt.savefig(targetdir + Title + File + '.png')
    plt.show()

def polyroi_bytearray(x,y,pos=None):
    """ Byte array of polygon roi with provided x and y coordinates
        See https://github.com/imagej/imagej1/blob/master/ij/io/RoiDecoder.java
    """
    def _int16(x):
        return int(x).to_bytes(2, byteorder='big', signed=True)
    def _uint16(x):
        return int(x).to_bytes(2, byteorder='big', signed=False)
    def _int32(x):
        return int(x).to_bytes(4, byteorder='big', signed=True)

    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = np.round(x)
    y = np.round(y)
    assert len(x) == len(y)
    top, left, bottom, right = y.min(), x.min(), y.max(), x.max() # bbox

    n_coords = len(x)
    bytes_header = 64
    bytes_total = bytes_header + n_coords*2*2
    B = [0] * bytes_total
    B[ 0: 4] = map(ord,'Iout')   # magic start
    B[ 4: 6] = _int16(227)       # version
    B[ 6: 8] = _int16(0)         # roi type (0 = polygon)
    B[ 8:10] = _int16(top)       # bbox top
    B[10:12] = _int16(left)      # bbox left
    B[12:14] = _int16(bottom)    # bbox bottom
    B[14:16] = _int16(right)     # bbox right
    B[16:18] = _uint16(n_coords) # number of coordinates
    if pos is not None:
        B[56:60] = _int32(pos)   # position (C, Z, or T)

    for i,(_x,_y) in enumerate(zip(x,y)):
        xs = bytes_header + 2*i
        ys = xs + 2*n_coords
        B[xs:xs+2] = _int16(_x - left)
        B[ys:ys+2] = _int16(_y - top)

    return bytearray(B)


 
def WingArea(LeftImage, RightImage):

   Leftcount =  np.sum(LeftImage > 0)
   Rightcount = np.sum(RightImage > 0)
   
   RightMinusLeft = Rightcount - Leftcount
   RightPlusLeft = Rightcount + Leftcount
   Assymetery = 2 * RightMinusLeft / RightPlusLeft
   
   return Rightcount, Leftcount, RightMinusLeft, RightPlusLeft, Assymetery


def Resizer3D(image, SizeX, SizeY, SizeZ):
    
    ResizeImage = np.zeros([SizeZ, SizeY, SizeX])
    j = 0
    for i in range(0, ResizeImage.shape[1]):
        
        if j < image.shape[1]:
           ResizeImage[:image.shape[0],i,:image.shape[2]] = image[:image.shape[0],j,:image.shape[2]]
           j = j + 1
        else:
            j = 0   
        
    j = 0
    for i in range(0, ResizeImage.shape[2]):
        
        if j < image.shape[2]:
           ResizeImage[:,:,i] = ResizeImage[:,:,j]
           j = j + 1
        else:
            j = 0     

    j = 0
    for i in range(0, ResizeImage.shape[0]):
        
        if j < image.shape[0]:
           ResizeImage[i,:,:] = ResizeImage[j,:,:]
           j = j + 1
        else:
            j = 0  
    

    return ResizeImage

def Resizer2D(image, SizeX, SizeY):
    
    ResizeImage = np.zeros([SizeY, SizeX])
    j = 0
    for i in range(0, ResizeImage.shape[1]):
        
        if j < image.shape[1]:
           ResizeImage[:image.shape[0],i] = image[:image.shape[0],j]
           j = j + 1
        else:
            j = 0   
        


    j = 0
    for i in range(0, ResizeImage.shape[0]):
        
        if j < image.shape[0]:
           ResizeImage[i,:] = ResizeImage[j,:]
           j = j + 1
        else:
            j = 0  
    

    return ResizeImage
    

def MasktoRoi(Mask, Savedir, Name):
    
    Edge = sobel(Mask)
    X = []
    Y = []
    for xindex,yindex in np.ndindex(Edge.shape):
        X.append(xindex)
        Y.append(yindex)
    roi = polyroi_bytearray(Y,X)
    zf = zipfile.ZipFile(Savedir+  Name + ".roi", mode="w", compression=zipfile.ZIP_DEFLATED)
    zf.writestr( Name, roi)
    zf.close()


def BinaryDilation(Image, iterations = 1):

    DilatedImage = binary_dilation(Image, iterations = iterations) 
    
    return DilatedImage


def CCLabels(fname, max_size = 15000):
    
    
    BinaryImageOriginal = imread(fname)
    Orig = normalizeFloatZeroOne(BinaryImageOriginal)
    InvertedBinaryImage = invertimage(BinaryImageOriginal)
    BinaryImage = normalizeFloatZeroOne(InvertedBinaryImage)
    image = binary_dilation(BinaryImage)
    image = invertimage(image)
    IntegerImage = label(image)
    labelclean = remove_big_objects(IntegerImage, max_size = max_size) 
    AugmentedLabel = dilation(labelclean, selem = square(3) )
    AugmentedLabel = np.multiply(AugmentedLabel ,  Orig)
    

    return AugmentedLabel 

def merge_labels_across_volume(labelvol, relabelfunc, threshold=3):
    nz, ny, nx = labelvol.shape
    res = np.zeros_like(labelvol)
    res[0,...] = labelvol[0,...]
    backup = labelvol.copy() # kapoors code modifies the input array
    for i in tqdm(range(nz-1)):
        
        res[i+1,...] = relabelfunc(res[i,...], labelvol[i+1,...],threshold=threshold)
        labelvol = backup.copy() # restore the input array
    return res

def RelabelZ(previousImage, currentImage,threshold):
      # This line ensures non-intersecting label sets
      copyImage = currentImage.copy()
      copypreviousImage = previousImage.copy()
      copyImage = relabel_sequential(copyImage,offset=copypreviousImage.max()+1)[0]
        # I also don't like modifying the input image, so we take a copy
      relabelimage = copyImage.copy()
      waterproperties = measure.regionprops(copypreviousImage, copypreviousImage)
      indices = [] 
      labels = []
      for prop in waterproperties:
        if prop.label > 0:
                 
                  labels.append(prop.label)
                  indices.append(prop.centroid) 
     
      if len(indices) > 0:
        tree = spatial.cKDTree(indices)
        currentwaterproperties = measure.regionprops(copyImage, copyImage)
        currentindices = [prop.centroid for prop in currentwaterproperties] 
        currentlabels = [prop.label for prop in currentwaterproperties] 
        if len(currentindices) > 0: 
            for i in range(0,len(currentindices)):
                index = currentindices[i]
                currentlabel = currentlabels[i] 
                if currentlabel > 0:
                        previouspoint = tree.query(index)
                        for prop in waterproperties:
                               
                                      if int(prop.centroid[0]) == int(indices[previouspoint[1]][0]) and int(prop.centroid[1]) == int(indices[previouspoint[1]][1]):
                                                previouslabel = prop.label
                                                break
                        
                        if previouspoint[0] > threshold:
                              relabelimage[np.where(copyImage == currentlabel)] = currentlabel
                        else:
                              relabelimage[np.where(copyImage == currentlabel)] = previouslabel
      
                              

    
      return relabelimage


def show_ransac_points_line(points,  min_samples=2, residual_threshold=0.1, max_trials=1000, Xrange = 100, displayoutlier= False):
   
    # fit line using all data
 model = LineModelND()
 if(len(points) > 2):
  model.estimate(points)
 
  fig, ax = plt.subplots()   

  # robustly fit line only using inlier data with RANSAC algorithm
  model_robust, inliers = ransac(points, LineModelND, min_samples=min_samples,
                               residual_threshold=residual_threshold, max_trials=max_trials)
  slope , intercept = model_robust.params
 
  outliers = inliers == False
  # generate coordinates of estimated models
  line_x = np.arange(0, Xrange)
  line_y = model.predict_y(line_x)
  line_y_robust = model_robust.predict_y(line_x)
 
  #print('Model Fit' , 'yVal = ' , line_y_robust)
  #print('Model Fit', 'xVal = ' , line_x)
  ax.plot(points[inliers, 0], points[inliers, 1], '.b', alpha=0.6,
        label='Inlier data')
  if displayoutlier:
   ax.plot(points[outliers, 0], points[outliers, 1], '.r', alpha=0.6,
        label='Outlier data')
  #ax.plot(line_x, line_y, '-r', label='Normal line model')
  ax.plot(line_x, line_y_robust, '-g', label='Robust line model')
  ax.legend(loc='upper left')
   
  ax.set_xlabel('Time (s)')
  ax.set_ylabel('Thickness (um)')
  print('Ransac Slope = ', str('%.3e'%((line_y_robust[Xrange - 1] - line_y_robust[0])/ (Xrange)) )) 
  print('Regression Slope = ', str('%.3e'%((line_y[Xrange - 1] - line_y_robust[0])/ (Xrange)) )) 
  print('Mean Thickness (After outlier removal) = ', str('%.3f'%(sum(points[inliers, 1])/len(points[inliers, 1]))), 'um')   
  plt.show()
  
  
def SMA(Velocity, Window):
    
    Moving_Average = []
    i = 0
    while i < len(Velocity) - Window + 1:

          this_window = Velocity[i:i + Window] 
          window_average = sum(this_window) / Window
          Moving_Average.append(window_average)
          i = i + 1  
            
    return Moving_Average       
    
def MakeInvertLabels(Binaryimage, Name, savedir):
    
  
  image = invertimage(Binaryimage)
   
  labelimage = label(image)  


  seqlabelimage, forward_map, inverse_map = relabel_sequential(labelimage) 
  
  imwrite((savedir + Name + '.tif'), seqlabelimage)
  

def MakeLabels(Binaryimage, Name, savedir):
    
  labelimage = label(Binaryimage)  


  seqlabelimage, forward_map, inverse_map = relabel_sequential(labelimage) 
  
  imwrite((savedir + Name + '.tif'), seqlabelimage)


def Prob_to_Binary(Image, Label):
    
    #Cutoff high threshold instead of low ones which are the boundary pixels
    ReturnImage = np.zeros([Image.shape[0], Image.shape[1] ])
    properties = measure.regionprops(Label, Image)
    Labelindex = [prop.label for prop in properties]
    IntensityImage = [prop.intensity_image for prop in properties]
    BoxImage = [prop.bbox for prop in properties]
    
    
    
    
    
    for i in range(0,len(Labelindex)):
        
        currentimage = IntensityImage[i]
        min_row, min_col, max_row, max_col = BoxImage[i]
        
        
        for xindex,yindex in np.ndindex(currentimage.shape):
            if currentimage[xindex,yindex] > 0:
                     
                     
                     if Image[min_row + xindex, min_col + yindex] > 0:
                
                        
                        ReturnImage[min_row + xindex, min_col + yindex] = 1
        
    
    ReturnImage = binary_fill_holes(ReturnImage)
    return ReturnImage
    
class Updater:
    def __init__(self, viewer , AllImages, AllSegImages, DisplayName, AllName, AllLabel, AllSphericity, AllVolume, AllArea, AllCentroid, AllIntensity):
           
             self.viewer = viewer
             self.AllImages = AllImages
             self.AllSegImages = AllSegImages
             self.DisplayName = DisplayName
             self.AllName = AllName
             self.AllLabel = AllLabel
             self.AllSphericity = AllSphericity
             self.AllVolume = AllVolume
             self.AllArea = AllArea
             self.AllCentroid = AllCentroid
             self.AllIntensity = AllIntensity
             self.time  = int(self.viewer.dims.point[0])
             self.z =  int(self.viewer.dims.point[1])
             
             self.viewer.dims.events.axis.connect(self.update_slider)
             self.shapes_layer = None
             self.Allshapes = []
    def update_slider(self, event):
                 
                  if event.axis == 0:
                          self.time = int(self.viewer.dims.point[0])
                          if len(self.Allshapes) > 0:
                                 for layer in self.Allshapes:
                                    try: 
                                      self.viewer.layers.remove(layer)
                                    except:
                                        pass
                                 self.Allshapes = []   
                          for i in range(0, len(self.AllName)):
                            if self.DisplayName[self.time] == self.AllName[i]:
                                
                               self.shapes_layer =  NapariText(self.viewer,self.AllSegImages[self.time], self.AllLabel[i], self.AllSphericity[i], self.AllVolume[i], self.AllArea[i], self.AllCentroid[i], self.AllIntensity[i], self.z, self.shapes_layer)
                               self.Allshapes.append(self.shapes_layer)
                  
def make_bbox(bbox_extents):
    """Get the coordinates of the corners of a
    bounding box from the extents
    Parameters
    ----------
    bbox_extents : list (4xN)
        List of the extents of the bounding boxes for each of the N regions.
        Should be ordered: [min_row, min_column, max_row, max_column]
    Returns
    -------
    bbox_rect : np.ndarray
        The corners of the bounding box. Can be input directly into a
        napari Shapes layer.
    """
    minr = bbox_extents[0]              
    minc = bbox_extents[1]
    maxr = bbox_extents[2]
    maxc = bbox_extents[3]

    bbox_rect = np.array(
        [[minr, minc], [maxr, minc], [maxr, maxc], [minr, maxc]]
    )
    bbox_rect = np.moveaxis(bbox_rect, 2, 0)

    return bbox_rect



def MakeTrees(segimage):
    
        AllTrees = {}
        for i in tqdm(range(0, segimage.shape[0])):
                currentimage = segimage[i, :].astype('uint16')
                waterproperties = measure.regionprops(currentimage, currentimage)
                indices = [prop.centroid for prop in waterproperties] 
                if len(indices) > 0:
                      tree = spatial.cKDTree(indices)
                
                      AllTrees[str(i)] =  [tree, indices]
                    
                    
                           
        return AllTrees


def ReturnLocalizations(TZYXimage, TZYXlabeled):
    
    for i in range(0, TZYXimage.shape[0]):
          
            image = TZYXimage[i,:]
            labeled = TZYXlabeled[i,:]
        
            idx = [p for p in np.unique(labeled) if p>0]
            
            centroids = np.array(measurements.center_of_mass(image,
                                                             labels=labeled,
                                                             index=idx))
            
        
        
            localizations = np.zeros((centroids.shape[0], 5), dtype=np.uint16)
            localizations[:,0] = i # time
            localizations[:,1:centroids.shape[1]+1] = centroids
            localizations[:,-1] = 0 #labels-1 #-1 because we use a label of zero for states
        
        
        
            # get the labels from the image data
            labels = np.array(measurements.maximum(image, labels=labeled, index=idx))
            localizations[:,-1] = labels-1 #-1 because we use a label of zero for states
            AllLocations = [localizations]
    
    AllLocations = np.concatenate(AllLocations, axis = 0)
            
    n_localizations = AllLocations.shape[0]
    idx = range(n_localizations)
                  
    factory = _PyTrackObjectFactory()
    objects = [factory.get(AllLocations[i,:4], label=int(AllLocations[i,-1])) for i in idx]
            
    return objects
    


def TimeLapseMarkers(ImageDir, SaveDir,fname, StarModel, NoiseModel = None, min_size = 100, DownsampleFactor = 1, n_tiles = (2,2)):
    
    print('Generating Markers')
    
    MarkersResults = SaveDir
    Path(MarkersResults).mkdir(exist_ok = True)
    #Read Image
    image = imread(fname)
    TimeMarkers = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype='uint16')
    image = DownsampleData(image, DownsampleFactor)
    Name = os.path.basename(os.path.splitext(fname)[0])
    for i in range(0, image.shape[0]):
        
            smallimage = image[i,:]
            if NoiseModel is not None:
                
                print('Denoising Image')
                smallimage = NoiseModel.predict(smallimage,'YX', n_tiles = n_tiles)
                
            SmartSeeds, Markers = STARPrediction(smallimage, StarModel, min_size, n_tiles)
            TimeMarkers[i,:] = Markers
            doubleplot(smallimage, Markers, "Image", "Markers")    
    
            
    TimeMarkers = DownsampleData(TimeMarkers, 1.0/DownsampleFactor)
    imwrite((MarkersResults + Name+ '.tif' ) , TimeMarkers.astype('uint16'))     
    
    

def SmartSeedPrediction3D(ImageDir, SaveDir, fname,  UnetModel, StarModel, NoiseModel = None, min_size_mask = 100, min_size = 10, DownsampleFactor = 1, 
n_tiles = (1,2,2), doMask = True, smartcorrection = None, threshold = 20, projection = False, start = 0, end = -1, sizeY = None, sizeX = None, UseProbability = True, filtersize = 0):
    
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    DenoisedResults = SaveDir + 'Denoised/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    Path(SaveDir).mkdir(exist_ok = True)


    if StarModel is not None:
       Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    
    if projection == True:
       ProjectionResults = SaveDir + 'ProjectedMask/'
       ProjectionResultsRaw = SaveDir + 'ProjectedRaw/'
       Path(ProjectionResults).mkdir(exist_ok = True)    
       Path(ProjectionResultsRaw).mkdir(exist_ok = True)
    #Read Image
    image = imread(fname)
    sizeZ = image.shape[0]
    if sizeY is None:
      sizeY = image.shape[1]
    
    if sizeX is None:  
       sizeX = image.shape[2]
    
    SizedMask = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    SizedSmartSeeds = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    Sizedimage = np.zeros([sizeZ, sizeY, sizeX])
    
    print('Sizing Same', SizedMask.shape)
    Name = os.path.basename(os.path.splitext(fname)[0])
    image = DownsampleData(image, DownsampleFactor)
    
    if NoiseModel is not None:
        
        print('Denoising Image')
        image = NoiseModel.predict(image,'ZYX', n_tiles = n_tiles)


    if doMask:
          
          Mask = UNETPrediction3D(gaussian_filter(image, filtersize), UnetModel, n_tiles, 'ZYX')
          
          SizedMask[:, :Mask.shape[1], :Mask.shape[2]] = Mask
          if StarModel is not None:
              SmartSeeds, _, StarImage = STARPrediction3D(gaussian_filter(image,filtersize), StarModel,  n_tiles, MaskImage = Mask, smartcorrection = smartcorrection)
              #Upsample images back to original size
              SmartSeeds = DownsampleData(SmartSeeds, 1.0/DownsampleFactor)
              image = DownsampleData(image, 1.0/DownsampleFactor)
              Mask = DownsampleData(Mask, 1.0/DownsampleFactor)
              for i in range(0, Mask.shape[0]):
                  Mask[i,:] = remove_small_objects(Mask[i,:].astype('uint16'), min_size = min_size)
                  SmartSeeds[i,:] = remove_small_objects(SmartSeeds[i,:].astype('uint16'), min_size = min_size)
              SmartSeeds = RemoveLabels(SmartSeeds)       
              SizedSmartSeeds[:, :SmartSeeds.shape[1], :SmartSeeds.shape[2]] = SmartSeeds
          if StarModel is None:
              #Only upsample
              Mask = DownsampleData(Mask, 1.0/DownsampleFactor)
              image = DownsampleData(image, 1.0/DownsampleFactor)
              
              for i in range(0, Mask.shape[0]):
                  Mask[i,:] = remove_small_objects(Mask[i,:].astype('uint16'), min_size = min_size)

              doubleplot(image[image.shape[0]//2,:], Mask[Mask.shape[0]//2,:], "Image", "UNET")
              
             
 
    if doMask == False:
        
        Mask = UNETPrediction3D(gaussian_filter(image,filtersize), UnetModel, n_tiles,  'ZYX')
        
        SizedMask[:, :Mask.shape[1], :Mask.shape[2]]  = Mask
        if StarModel is not None:
            SmartSeeds, _,StarImage = STARPrediction3D(gaussian_filter(image,filtersize), StarModel, n_tiles)
            #Upsample images back to original size
            SmartSeeds = DownsampleData(SmartSeeds, 1.0/DownsampleFactor)
            image = DownsampleData(image, 1.0/DownsampleFactor)
            Mask = DownsampleData(Mask, 1.0/DownsampleFactor)
            
            
            for i in range(0, Mask.shape[0]):
                  Mask[i,:] = remove_small_objects(Mask[i,:].astype('uint16'), min_size = min_size)
                  SmartSeeds[i,:] = remove_small_objects(SmartSeeds[i,:].astype('uint16'), min_size = min_size)
            
            SmartSeeds = RemoveLabels(SmartSeeds)       
            SizedSmartSeeds[:, :SmartSeeds.shape[1], :SmartSeeds.shape[2]] = SmartSeeds
        if StarModel is None:
            #Only upsample
            Mask = DownsampleData(Mask, 1.0/DownsampleFactor)
            image = DownsampleData(image, 1.0/DownsampleFactor)
           
            for i in range(0, Mask.shape[0]):
                  Mask[i,:] = remove_small_objects(Mask[i,:].astype('uint16'), min_size = min_size)

            doubleplot(image[image.shape[0]//2,:], Mask[Mask.shape[0]//2,:], "Image", "UNET")    
            
    if NoiseModel is not None:

                   Path(DenoisedResults).mkdir(exist_ok = True)
                   imwrite((DenoisedResults + Name + '.tif' ) , image.astype('float32'))
           
            
    if StarModel is not None:
        imwrite((SmartSeedsResults + Name+ '.tif' ) , SizedSmartSeeds.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , SizedMask.astype('uint16')) 
    
    if projection:
        
        print('Projecting Unet mask')
        if end==-1:
            end = Mask.shape[0]
        ProjectMask = np.amax(SizedMask[start:end,:], axis = 0)
        ProjectionRaw = np.sum(Sizedimage, axis = 0)
        ProjectMask = remove_small_objects(ProjectMask.astype('uint16'), min_size)
        plt.imshow(ProjectMask)
        plt.show()
        imwrite((ProjectionResults + Name+ '.tif' ) , ProjectMask.astype('uint16')) 
        imwrite((ProjectionResultsRaw + Name+ '.tif' ) , ProjectionRaw.astype('float32')) 
        
    return SizedSmartSeeds, SizedMask    



def plot3D(ImageDir, ResultsDir, time = 0, Z = 0, showMany = -1):
    
    Raw_path = os.path.join(ImageDir, '*tif')
    filesRaw = glob.glob(Raw_path)
    filesRaw.sort
    
    Result_path = os.path.join(ResultsDir, '*tif')
    filesResult = glob.glob(Result_path)
    filesResult.sort
    
    totalfiles = len(filesRaw)
    if totalfiles > 5:
        totalfiles = 5
    count = 0
    X = []
    Y = []
    for fname in filesRaw:
     
         for secfname in filesResult:
                      
                image = imread(fname)
                ndim = len(image.shape)
                Name = os.path.basename(os.path.splitext(fname)[0])
                ResName = os.path.basename(os.path.splitext(secfname)[0])
                if Name == ResName:
                
                      Resimage = imread(secfname)
                      count = count + 1
                      print(count, ndim)
                      if ndim == 4:
                          X.append(image[time,Z,:])
                          Y.append(Resimage[time,Z,:])
                          
                      if ndim  == 3:
                          X.append(image[Z,:])
                          Y.append(Resimage[Z,:])
                      if ndim == 2:
                          X.append(image)
                          Y.append(Resimage)
                          
                if totalfiles > 1 and count%totalfiles == 0 and count > 0 :          
                  plt.figure(figsize=(2,5 * totalfiles))
                  plot_some(X[:totalfiles],Y[:totalfiles])
                  plt.show()

                  X = []
                  Y = []
                else:
                    plt.figure(figsize=(2,5))
                    plot_some(X[:totalfiles],Y[:totalfiles])
                    plt.show()
                  
                    X = []
                    Y = []
         if showMany > 0 and count>= showMany:
             break
         
            
            

def EmbryoSegFunction2D(ImageDir, SaveDir,fname,  UnetModel, StarModel, min_size_mask = 100, min_size = 20, DownsampleFactor = 1, n_tiles = (2,2), 
                          smartcorrection = None,  UseProbability = True, filtersize = 0):
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    Path(SaveDir).mkdir(exist_ok = True)
    Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    
    #Read Image
    image = imread(fname)
    Name = os.path.basename(os.path.splitext(fname)[0])
    image = DownsampleData2D(image, DownsampleFactor)
    
    Mask = UNETPrediction(gaussian_filter(image, filtersize), UnetModel, min_size_mask, n_tiles, 'YX')
    imwrite((UNETResults + Name+ '.tif' ) , Mask.astype('uint16')) 
    SmartSeeds, _, StarImage = STARPrediction(gaussian_filter(image, filtersize), StarModel, min_size, n_tiles, MaskImage = Mask, smartcorrection = smartcorrection, UseProbability = UseProbability)
    #Upsample downsampled results
    image = DownsampleData2D(image, 1.0/DownsampleFactor)
    Mask = DownsampleData2D(Mask, 1.0/DownsampleFactor)
    SmartSeeds = DownsampleData2D(SmartSeeds, 1.0/DownsampleFactor)
    
    
    imwrite((SmartSeedsResults + Name+ '.tif' ) , SmartSeeds.astype('uint16'))
 
    return SmartSeeds, Mask
    

def EmbryoSegFunction3D(ImageDir, SaveDir, fname,  UnetModel, StarModel, DownsampleFactor = 1, 
n_tiles = (1,2,2), smartcorrection = None,  start = 0, end = -1, sizeY = None, sizeX = None, UseProbability = True, filtersize = 0):
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    Path(SaveDir).mkdir(exist_ok = True)


    Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    
    #Read Image
    image = imread(fname)
    sizeZ = image.shape[0]
    if sizeY is None:
      sizeY = image.shape[1]
    
    if sizeX is None:  
       sizeX = image.shape[2]
    
    SizedMask = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    SizedSmartSeeds = np.zeros([sizeZ, sizeY, sizeX], dtype = 'uint16')
    
    print('Sizing Same', SizedMask.shape)
    Name = os.path.basename(os.path.splitext(fname)[0])
    image = DownsampleData(image, DownsampleFactor)
    
    Mask = UNETPrediction3D(gaussian_filter(image, filtersize), UnetModel, n_tiles, 'ZYX')
    SizedMask[:, :Mask.shape[1], :Mask.shape[2]] = Mask
    imwrite((UNETResults + Name+ '.tif' ) , SizedMask.astype('uint16'))
    
    
    SmartSeeds, _, StarImage = STARPrediction3D(gaussian_filter(image,filtersize), StarModel,  n_tiles, MaskImage = Mask)
    #Upsample images back to original size
    SmartSeeds = DownsampleData(SmartSeeds, 1.0/DownsampleFactor)
    image = DownsampleData(image, 1.0/DownsampleFactor)
    Mask = DownsampleData(Mask, 1.0/DownsampleFactor)
    SizedSmartSeeds[:, :SmartSeeds.shape[1], :SmartSeeds.shape[2]] = SmartSeeds
 
    imwrite((SmartSeedsResults + Name+ '.tif' ) , SizedSmartSeeds.astype('uint16'))
     
    
        
    return SizedSmartSeeds, SizedMask    

def SmartSeedPrediction3DTime(ImageDir, SaveDir, fname,  UnetModel, StarModel, NoiseModel = None, min_size_mask = 100, min_size = 10, DownsampleFactor = 1, 
n_tiles = (1,2,2), doMask = True, smartcorrection = None, threshold = 20, projection = False, start = 0, end = -1, sizeY = None, sizeX = None, UseProbability = True, filtersize = 0):
    
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    DenoisedResults = SaveDir + 'Denoised/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    Path(SaveDir).mkdir(exist_ok = True)


    if StarModel is not None:
       Path(SmartSeedsResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    
    if projection == True:
       ProjectionResults = SaveDir + 'ProjectedMask/'
       ProjectionResultsRaw = SaveDir + 'ProjectedRaw/'
       Path(ProjectionResults).mkdir(exist_ok = True)    
       Path(ProjectionResultsRaw).mkdir(exist_ok = True)
    #Read Image
    image = imread(fname)
    sizeT = image.shape[0]
    sizeZ = image.shape[1]
    if sizeY is None:
      sizeY = image.shape[2]
    
    if sizeX is None:  
       sizeX = image.shape[3]
    
    SizedMask = np.zeros([sizeT, sizeZ, sizeY, sizeX], dtype = 'uint8')
    Mask = np.zeros([sizeT, sizeZ, sizeY, sizeX], dtype = 'uint8')
    SizedSmartSeeds = np.zeros([sizeT, sizeZ, sizeY, sizeX], dtype = 'uint16')
    Sizedimage = np.zeros([sizeT, sizeZ, sizeY, sizeX])
    
    print('Size', SizedMask.shape)
    Name = os.path.basename(os.path.splitext(fname)[0])
    image = DownsampleData(image, DownsampleFactor)
    
    
    for i in tqdm(range(0, image.shape[0])):
            Mask[i,:] = UNETPrediction3D(gaussian_filter(image[i,:], filtersize), UnetModel, n_tiles,  'ZYX')
        
            SizedMask[i,:, :Mask.shape[2], :Mask.shape[3]]  = Mask[i,:]
    imwrite((UNETResults + Name+ '.tif' ) , SizedMask.astype('uint16'))         
    
    if NoiseModel is not None:
        
        print('Denoising Image')
        for i in range(0, image.shape[0]):
            image[i,:] = NoiseModel.predict(image[i,:],'ZYX', n_tiles = n_tiles)


    if doMask:

          if StarModel is not None:
              for i in tqdm(range(0, image.shape[0])):
                      SmartSeeds, _, StarImage = STARPrediction3D(gaussian_filter(image[i,:], filtersize), StarModel, n_tiles, MaskImage = Mask[i,:], smartcorrection = smartcorrection)
                      #Upsample images back to original size
                      SmartSeeds = DownsampleData(SmartSeeds, 1.0/DownsampleFactor)
                      image[i,:] = DownsampleData(image[i,:], 1.0/DownsampleFactor)
                      Mask[i,:] = DownsampleData(Mask[i,:], 1.0/DownsampleFactor)
                      for j in range(0, Mask.shape[1]):
                              Mask[i,j,:,:] = remove_small_objects(Mask[i,j,:,:].astype('uint16'), min_size = min_size)
                              SmartSeeds[j,:] = remove_small_objects(SmartSeeds[j,:].astype('uint16'), min_size = min_size)
                      SmartSeeds = RemoveLabels(SmartSeeds)          
                      SizedSmartSeeds[i,:, :SmartSeeds.shape[1], :SmartSeeds.shape[2]] = SmartSeeds
          if StarModel is None:
              #Only upsample
              for i in tqdm(range(0, image.shape[0])):
                   Mask[i,:] = DownsampleData(Mask[i,:], 1.0/DownsampleFactor)
                   image[i,:] = DownsampleData(image[i,:], 1.0/DownsampleFactor)
                    
                   for j in range(0, Mask.shape[1]):
                              Mask[i,j,:,:] = remove_small_objects(Mask[i,j,:,:].astype('uint16'), min_size = min_size)

 
    if doMask == False:

        if StarModel is not None:
            for i in tqdm(range(0, image.shape[0])):
                    SmartSeeds, _,StarImage = STARPrediction3D(gaussian_filter(image[i,:],filtersize), StarModel, n_tiles)
                    #Upsample images back to original size
                    SmartSeeds = DownsampleData(SmartSeeds, 1.0/DownsampleFactor)
                    image[i,:]  = DownsampleData(image[i,:] , 1.0/DownsampleFactor)
                    Mask[i,:]  = DownsampleData(Mask[i,:] , 1.0/DownsampleFactor)
                     
                    for j in range(0, Mask.shape[1]):
                              Mask[i,j,:,:] = remove_small_objects(Mask[i,j,:,:].astype('uint16'), min_size = min_size)
                              SmartSeeds[j,:] = remove_small_objects(SmartSeeds[j,:].astype('uint16'), min_size = min_size)
                    SmartSeeds = RemoveLabels(SmartSeeds)           
                    SizedSmartSeeds[i,:, :SmartSeeds.shape[2], :SmartSeeds.shape[3]] = SmartSeeds
        if StarModel is None:
            #Only upsample
            for i in tqdm(range(0, image.shape[0])):
                    Mask[i,:]  = DownsampleData(Mask[i,:] , 1.0/DownsampleFactor)
                    image[i,:]  = DownsampleData(image[i,:] , 1.0/DownsampleFactor)
                   
                    for j in range(0, Mask.shape[1]):
                              Mask[i,j,:,:] = remove_small_objects(Mask[i,j,:,:].astype('uint16'), min_size = min_size)
        
                    doubleplot(image[i,image.shape[1]//2,:], Mask[i,Mask.shape[1]//2,:], "Image", "UNET")    
            
    if NoiseModel is not None:

                   Path(DenoisedResults).mkdir(exist_ok = True)
                   imwrite((DenoisedResults + Name + '.tif' ) , image.astype('float32'))
           
            
    if StarModel is not None:
        imwrite((SmartSeedsResults + Name+ '.tif' ) , SizedSmartSeeds.astype('uint16'))
   
    
    if projection:
        
        print('Projecting Unet mask')
        if end==-1:
            end = Mask.shape[0]
        ProjectMask = np.amax(SizedMask[start:end,:], axis = 0)
        ProjectionRaw = np.sum(Sizedimage, axis = 0)
        ProjectMask = remove_small_objects(ProjectMask.astype('uint16'), min_size)
        plt.imshow(ProjectMask)
        plt.show()
        imwrite((ProjectionResults + Name+ '.tif' ) , ProjectMask.astype('uint16')) 
        imwrite((ProjectionResultsRaw + Name+ '.tif' ) , ProjectionRaw.astype('float32')) 


    return SizedSmartSeeds, Mask
    


def DownsampleData2D(image, DownsampleFactor):
                    
               if DownsampleFactor!=1:  
                    print('Downsampling Image in XY by', DownsampleFactor)
                    scale_percent = int(100/DownsampleFactor) # percent of original size
                    width = int(image.shape[1] * scale_percent / 100)
                    height = int(image.shape[0] * scale_percent / 100)
                    dim = (width, height)
                    image = cv2.resize(image.astype('float32'), dim)         
         
                    return image
               else:
                   
                   return image
                

def DownsampleData(image, DownsampleFactor):
                    
                if DownsampleFactor!=1:  
                    print('Downsampling Image in XY by', DownsampleFactor)
                    scale_percent = int(100/DownsampleFactor) # percent of original size
                    width = int(image.shape[2] * scale_percent / 100)
                    height = int(image.shape[1] * scale_percent / 100)
                    dim = (width, height)
                    smallimage = np.zeros([image.shape[0],  height,width])
                    for i in range(0, image.shape[0]):
                          # resize image
                          smallimage[i,:] = cv2.resize(image[i,:].astype('float32'), dim)         
         
                    return smallimage
                else:
                    
                    return image
                
def UNETPrediction(image, model, min_size, n_tiles, axis, threshold = 20):
    
    
    Segmented = model.predict(image, axis, n_tiles = n_tiles)
    
    try:
       thresh = threshold_otsu(Segmented)
       Binary = Segmented > thresh
    except:
        Binary = Segmented > 0
    #Postprocessing steps
    Filled = binary_fill_holes(Binary)
    Finalimage = label(Filled)
    Finalimage = fill_label_holes(Finalimage)
    Finalimage = relabel_sequential(Finalimage)[0]
    #Stitch 2D slices using relabel function
    if len(image.shape) > 2:
           Finalimage = merge_labels_across_volume(Finalimage.astype('uint16'), RelabelZ, threshold= threshold)
    return Finalimage


def UNETPrediction3D(image, model, n_tiles, axis):
    
    
    Segmented = model.predict(image, axis, n_tiles = n_tiles)
    
    try:
       thresh = threshold_otsu(Segmented)
       Binary = Segmented > thresh
    except:
        Binary = Segmented > 0
    #Postprocessing steps
    Filled = binary_fill_holes(Binary)
    Finalimage = label(Filled)
    Finalimage = fill_label_holes(Finalimage)
    Finalimage = relabel_sequential(Finalimage)[0]
    
          
    return Finalimage

def RemoveLabels(LabelImage, minZ = 2):
    
    properties = measure.regionprops(LabelImage, LabelImage)
    for prop in properties:
                regionlabel = prop.label
                sizeZ = abs(prop.bbox[0] - prop.bbox[3])
                if sizeZ <= minZ:
                    LabelImage[LabelImage == regionlabel] = 0
    return LabelImage                

def STARPrediction3D(image, model, n_tiles, MaskImage = None, smartcorrection = None, UseProbability = True):
    
    copymodel = model
    image = normalize(image, 1, 99.8, axis = (0,1,2))
    shape = [image.shape[1], image.shape[2]]
    image = zero_pad_time(image, 64, 64)
    grid = copymodel.config.grid

    try:
         MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
         SmallProbability, SmallDistance = model.predict(image, n_tiles = n_tiles)

    except:
            conf = copymodel.config
            Dummy = StarDist3D(conf)
            overlap = Dummy._axes_tile_overlap('ZYX')
            model._tile_overlap = [overlap]
            MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
            SmallProbability, SmallDistance = model.predict(image, n_tiles = n_tiles)

    StarImage = MidImage[:image.shape[0],:shape[0],:shape[1]]
    SmallDistance = MaxProjectDist(SmallDistance, axis=-1)
    Probability = np.zeros([SmallProbability.shape[0] * grid[0],SmallProbability.shape[1] * grid[1], SmallProbability.shape[2] * grid[2] ])
    Distance = np.zeros([SmallDistance.shape[0] * grid[0], SmallDistance.shape[1] * grid[1], SmallDistance.shape[2] * grid[2] ])
    #We only allow for the grid parameter to be 1 along the Z axis
    for i in range(0, SmallProbability.shape[0]):
        Probability[i,:] = cv2.resize(SmallProbability[i,:], dsize=(SmallProbability.shape[2] * grid[2] , SmallProbability.shape[1] * grid[1] ))
        Distance[i,:] = cv2.resize(SmallDistance[i,:], dsize=(SmallDistance.shape[2] * grid[2] , SmallDistance.shape[1] * grid[1] ))
    
    if UseProbability:
        
        MaxProjectDistance = Probability[:image.shape[0],:shape[0],:shape[1]]

    else:
        
        MaxProjectDistance = Distance[:image.shape[0],:shape[0],:shape[1]]

    if MaskImage is not None:
        
       if smartcorrection is None: 
          
          Watershed, Markers = WatershedwithMask3D(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), MaskImage.astype('uint16'), grid )
          Watershed = fill_label_holes(Watershed.astype('uint16'))
    
       if smartcorrection is not None:
           
          Watershed, Markers = WatershedSmartCorrection3D(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), MaskImage.astype('uint16'), grid, smartcorrection = smartcorrection )
          Watershed = fill_label_holes(Watershed.astype('uint16'))

    if MaskImage is None:

       Watershed, Markers = WatershedNOMask3D(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), grid)
       

    return Watershed, Markers, StarImage  
 
def STARPrediction(image, model, min_size, n_tiles, MaskImage = None, smartcorrection = None, UseProbability = True):
    
    
    image = normalize(image, 1, 99.8, axis = (0,1))
    shape = [image.shape[0], image.shape[1]]
    image = zero_pad(image, 64, 64)
    
    MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
    
    StarImage = MidImage[:shape[0],:shape[1]]
    
    SmallProbability, SmallDistance = model.predict(image, n_tiles = n_tiles)
    grid = model.config.grid
    Probability = cv2.resize(SmallProbability, dsize=(SmallProbability.shape[1] * grid[1] , SmallProbability.shape[0] * grid[0] ))
    Distance = MaxProjectDist(SmallDistance, axis=-1)
    Distance = cv2.resize(Distance, dsize=(Distance.shape[1] * grid[1] , Distance.shape[0] * grid[0] ))
    if UseProbability:
        
        MaxProjectDistance = Probability[:shape[0],:shape[1]]

    else:
        
        MaxProjectDistance = Distance[:shape[0],:shape[1]]

    if MaskImage is not None:
        
       if smartcorrection is None: 
          
          Watershed, Markers = WatershedwithMask(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), MaskImage.astype('uint16'), grid)
          Watershed = fill_label_holes(Watershed.astype('uint16'))
    
       if smartcorrection is not None:
           
          Watershed, Markers = WatershedSmartCorrection(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), MaskImage.astype('uint16'), grid, smartcorrection = smartcorrection)
          Watershed = fill_label_holes(Watershed.astype('uint16'))   

    if MaskImage is None:

         Watershed, Markers = WatershedNOMask(MaxProjectDistance.astype('uint16'), StarImage.astype('uint16'), grid)
         Watershed = fill_label_holes(Watershed.astype('uint16'))

    return Watershed, Markers, StarImage     


def SimplePrediction(X, UnetModel, StarModel, n_tiles = (2,2), UseProbability = True, min_size = 20):
    
                Yhat_val = []
    
                for x in X:
                      
                      Mask = UNETPrediction(x, UnetModel, min_size, n_tiles, 'YX')
                      
                      SmartSeeds, _, StarImage = STARPrediction(x, StarModel, min_size, n_tiles, MaskImage = Mask, smartcorrection = None, UseProbability = UseProbability)
                      
                      Yhat_val.append(SmartSeeds.astype('uint16'))
                     
                Yhat_val = np.asarray(Yhat_val)
                
                return Yhat_val


def SmartSeedPrediction2D(ImageDir, SaveDir,fname,  UnetModel, StarModel, NoiseModel = None,min_size_mask =100, min_size = 100, DownsampleFactor = 1, n_tiles = (2,2), doMask = True, 
                          smartcorrection = None,  UseProbability = True, filtersize = 0):
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    DenoisedResults = SaveDir + 'Denoised/'
    StarImageResults = SaveDir + 'StarDistMask/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    Path(SaveDir).mkdir(exist_ok = True)

    if StarModel is not None:
       Path(SmartSeedsResults).mkdir(exist_ok = True)
       Path(StarImageResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    
    #Read Image
    image = imread(fname)
    Name = os.path.basename(os.path.splitext(fname)[0])
    image = DownsampleData2D(image, DownsampleFactor)
    
    if NoiseModel is not None:
        
        print('Denoising Image')
        image = NoiseModel.predict(image,'YX', n_tiles = n_tiles)
    
    if doMask:
          Mask = UNETPrediction(gaussian_filter(image, filtersize), UnetModel, min_size_mask, n_tiles, 'YX')
    
          if StarModel is not None:
             SmartSeeds, _, StarImage = STARPrediction(gaussian_filter(image, filtersize), StarModel, min_size, n_tiles, MaskImage = Mask, smartcorrection = smartcorrection, UseProbability = UseProbability)
             #Upsample downsampled results
             image = DownsampleData2D(image, 1.0/DownsampleFactor)
             Mask = DownsampleData2D(Mask, 1.0/DownsampleFactor)
             SmartSeeds = DownsampleData2D(SmartSeeds, 1.0/DownsampleFactor)
             StarImage = DownsampleData2D(StarImage, 1.0/DownsampleFactor)
             multiplot(image, Mask, SmartSeeds, "Image", "UNET", "SmartSeeds")  
             
    if doMask == False:
        
        Mask = UNETPrediction(image, UnetModel, min_size_mask, n_tiles, 'YX')
        if StarModel is not None:     
             SmartSeeds, _, StarImage = STARPrediction(gaussian_filter(image, filtersize), StarModel, min_size, n_tiles)
             #Upsample downsampled results
             image = DownsampleData2D(image, 1.0/DownsampleFactor)
             Mask = DownsampleData2D(Mask, 1.0/DownsampleFactor)
             SmartSeeds = DownsampleData2D(SmartSeeds, 1.0/DownsampleFactor)
             StarImage = DownsampleData2D(StarImage, 1.0/DownsampleFactor)
             multiplot(image, Mask, SmartSeeds, "Image", "UNET", "SmartSeeds")  
    
    if StarModel is None:
            
             #Upsample downsampled results
             image = DownsampleData2D(image, 1.0/DownsampleFactor)
             Mask = DownsampleData2D(Mask, 1.0/DownsampleFactor) 
             doubleplot(image, Mask, "Image", "UNET")  
           
    if NoiseModel is not None:
                Path(DenoisedResults).mkdir(exist_ok = True)
                imwrite((DenoisedResults + Name + '.tif' ) , image.astype('float32')) 
         
    if StarModel is not None:  
       imwrite((SmartSeedsResults + Name+ '.tif' ) , SmartSeeds.astype('uint16'))
       imwrite((StarImageResults + Name+ '.tif' ) , StarImage.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , Mask.astype('uint16'))   
    
 
    return SmartSeeds, Mask
    
    
    
def SmartSeedProject(ImageDir, SaveDir,fname, UnetModel, StarModel, NoiseModel = None, min_size_mask = 100, min_size = 100,start = 0, end = 10, DownsampleFactor = 1, n_tiles = (2,2), doMask = True, 
                     smartcorrection = None, UseProbability = True, filtersize = 0):
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    DenoisedResults = SaveDir + 'Denoised/'
    StarImageResults = SaveDir + 'StarDistMask/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    Path(SaveDir).mkdir(exist_ok = True)
        
    

    if StarModel is not None:
       Path(SmartSeedsResults).mkdir(exist_ok = True)
       Path(StarImageResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    
    #Read Image
    image = imread(fname)
    Project = np.sum(image[start:end,:], axis = 0)
        
    Name = os.path.basename(os.path.splitext(fname)[0])
    
    Project = DownsampleData2D(Project, DownsampleFactor)
    
    if NoiseModel is not None:
        
        print('Denoising Image')
        Project = NoiseModel.predict(Project,'YX', n_tiles = n_tiles)
        
    if doMask:
        
          Mask = UNETPrediction(gaussian_filter(Project, filtersize), UnetModel, min_size, n_tiles, 'YX')
    
          if StarModel is not None:
             SmartSeeds, _, StarImage = STARPrediction(gaussian_filter(Project, filtersize), StarModel, min_size, n_tiles, MaskImage = Mask, smartcorrection = smartcorrection, UseProbability= UseProbability)
             
             Project = DownsampleData2D(Project, 1.0/DownsampleFactor)
             Mask = DownsampleData2D(Mask, 1.0/DownsampleFactor)
             SmartSeeds = DownsampleData2D(SmartSeeds, 1.0/DownsampleFactor)
             StarImage = DownsampleData2D(StarImage, 1.0/DownsampleFactor)
             
             multiplot(image, Mask, SmartSeeds, "Image", "UNET", "SmartSeeds")  
    if doMask == False:
        
        Mask = UNETPrediction(gaussian_filter(Project, filtersize), UnetModel, min_size_mask, n_tiles, 'YX')
        if StarModel is not None:     
             SmartSeeds, _, StarImage = STARPrediction(gaussian_filter(image, filtersize), StarModel, min_size, n_tiles, UseProbability = UseProbability)
             
             Project = DownsampleData2D(Project, 1.0/DownsampleFactor)
             Mask = DownsampleData2D(Mask, 1.0/DownsampleFactor)
             SmartSeeds = DownsampleData2D(SmartSeeds, 1.0/DownsampleFactor)
             StarImage = DownsampleData2D(StarImage, 1.0/DownsampleFactor)
             
             multiplot(Project, Mask, SmartSeeds, "Image", "UNET", "SmartSeeds")  
    
        if StarModel is None:
             Project = DownsampleData2D(Project, 1.0/DownsampleFactor)
             Mask = DownsampleData2D(Mask, 1.0/DownsampleFactor)
             
             doubleplot(Project, Mask, "Image", "UNET")  
  
        
    if NoiseModel is not None:
                Path(DenoisedResults).mkdir(exist_ok = True)
                imwrite((DenoisedResults + Name + '.tif' ) , Project.astype('float32'))  
         
    if StarModel is not None:  
       imwrite((SmartSeedsResults + Name+ '.tif' ) , SmartSeeds.astype('uint16'))
       imwrite((StarImageResults + Name+ '.tif' ) , StarImage.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , Mask.astype('uint16'))      
    
 
    return SmartSeeds, Mask
    
    
    
def SmartSeedPrediction3DMO2D(ImageDir, SaveDir, fname, UnetModel, StarModel, NoiseModel = None,min_size_mask = 100, min_size = 100, 
                              DownsampleFactor = 1, n_tiles = (2,2), doMask = True, smartcorrection = None, threshold = 20, UseProbability = True, filtersize = 0):
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    DenoisedResults = SaveDir + 'Denoised/'
    StarImageResults = SaveDir + 'StarDistMask/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    Path(SaveDir).mkdir(exist_ok = True)
    if StarModel is not None:
       Path(SmartSeedsResults).mkdir(exist_ok = True)
       Path(StarImageResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    #Read Image
    Name = os.path.basename(os.path.splitext(fname)[0])
    image = imread(fname)
    image = DownsampleData(image, DownsampleFactor)
    if NoiseModel is not None:
                
                print('Denoising Image')
                image = NoiseModel.predict(image,'ZYX')
                
               
    TimeSmartSeeds = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype='uint16')
    TimeStarImage = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype = 'uint16')
    TimeDownSample = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype = 'float32')
    #UNET Mask prediction in 3D
    TimeMask = UNETPrediction(image, UnetModel, min_size_mask, (1,n_tiles[0],n_tiles[1]), 'ZYX') 
    for i in range(0, image.shape[0]):
        
            smallimage = image[i,:]
            TimeDownSample[i,:] = smallimage
            if doMask:
            
                  if StarModel is not None:
                     SmartSeeds, _ , StarImage = STARPrediction(gaussian_filter(smallimage, filtersize), StarModel, min_size, n_tiles, MaskImage = TimeMask[i,:], smartcorrection = smartcorrection, UseProbability = UseProbability)
                     TimeSmartSeeds[i,:] = SmartSeeds 
                     TimeStarImage[i,:] = StarImage     
         
            if doMask == False:
                
                if StarModel is not None:
                    SmartSeeds, _, StarImage = STARPrediction(gaussian_filter(smallimage, filtersize), StarModel, min_size, n_tiles, UseProbability = UseProbability)
                    TimeSmartSeeds[i,:] = SmartSeeds 
                    TimeStarImage[i,:] = StarImage
        
            multiplot(smallimage, TimeMask[i,:], SmartSeeds, "Image", "UNET", "SmartSeeds")  
            
    image = DownsampleData(image, 1.0/DownsampleFactor) 
    if NoiseModel is not None:
                
                Path(DenoisedResults).mkdir(exist_ok = True)
                imwrite((DenoisedResults + Name + '.tif' ) , image.astype('float32'))        
    
    TimeMask = DownsampleData(TimeMask, 1.0/DownsampleFactor)  
    if StarModel:
           TimeSmartSeeds = DownsampleData(TimeSmartSeeds, 1.0/DownsampleFactor)  
           TimeStarImage = DownsampleData(TimeStarImage, 1.0/DownsampleFactor) 
    TimeSmartSeeds = relabel_sequential(TimeSmartSeeds)
    TimeStarImage  = relabel_sequential(TimeStarImage)
    TimeMask = relabel_sequential(TimeMask)
    TimeSmartSeeds = merge_labels_across_volume(TimeSmartSeeds.astype('uint16'), RelabelZ, threshold= threshold)    
    TimeStarImage = merge_labels_across_volume(TimeStarImage.astype('uint16'), RelabelZ, threshold= threshold)
    TimeMask = merge_labels_across_volume(TimeMask.astype('uint16'), RelabelZ, threshold= threshold)  
      
    if StarModel is not None:  
        imwrite((SmartSeedsResults + Name+ '.tif' ) , TimeSmartSeeds.astype('uint16'))
        imwrite((StarImageResults + Name+ '.tif' ) , TimeStarImage.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , TimeMask.astype('uint16'))       
    
    return TimeSmartSeeds, TimeMask
    
        

def SmartSeedPredictionSliced(ImageDir, SaveDir, fname, UnetModel, StarModel, NoiseModel = None, min_size_mask = 100, min_size = 100, 
                              DownsampleFactor = 1, n_tiles = (2,2), doMask = True, smartcorrection = None, threshold = 20, masklinkthreshold = 100, UseProbability = True, filtersize = 0):
    
    
    print('Generating SmartSeed results')
    UNETResults = SaveDir + 'BinaryMask/'
    DenoisedResults = SaveDir + 'Denoised/'
    StarImageResults = SaveDir + 'StarDistMask/'
    SmartSeedsResults = SaveDir + 'SmartSeedsMask/' 
    Path(SaveDir).mkdir(exist_ok = True)
    if StarModel is not None:
       Path(SmartSeedsResults).mkdir(exist_ok = True)
       Path(StarImageResults).mkdir(exist_ok = True)
    Path(UNETResults).mkdir(exist_ok = True)
    #Read Image
    Name = os.path.basename(os.path.splitext(fname)[0])
    image = imread(fname)
    
    image = DownsampleData(image, DownsampleFactor)
    
    if NoiseModel is not None:
                
                print('Denoising Image')
                image = NoiseModel.predict(image,'ZYX')
                
    TimeSmartSeeds = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype='uint16')
    TimeStarImage = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype='uint16')
    TimeMask = np.zeros([image.shape[0], image.shape[1], image.shape[2]], dtype = 'uint16')
    
    
    for i in range(0, image.shape[0]):
        
            smallimage = image[i,:]
            Mask = UNETPrediction(gaussian_filter(smallimage, filtersize), UnetModel, min_size_mask, n_tiles, 'YX')
            TimeMask[i,:] = Mask
            if doMask:
                
                  if StarModel is not None:
                     SmartSeeds, _, StarImage = STARPrediction(gaussian_filter(smallimage, filtersize), StarModel, min_size, n_tiles, MaskImage = Mask, smartcorrection = smartcorrection, UseProbability = UseProbability)
                     TimeSmartSeeds[i,:] = SmartSeeds
                     TimeStarImage[i,:] = StarImage
                     multiplot(smallimage, Mask, SmartSeeds, "Image", "UNET", "SmartSeeds")  

                           
         
            if doMask == False:
                
                if StarModel is not None:
                    SmartSeeds, _, StarImage = STARPrediction(gaussian_filter(smallimage, filtersize), StarModel, min_size, n_tiles, UseProbability = UseProbability)
                    TimeSmartSeeds[i,:] = SmartSeeds
                    TimeStarImage[i,:] = StarImage
        
                    multiplot(smallimage, Mask, SmartSeeds, "Image", "UNET", "SmartSeeds")  
            doubleplot(smallimage, Mask, "Image", "UNET")  

    if NoiseModel is not None:
                image = DownsampleData(image, 1.0/DownsampleFactor) 
                Path(DenoisedResults).mkdir(exist_ok = True)
                imwrite((DenoisedResults + Name + '.tif' ) , image.astype('float32'))       
    TimeMask = DownsampleData(TimeMask, 1.0/DownsampleFactor)
    
    if StarModel is not None: 
        TimeSmartSeeds = DownsampleData(TimeSmartSeeds, 1.0/DownsampleFactor)
        TimeStarImage = DownsampleData(TimeStarImage, 1.0/DownsampleFactor)
        TimeSmartSeeds = relabel_sequential(TimeSmartSeeds)[0]
        TimeStarImage = relabel_sequential(TimeStarImage)[0]
        TimeSmartSeeds = merge_labels_across_volume(TimeSmartSeeds.astype('uint16'), RelabelZ, threshold = threshold)    
        TimeStarImage = merge_labels_across_volume(TimeStarImage.astype('uint16'), RelabelZ, threshold = threshold)  
    TimeMask = relabel_sequential(TimeMask)[0]    
    TimeMask = merge_labels_across_volume(TimeMask.astype('uint16'), RelabelZ, threshold = masklinkthreshold)  
      
    if StarModel is not None:  
        imwrite((SmartSeedsResults + Name+ '.tif' ) , TimeSmartSeeds.astype('uint16'))
        imwrite((StarImageResults + Name+ '.tif' ) , TimeStarImage.astype('uint16'))
    imwrite((UNETResults + Name+ '.tif' ) , TimeMask.astype('uint16'))       
    
    return TimeSmartSeeds, TimeMask         
    
                
def BTracker(config, objects, search_radius = 100, ndim = 4):
  # initialise a tracker session using a context manager
  with btrack.BayesianTracker() as tracker:

          # configure the tracker using a config file
          tracker.configure_from_file(config)
          tracker.append(objects)
          tracker.update_method = BayesianUpdates.EXACT
        
          if search_radius is not None:
              tracker.max_search_radius = search_radius        
          # track them (in interactive mode)
          tracker.track_interactive(step_size=100)
          tracker.optimize(options={'tm_lim': int(6e4)})
        
          # get the tracks in a format for napari visualization
          data, properties, graph = tracker.to_napari(ndim=ndim)
    
  return data, properties, graph  


def Tracker(Label, Image, AllTrees):
    
    
    for i in range(0, Label.shape[0] - 1):
        
        # Get all indices in the current frame
        currentimage = Label[i,:].astype('uint16')
        nextimage = Label[i + 1,:].astype('uint16')
        waterproperties = measure.regionprops(currentimage, currentimage)
        centroids = [prop.centroid for prop in waterproperties]
        # Search in the tree for next frame for the nearest point
        
        tree, nextindices = AllTrees[str(i + 1)]
        for currlocation in centroids:
            
            distance, location = tree.query(currlocation)
            if distance < 50:
              currentlabel = currentimage[int(currlocation[0]), int(currlocation[1]), int(currlocation[2])]
              if currentlabel > 0:
                 nextlabel = nextimage[int(nextindices[location][0]), int(nextindices[location][1]), int(nextindices[location][2])]
                 print('Time', i , 'Nearest neighbour of curent label', currentlabel, 'current location', int(currlocation[0]), int(currlocation[1]), int(currlocation[2]), 'next label', nextlabel, 'next location', (int(nextindices[location][0]), int(nextindices[location][1]), int(nextindices[location][2])))
              
              
def NapariText(viewer, segimage, label, Sphericity, Volume, Area, Coordinates, Intensity, z, shapes_layer):
    

    warnings.filterwarnings("ignore")
    segimage = segimage.astype('uint16')

    properties = {}
    properties['Label'] = [label, label]
    properties['Sphericity'] = [Sphericity, Sphericity]
    properties['Volume'] = [Volume, Volume]
    properties['Area'] = [Area, Area]
    properties['Intensity'] = [Intensity, Intensity]
    Cord = [Coordinates[1], Coordinates[2]]
    Shape = np.array([Cord, Cord])
    text_parameters = {
        
        'text': 'Label: {Label} \nSphericity{Sphericity: .2f} \nVolume{Volume: .2f}  \nArea{Area: .2f}  \nIntensity{Intensity: .2f} ',
        'size': 12,
        'color': 'green',
        'anchor': 'upper_left',
        'translation': [3, 0],
        }
   
    shapes_layer = viewer.add_shapes(
        Shape,
        properties=properties,
        text=text_parameters,
        name='Tomato 3D properties',
    )
    
    return shapes_layer
    
            
          
def XYZImageResize(ImageDir,SegmentationDir,NameDifference,CsvFile, DownsampleFactor, XYCalibration):

    
    Raw_path = os.path.join(ImageDir, '*tif')
    filesRaw = glob.glob(Raw_path)
    filesRaw.sort
    
    Seg_path = os.path.join(SegmentationDir, '*tif')
    filesSeg = glob.glob(Seg_path)
    filesSeg.sort
    
    maxY = 0
    maxX = 0
    maxZ = 0
    for fname in filesRaw:
        
        image = imread(fname)
        if image.shape[1] > maxZ:
            maxZ = image.shape[0]
        if image.shape[1] > maxY:
            maxY = image.shape[1]
        if image.shape[2] > maxX:
            maxX = image.shape[2]
            
    
    AllImages = []
    AllSegImages = []
    DisplayName = []


    for fname in filesRaw:
        Resizeimages = np.zeros([maxZ, maxY, maxX])
        Resizesegimages = np.zeros([maxZ, maxY, maxX])
        image = imread(fname)
        Name = os.path.basename(os.path.splitext(fname)[0])
        for secondfname in filesSeg:
               SecondName = os.path.basename(os.path.splitext(secondfname)[0]) 
               if SecondName == NameDifference + Name:
                        segimages = imread(secondfname)
                        
                        #AllProperties  = TomatoAnalyzer(image, segimages)
                        DisplayName.append(Name)
                        
                        
                        #for i in range(0, len(AllProperties)):
                                      #Properties = AllProperties[i]
                                      #AllName.append(Name)
                                      #AllLabel.append(Properties[0])
                                      #AllSphericity.append(Properties[1])
                                      #AllVolume.append(Properties[2] * DownsampleFactor * DownsampleFactor * XYCalibration)
                                      #AllArea.append(Properties[3] * DownsampleFactor * DownsampleFactor * XYCalibration)
                                      #AllCentroid.append(Properties[4])
                                      #AllIntensity.append(Properties[5])
                        Resizeimages[:image.shape[0],:image.shape[1],:image.shape[2]] = image
                        Resizesegimages[:segimages.shape[0],:segimages.shape[1],:segimages.shape[2]] = segimages
                        AllImages.append(Resizeimages)
                        AllSegImages.append(Resizesegimages)
    AllImages = np.array(AllImages)        
    AllSegImages = np.array(AllSegImages)
    AllLocations = ReturnLocalizations(AllImages, AllSegImages)
    #AllLabel = np.array(AllLabel)
    #AllSphericity = np.array(AllSphericity)
    #AllVolume = np.array(AllVolume)
    #AllArea = np.array(AllArea)
    #AllCentroid = np.array(AllCentroid)
    #AllIntensity = np.array(AllIntensity)
    
    #df = pd.DataFrame(list(zip(AllName, AllLabel, AllSphericity, AllVolume, AllArea, AllCentroid, AllIntensity)), 
                                              #columns =['Name', 'Label', 'Sphericity', 'Volume', 'Area', 'Centroid', 'Intensity'])

    #df.to_csv(CsvFile)  
    #df  
    
    return AllImages, AllSegImages,  DisplayName, AllLocations  #, AllName, AllLabel, AllSphericity, AllVolume, AllArea, AllCentroid, AllIntensity


def VetoRegions(Image, Zratio = 3):
    
    Image = Image.astype('uint16')
    
    properties = measure.regionprops(Image, Image)
    
    for prop in properties:
        
        LabelImage = prop.image
        if LabelImage.shape[0] < Image.shape[0]/Zratio :
            indices = zip(*np.where(LabelImage > 0))
            for z, y, x in indices:

                 Image[z,y,x] = 0

    return Image
    


def TomatoAnalyzer(Image, Label):
    
    Label = Label.astype('uint16')
    
    TomatoProperties = {}
    
    properties = measure.regionprops(Label, Image)
    AllTomatoProperties = []
    for prop in properties:
        
        currentlabel = prop.label
        Intensity = prop.mean_intensity
        LabelImage = prop.image
        Centroid = prop.centroid
        if LabelImage.shape[0] >= Image.shape[0]/3 :
            Sphere, Volume, Area = Sphericity(LabelImage)

            AllTomatoProperties.append([currentlabel,Sphere,Volume,Area,Centroid,Intensity])       

    return AllTomatoProperties

def Sphericity(LabelImage):
    verts, faces, _, _ = measure.marching_cubes(LabelImage)
    Volume = np.sum(LabelImage > 0)
    Area = measure.mesh_surface_area(verts, faces)
    #Coordinates = 
    Sphere = ( 3.14 ** (1.0/3.0) * (6 * Volume ) ** (2.0 / 3.0) )/Area

    return Sphere, Volume, Area
   

def WatershedNOMask(Image, Label, grid):
    
    
    properties = measure.regionprops(Label, Image)
    Coordinates = [prop.centroid for prop in properties]
    
    Coordinates = sorted(Coordinates , key=lambda k: [k[1], k[0]])
    Coordinates.append((0,0))
    Coordinates = np.asarray(Coordinates) 

    
    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)  
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))

    markers = morphology.dilation(markers_raw, morphology.disk(2))
    
    #Image = sobel(Image)
    watershedImage = watershed(-Image, markers)
    return watershedImage, markers
    

#Default method that works well with cells which are below a certain shape and do not have weak edges
    
def WatershedwithMask(Image, Label,mask, grid):
    
    
   
    properties = measure.regionprops(Label, Image)
    Coordinates = [prop.centroid for prop in properties]
    
    Coordinates = sorted(Coordinates , key=lambda k: [k[1], k[0]])
    Coordinates.append((0,0))
    Coordinates = np.asarray(Coordinates)
    
    

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)  
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
    
    markers = morphology.dilation(markers_raw, morphology.disk(2))
    #Image = sobel(Image)
    watershedImage = watershed(-Image, markers, mask = mask.copy())
    
    return watershedImage, markers  

#Default method that works well with cells which are below a certain shape and do not have weak edges
    
def iou3D(boxA, centroid):
    
    ndim = len(centroid)
    inside = False
    
    Condition = [Conditioncheck(centroid, boxA, p, ndim) for p in range(0,ndim)]
        
    inside = all(Condition)
    
    return inside

def Conditioncheck(centroid, boxA, p, ndim):
    
      condition = False
    
      if centroid[p] >= boxA[p] and centroid[p] <= boxA[p + ndim]:
          
           condition = True
           
      return condition     
    
    

def WatershedwithMask3D(Image, Label,mask, grid): 
    properties = measure.regionprops(Label, Image) 
    binaryproperties = measure.regionprops(label(mask), Image) 
    
    
    Coordinates = [prop.centroid for prop in properties] 
    BinaryCoordinates = [prop.centroid for prop in binaryproperties]
    
    Binarybbox = [prop.bbox for prop in binaryproperties]
    Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1], k[2]]) 
    
    if len(Binarybbox) > 0:    
            for i in range(0, len(Binarybbox)):
                
                box = Binarybbox[i]
                inside = [iou3D(box, star) for star in Coordinates]
                
                if not any(inside) :
                         Coordinates.append(BinaryCoordinates[i])    
                         
    
    Coordinates.append((0,0,0))
    #tree = spatial.cKDTree(Coordinates)

    
    #BinaryCoordinates = sorted(BinaryCoordinates , key=lambda k: [k[0], k[1], k[2]]) 
    #if len(BinaryCoordinates) > 0:
        #for i in range(0,len(BinaryCoordinates)):
             #index = BinaryCoordinates[i]
             #distance, point = tree.query(index)
             #if distance > :
                  #Coordinates.append(index)

    Coordinates = np.asarray(Coordinates)
    coordinates_int = np.round(Coordinates).astype(int) 
    
    markers_raw = np.zeros_like(Image) 
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates)) 
    markers = morphology.dilation(markers_raw.astype('uint16'), morphology.ball(2))


    watershedImage = watershed(-Image, markers, mask = mask.copy()) #watershedImage[mask == 0] = 0 
    return watershedImage, markers

    
def WatershedNOMask3D(Image, Label, grid):
    
    
   
    properties = measure.regionprops(Label, Image)
    Coordinates = [prop.centroid for prop in properties]
    
    Coordinates = sorted(Coordinates , key=lambda k: [k[0], k[1], k[2]])
    Coordinates.append((0,0,0))
    Coordinates = np.asarray(Coordinates)
    
    

    coordinates_int = np.round(Coordinates).astype(int)
    markers_raw = np.zeros_like(Image)  
    markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
    
    markers = morphology.dilation(markers_raw.astype('uint16'), morphology.ball(2))
    #for i in range(0, Image.shape[0]):
       #Image[i,:] = sobel(Image[i,:] )
    watershedImage = watershed(-Image, markers)
   
    return watershedImage, markers


# To be used for BIG cells like mouse embryos to get proper boundary reconstruction. This method used the distance map for doing the watershedding and then
# does the smart correction over the pixels specified to color the mask with the color of the closest label. Very useful for weak edges
    
def WatershedSmartCorrection(Image, Label, mask, grid, smartcorrection = 20, max_size = 100000):
    
    
   
    CopyDist = Image.copy()
    thresh = threshold_otsu(CopyDist)
    CopyDist = CopyDist > thresh


    ## Use markers from Label image
    Labelproperties = measure.regionprops(Label, Image)
    LabelCoordinates = [prop.centroid for prop in Labelproperties] 
    LabelCoordinates.append((0,0))
    LabelCoordinates = sorted(LabelCoordinates , key=lambda k: [k[1], k[0]])
    LabelCoordinates = np.asarray(LabelCoordinates)
    sexyImage = np.zeros_like(Image)
    Labelcoordinates_int = np.round(LabelCoordinates).astype(int)
    
    Labelmarkers_raw = np.zeros([Image.shape[0], Image.shape[1]]) 
    if(len(LabelCoordinates) > 0) :
     Labelmarkers_raw[tuple(Labelcoordinates_int.T)] = 1 + np.arange(len(LabelCoordinates))
     Labelmarkers = morphology.dilation(Labelmarkers_raw, morphology.disk(5))
  

   
    Image = sobel(Image)


    watershedImage = watershed(Image, markers = Labelmarkers)
    
    watershedImage[thin(CopyDist, max_iter = smartcorrection//2) == 0] = 0
    sexyImage = watershedImage
    copymask = mask.copy()
    
    Binary = watershedImage > 0
   
    if smartcorrection > 0:
       indices = list(zip(*np.where(Binary>0)))
       if(len(indices) > 0):
        indices = np.asarray(indices)
        tree = spatial.cKDTree(indices)
        copymask = copymask - Binary
        maskindices = list(zip(*((np.where(copymask>0)))))
        maskindices = np.asarray(maskindices)
    
        for i in (range(0,maskindices.shape[0])):
    
           pt = maskindices[i]
           closest =  tree.query(pt)
        
           if closest[0] < smartcorrection:
               sexyImage[pt[0], pt[1]] = watershedImage[indices[closest[1]][0], indices[closest[1]][1]]  
       
    sexyImage = fill_label_holes(sexyImage)
    sexyImage, forward_map, inverse_map = relabel_sequential(sexyImage)
    
    
    return sexyImage, Labelmarkers  


def WatershedSmartCorrection3D(Image, Label, mask, grid, smartcorrection = 20, max_size = 100000):
    
    
   
    CopyDist = Image.copy()
    thresh = threshold_otsu(CopyDist)
    CopyDist = CopyDist > thresh
    ThinCopyDist = np.zeros([CopyDist.shape[0],CopyDist.shape[1],CopyDist.shape[2]])
    for i in range(0, CopyDist.shape[0]):
       ThinCopyDist[i,:] = thin(CopyDist[i,:] , max_iter = smartcorrection//4)
  
    ThinCopyDist = CCLabels(ThinCopyDist)


    ## Use markers from Label image
    Labelproperties = measure.regionprops(Label, Image)
    LabelCoordinates = [prop.centroid for prop in Labelproperties] 
    LabelCoordinates.append((0,0,0))
    LabelCoordinates = sorted(LabelCoordinates , key=lambda k: [k[1], k[0], k[2]])
    LabelCoordinates = np.asarray(LabelCoordinates)
    sexyImage = np.zeros_like(Image)
    Labelcoordinates_int = np.round(LabelCoordinates).astype(int)
    
    Labelmarkers_raw = np.zeros([Image.shape[0], Image.shape[1],Image.shape[2] ]) 
    if(len(LabelCoordinates) > 0) :
     Labelmarkers_raw[tuple(Labelcoordinates_int.T)] = 1 + np.arange(len(LabelCoordinates))
     
     Labelmarkers = morphology.dilation(Labelmarkers_raw.astype('uint16'), morphology.ball(5))
  

   
    for i in range(0, Image.shape[0]):
        Image[i,:] = sobel(Image[i,:].astype('uint16'))


    watershedImage = watershed(Image, markers = Labelmarkers)

    TestCopyDist = np.zeros([CopyDist.shape[0],CopyDist.shape[1],CopyDist.shape[2]])
    for i in range(0, CopyDist.shape[0]):
       TestCopyDist[i,:] = thin(CopyDist[i,:] , max_iter = smartcorrection//2)

    watershedImage[TestCopyDist == 0] = 0
    sexyImage = watershedImage
    copymask = mask.copy()
    
    Binary = watershedImage > 0
   
    if smartcorrection > 0:
       indices = list(zip(*np.where(Binary>0)))
       if(len(indices) > 0):
        indices = np.asarray(indices)
        tree = spatial.cKDTree(indices)
        copymask = copymask - Binary
        maskindices = list(zip(*((np.where(copymask>0)))))
        maskindices = np.asarray(maskindices)
    
        for i in (range(0,maskindices.shape[0])):
    
           pt = maskindices[i]
           closest =  tree.query(pt)
        
           if closest[0] < smartcorrection:
               sexyImage[pt[0], pt[1]] = watershedImage[indices[closest[1]][0], indices[closest[1]][1]]  
       
    sexyImage = fill_label_holes(sexyImage)
    sexyImage, forward_map, inverse_map = relabel_sequential(sexyImage)
    
    
    return sexyImage, Labelmarkers  


def WatershedDistanceMarker(Image, Label, mask, grid, smartcorrection, max_size = 100000):
    
    
   
    CopyDist = Image.copy()
    thresh = threshold_otsu(CopyDist)
    CopyDist = CopyDist > thresh
    ThinCopyDist = thin(CopyDist, max_iter = 5)
  
    ThinCopyDist = CCLabels(ThinCopyDist)
    
    ## Use markers from distance map
    properties = measure.regionprops(ThinCopyDist, Image)
    Coordinates = [prop.centroid for prop in properties] 
    Coordinates.append((0,0))
    Coordinates = sorted(Coordinates , key=lambda k: [k[1], k[0]])
    Coordinates = np.asarray(Coordinates)
    sexyImage = np.zeros_like(Image)
    coordinates_int = np.round(Coordinates).astype(int)
    
    markers_raw = np.zeros([Image.shape[0], Image.shape[1]]) 
    if(len(Coordinates) > 0) :
     markers_raw[tuple(coordinates_int.T)] = 1 + np.arange(len(Coordinates))
     markers = morphology.dilation(markers_raw, morphology.disk(2))
   
    Image = sobel(Image)


    watershedImage = watershed(Image, markers = markers)
    watershedImage[thin(CopyDist, max_iter = 2) == 0] = 0
    
    sexyImage = watershedImage
    
    sexyImage = fill_label_holes(sexyImage)
    sexyImage, forward_map, inverse_map = relabel_sequential(sexyImage)
   
    
    return sexyImage, markers 

                 


    
def Integer_to_border(Label, max_size = 6400):

        SmallLabel = remove_big_objects(Label, max_size = max_size)
        BoundaryLabel =  find_boundaries(SmallLabel, mode='outer')
           
        Binary = BoundaryLabel > 0
        
        return Binary
        
def zero_pad(image, PadX, PadY):

          sizeY = image.shape[1]
          sizeX = image.shape[0]
          
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%PadX!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%PadY!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([sizeXextend, sizeYextend])
          
          extendimage[0:sizeX, 0:sizeY] = image
              
              
          return extendimage 
    
        
def zero_pad_color(image, PadX, PadY):

          sizeY = image.shape[1]
          sizeX = image.shape[0]
          color = image.shape[2]  
          
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%PadX!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%PadY!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([sizeXextend, sizeYextend, color])
          
          extendimage[0:sizeX, 0:sizeY, 0:color] = image
              
              
          return extendimage      
    
def zero_pad_time(image, PadX, PadY):

          sizeY = image.shape[2]
          sizeX = image.shape[1]
          
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%PadX!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%PadY!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([image.shape[0], sizeXextend, sizeYextend])
          
          extendimage[:,0:sizeX, 0:sizeY] = image
              
              
          return extendimage   
      
def BackGroundCorrection2D(Image, sigma):
    
    
     Blur = gaussian(Image.astype(float), sigma)
     
     
     Corrected = Image - Blur
     
     return Corrected  
 
def OtsuThreshold2D(Image, size = 10):
    
    
    adaptive_thresh = threshold_otsu(Image)
    Binary  = Image > adaptive_thresh
    Clean =  remove_small_objects(Binary, min_size=size, connectivity=4, in_place=False)

    return Clean

def SeedStarDistMaskOZ(Image, Label, grid, max_size = 100000, min_size = 1000):
    
    
    Image = Image > 0
    Image = binary_fill_holes(Image)
    Image= binary_erosion(Image,iterations = 10)
    
    

    return Image             

def MaxProjectDist(Image, axis = -1):
    
    MaxProject = np.amax(Image, axis = axis)
        
    return MaxProject

def MidProjectDist(Image, axis = -1, slices = 1):
    
    assert len(Image.shape) >=3
    SmallImage = Image.take(indices = range(Image.shape[axis]//2 - slices, Image.shape[axis]//2 + slices), axis = axis)
    
    MaxProject = np.amax(SmallImage, axis = axis)
    return MaxProject


def multiplot(imageA, imageB, imageC, titleA, titleB, titleC, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.gray)
    ax[0].set_title(titleA)
    ax[0].set_axis_off()
    ax[1].imshow(imageB, cmap=plt.cm.nipy_spectral)
    ax[1].set_title(titleB)
    ax[1].set_axis_off()
    ax[2].imshow(imageC, cmap=plt.cm.nipy_spectral)
    ax[2].set_title(titleC)
    ax[2].set_axis_off()
    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off()
      
def doubleplot(imageA, imageB, titleA, titleB, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax = axes.ravel()
    ax[0].imshow(imageA, cmap=cm.gray)
    ax[0].set_title(titleA)
    ax[0].set_axis_off()
    ax[1].imshow(imageB, cmap=plt.cm.nipy_spectral)
    ax[1].set_title(titleB)
    ax[1].set_axis_off()

    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off() 

def _check_dtype_supported(ar):
    # Should use `issubdtype` for bool below, but there's a bug in numpy 1.7
    if not (ar.dtype == bool or np.issubdtype(ar.dtype, np.integer)):
        raise TypeError("Only bool or integer image types are supported. "
                        "Got %s." % ar.dtype)


    

    
    


def normalizeFloatZeroOne(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalizer(x, mi, ma, eps = eps, dtype = dtype)

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def move_image_axes(x, fr, to, adjust_singletons=False):
    """
    x: ndarray
    fr,to: axes string (see `axes_dict`)
    """
    fr = axes_check_and_normalize(fr, length=x.ndim)
    to = axes_check_and_normalize(to)

    fr_initial = fr
    x_shape_initial = x.shape
    adjust_singletons = bool(adjust_singletons)
    if adjust_singletons:
        # remove axes not present in 'to'
        slices = [slice(None) for _ in x.shape]
        for i,a in enumerate(fr):
            if (a not in to) and (x.shape[i]==1):
                # remove singleton axis
                slices[i] = 0
                fr = fr.replace(a,'')
        x = x[slices]
        # add dummy axes present in 'to'
        for i,a in enumerate(to):
            if (a not in fr):
                # add singleton axis
                x = np.expand_dims(x,-1)
                fr += a

    if set(fr) != set(to):
        _adjusted = '(adjusted to %s and %s) ' % (x.shape, fr) if adjust_singletons else ''
        raise ValueError(
            'image with shape %s and axes %s %snot compatible with target axes %s.'
            % (x_shape_initial, fr_initial, _adjusted, to)
        )

    ax_from, ax_to = axes_dict(fr), axes_dict(to)
    if fr == to:
        return x
    return np.moveaxis(x, [ax_from[a] for a in fr], [ax_to[a] for a in fr])
def consume(iterator):
    collections.deque(iterator, maxlen=0)

def _raise(e):
    raise e
def compose(*funcs):
    return lambda x: reduce(lambda f,g: g(f), funcs, x)

def normalizeZeroOne(x):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x
    
def normalizeZero255(x):

     x = x.astype('float32')

     minVal = np.min(x)
     maxVal = np.max(x)
     
     x = ((x-minVal) / (maxVal - minVal + 1.0e-20))
     
     return x * 255   
    
    
def normalizer(x, mi , ma, eps = 1e-20, dtype = np.float32):


    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """


    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        x = normalizeZeroOne(x)
    return x    

    
def LocalThreshold2D(Image, boxsize, offset = 0, size = 10):
    
    if boxsize%2 == 0:
        boxsize = boxsize + 1
    adaptive_thresh = threshold_local(Image, boxsize, offset=offset)
    Binary  = Image > adaptive_thresh
    #Clean =  remove_small_objects(Binary, min_size=size, connectivity=4, in_place=False)

    return Binary

def OtsuThreshold2D(Image, size = 10):
    
    
    adaptive_thresh = threshold_otsu(Image)
    Binary  = Image > adaptive_thresh
    #Clean =  remove_small_objects(Binary, min_size=size, connectivity=4, in_place=False)

    return Binary.astype('uint16')

   ##CARE csbdeep modification of implemented function
def normalizeFloat(x, pmin = 3, pmax = 99.8, axis = None, eps = 1e-20, dtype = np.float32):
    """Percentile based Normalization
    
    Normalize patches of image before feeding into the network
    
    Parameters
    ----------
    x : np array Image patch
    pmin : minimum percentile value for normalization
    pmax : maximum percentile value for normalization
    axis : axis along which the normalization has to be carried out
    eps : avoid dividing by zero
    dtype: type of numpy array, float 32 default
    """
    mi = np.percentile(x, pmin, axis = axis, keepdims = True)
    ma = np.percentile(x, pmax, axis = axis, keepdims = True)
    return normalize_mi_ma(x, mi, ma, eps = eps, dtype = dtype)


def normalize_mi_ma(x, mi , ma, eps = 1e-20, dtype = np.float32):
    
    
    """
    Number expression evaluation for normalization
    
    Parameters
    ----------
    x : np array of Image patch
    mi : minimum input percentile value
    ma : maximum input percentile value
    eps: avoid dividing by zero
    dtype: type of numpy array, float 32 defaut
    """
    
    
    if dtype is not None:
        x = x.astype(dtype, copy = False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy = False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy = False)
        eps = dtype(eps)
        
    try: 
        import numexpr
        x = numexpr.evaluate("(x - mi ) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

        
    return x    





def load_full_training_data(directory, filename,axes=None, verbose= True):
    """ Load training data in .npz format.
    The data file is expected to have the keys 'data' and 'label'     
    """
    
    if directory is not None:
      npzdata=np.load(directory + filename)
    else:
      npzdata=np.load(filename)  
    
    
    X = npzdata['data']
    Y = npzdata['label']
    
    
        
    
    if axes is None:
        axes = npzdata['axes']
    axes = axes_check_and_normalize(axes)
    assert 'C' in axes
    n_images = X.shape[0]
    assert X.shape[0] == Y.shape[0]
    assert 0 < n_images <= X.shape[0]
  
    
    X, Y = X[:n_images], Y[:n_images]
    channel = axes_dict(axes)['C']
    

       

    X = move_channel_for_backend(X,channel=channel)
    
    axes = axes.replace('C','') # remove channel
    if backend_channels_last():
        axes = axes+'C'
    else:
        axes = axes[:1]+'C'+axes[1:]

   

    if verbose:
        ax = axes_dict(axes)
        n_train = len(X)
        image_size = tuple( X.shape[ax[a]] for a in 'TZYX' if a in axes )
        n_dim = len(image_size)
        n_channel_in = X.shape[ax['C']]

        print('number of  images:\t', n_train)
       
        print('image size (%dD):\t\t'%n_dim, image_size)
        print('axes:\t\t\t\t', axes)
        print('channels in / out:\t\t', n_channel_in)

    return (X,Y), axes


def backend_channels_last():
    import keras.backend as K
    assert K.image_data_format() in ('channels_first','channels_last')
    return K.image_data_format() == 'channels_last'


def move_channel_for_backend(X,channel):
    if backend_channels_last():
        return np.moveaxis(X, channel, -1)
    else:
        return np.moveaxis(X, channel,  1)
        

def axes_check_and_normalize(axes,length=None,disallowed=None,return_allowed=False):
    """
    S(ample), T(ime), C(hannel), Z, Y, X
    """
    allowed = 'STCZYX'
    axes = str(axes).upper()
    consume(a in allowed or _raise(ValueError("invalid axis '%s', must be one of %s."%(a,list(allowed)))) for a in axes)
    disallowed is None or consume(a not in disallowed or _raise(ValueError("disallowed axis '%s'."%a)) for a in axes)
    consume(axes.count(a)==1 or _raise(ValueError("axis '%s' occurs more than once."%a)) for a in axes)
    length is None or len(axes)==length or _raise(ValueError('axes (%s) must be of length %d.' % (axes,length)))
    return (axes,allowed) if return_allowed else axes
def axes_dict(axes):
    """
    from axes string to dict
    """
    axes, allowed = axes_check_and_normalize(axes,return_allowed=True)
    return { a: None if axes.find(a) == -1 else axes.find(a) for a in allowed }
    # return collections.namedt     
    
    
def _raise(e):
    raise e

# https://docs.python.org/3/library/itertools.html#itertools-recipes
def consume(iterator):
    collections.deque(iterator, maxlen=0)   