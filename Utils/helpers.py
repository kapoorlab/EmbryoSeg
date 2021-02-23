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
from tifffile import imread, imwrite
import warnings
from skimage.morphology import erosion, dilation, square
from scipy.ndimage.interpolation import zoom
from skimage.morphology import skeletonize
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



 
def STARPrediction(image, model, min_size, n_tiles, MaskImage = None, smartcorrection = None, UseProbability = True):
    
    print('Applying StarDist prediction')
    
    image = normalize(image, 1, 99.8, axis = (0,1))
    shape = [image.shape[0], image.shape[1]]
    image = zero_pad(image, 64, 64)
    
    MidImage, details = model.predict_instances(image, n_tiles = n_tiles)
    
    StarImage = MidImage[:shape[0],:shape[1]]
    
    SmallProbability, SmallDistance = model.predict(image, n_tiles = n_tiles)
    grid = model.config.grid
    Probability = zoom(SmallProbability, ( grid[0] , grid[1] ))
    Distance = MaxProjectDist(SmallDistance, axis=-1)
    Distance = zoom(Distance, ( grid[0] , grid[1] ))
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

    Watershed = remove_small_objects(Watershed.astype('uint16'), min_size = min_size)
 
    


    return Watershed, Markers, StarImage     


def UNETPrediction(image, model, min_size, n_tiles, axis, threshold = 20):
    
    print('Applying UNET prediction')
    
    Segmented = model.predict(image, axis, n_tiles = n_tiles)
    thresh = threshold_otsu(Segmented)
    Binary = Segmented > thresh
    
    #Postprocessing steps
    Filled = binary_fill_holes(Binary)
    Finalimage = label(Filled)
    Finalimage = fill_label_holes(Finalimage)
    if len(image.shape) == 2:
        Finalimage = remove_small_objects(Finalimage.astype('uint16'), min_size = min_size)
    if len(image.shape) > 2:
        for i in range(0, Finalimage.shape[0]):
            Finalimage[i,:] = remove_small_objects(label(Finalimage[i,:]).astype('uint16'), min_size = min_size)
    Finalimage = relabel_sequential(Finalimage)[0]
    #Stitch 2D slices using relabel function
    #if len(image.shape) > 2:
           #Finalimage = merge_labels_across_volume(Finalimage.astype('uint16'), RelabelZ, threshold= threshold)
    return Finalimage        

def SmartSeedPredictionSliced(ImageDir, SaveDir, fname, UnetModel, StarModel, NoiseModel = None, min_size_mask = 100, min_size = 100, 
                              n_tiles = (2,2), doMask = True, smartcorrection = None, threshold = 20, masklinkthreshold = 100, UseProbability = True, filtersize = 0):
    
    
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
                Path(DenoisedResults).mkdir(exist_ok = True)
                imwrite((DenoisedResults + Name + '.tif' ) , image.astype('float32'))       
    TimeMask = DownsampleData(TimeMask, 1.0/DownsampleFactor)
    
    if StarModel is not None: 
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
    

# To be used for BIG cells like mouse embryos to get proper boundary reconstruction. This method used the distance map for doing the watershedding and then
# does the smart correction over the pixels specified to color the mask with the color of the closest label. Very useful for weak edges
    
def WatershedSmartCorrection(Image, Label, mask, grid, smartcorrection = 20, max_size = 100000, min_size = 1):
    
    
   
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
       
    sexyImage = remove_small_objects(sexyImage.astype('uint16'), min_size = min_size)
    sexyImage = fill_label_holes(sexyImage)
    sexyImage, forward_map, inverse_map = relabel_sequential(sexyImage)
    
    
    return sexyImage, Labelmarkers  





def WatershedDistanceMarker(Image, Label, mask, grid, smartcorrection, max_size = 100000, min_size = 1):
    
    
   
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
    copymask = mask.copy()
    

    
    sexyImage = remove_small_objects(sexyImage.astype('uint16'), min_size = min_size)
    sexyImage = fill_label_holes(sexyImage)
    sexyImage, forward_map, inverse_map = relabel_sequential(sexyImage)
   
    
    return sexyImage, markers 

                 

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




 

def quadplot(imageA, imageB, imageC, imageD, titleA, titleB, titleC, titleD, targetdir = None, File = None, plotTitle = None):
    fig, axes = plt.subplots(1, 4, figsize=(15, 6))
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
    ax[3].imshow(imageD, cmap=plt.cm.nipy_spectral)
    ax[3].set_title(titleD)
    ax[3].set_axis_off()
    
    plt.tight_layout()
    plt.show()
    for a in ax:
      a.set_axis_off()



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