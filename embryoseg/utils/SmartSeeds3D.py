#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:40:47 2019

@author: aimachine
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:38:04 2019

@author: aimachine
"""

import numpy as np
import os
#from IPython.display import clear_output
from stardist.models import Config3D, StarDist3D
from stardist import calculate_extents, Rays_GoldenSpiral
from scipy.ndimage.morphology import binary_fill_holes
from tqdm import tqdm
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import  binary_dilation
from csbdeep.utils import normalize
import glob
import cv2
import sys
from csbdeep.data import RawData, create_patches
from csbdeep.io import load_training_data
from csbdeep.utils import axes_dict, axes_check_and_normalize
from csbdeep.models import Config, CARE
from tifffile import imread, imwrite
from skimage.morphology import label    
import warnings
import collections
import platform
import random
from six.moves import range, zip, map, reduce, filter
from pathlib import Path
from csbdeep.data.transform import Transform, permute_axes, broadcast_target   
from csbdeep.utils import compose
from csbdeep.io import save_training_data
from scipy.ndimage.filters import minimum_filter, maximum_filter, median_filter, uniform_filter
from tensorflow.keras.utils import Sequence



def _raise(e):
    raise e

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
    
class SmartSeeds3D(object):






     def __init__(self, BaseDir, NPZfilename, model_name, model_dir, n_patches_per_image, DownsampleFactor = 1, TrainUNET = True, TrainSTAR = True, GenerateNPZ = True, GenerateStarNPZ = True,  copy_model_dir = None, PatchX=256, PatchY=256, PatchZ = 16,  use_gpu = True,  batch_size = 4, depth = 3, kern_size = 3, startfilter = 48, n_rays = 16, epochs = 400, learning_rate = 0.0001):
         
         
         
         
         self.NPZfilename = NPZfilename
         self.BaseDir = BaseDir
         self.DownsampleFactor = DownsampleFactor
         self.model_dir = model_dir
         self.GenerateNPZ = GenerateNPZ
         self.GenerateStarNPZ = GenerateStarNPZ
         self.TrainUNET = TrainUNET
         self.TrainSTAR = TrainSTAR
         self.copy_model_dir = copy_model_dir
         self.model_name = model_name
         self.epochs = epochs
         self.learning_rate = learning_rate
         self.depth = depth
         self.n_rays = n_rays
         self.kern_size = kern_size
         self.PatchX = PatchX
         self.PatchY = PatchY
         self.PatchZ = PatchZ
         self.batch_size = batch_size
         self.use_gpu = use_gpu
         self.startfilter = startfilter
         #Attributes to be filled later
         self.n_patches_per_image =  n_patches_per_image
        
         
         
         #Load training and validation data
         self.Train()
         
     class DataSequencer(Sequence):
         
         
            def __init__(self, files, axis_norm, Normalize = True, labelMe = False):
                super().__init__() 
                
                self.files = files
               
                self.axis_norm = axis_norm
                self.labelMe = labelMe
                self.Normalize = Normalize
                
            def __len__(self):
                return len(self.files)
            
            
            def __getitem__(self, i):
                
                        #Read Raw images
                         if self.Normalize == True:
                                 x = ReadFloat(self.files[i]) 
                                 x = normalize(x,1,99.8,axis= self.axis_norm) 
                         if self.labelMe == True:
                                 #Read Label images
                                 x = ReadInt(self.files[i])
                                 x = label(x)
                   
                         return x
        
        
        
         
     def Train(self):
         
                    
                    Raw = sorted(glob.glob(self.BaseDir + '/Raw/' + '*.tif'))
                    Path(self.BaseDir + '/BinaryMask/').mkdir(exist_ok=True)
                    Path(self.BaseDir + '/RealMask/').mkdir(exist_ok=True)
                    RealMask = sorted(glob.glob(self.BaseDir + '/RealMask/' + '*.tif'))
                    
                    
                    print('Instance segmentation masks:', len(RealMask))
                    if len(RealMask)== 0:
                        
                        print('Making labels')
                        Mask = sorted(glob.glob(self.BaseDir + '/BinaryMask/' + '*.tif'))
                        
                        for fname in Mask:
                    
                           image = imread(fname)
                    
                           Name = os.path.basename(os.path.splitext(fname)[0])
                    
                           Binaryimage = label(image) 
                    
                           imwrite((self.BaseDir + '/RealMask/' + Name + '.tif'), Binaryimage.astype('uint16'))
                           
                
                    Mask = sorted(glob.glob(self.BaseDir + '/BinaryMask/' + '*.tif'))
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
                    
                   
                    if self.GenerateNPZ:
                        
                      raw_data = RawData.from_folder (
                      basepath    = self.BaseDir,
                      source_dirs = ['Raw/'],
                      target_dir  = 'BinaryMask/',
                      axes        = 'ZYX',
                       )
                    
                      X, Y, XY_axes = create_patches (
                      raw_data            = raw_data,
                      patch_size          = (self.PatchZ,self.PatchY,self.PatchX),
                      n_patches_per_image = self.n_patches_per_image,
                      save_file           = self.BaseDir + self.NPZfilename + '.npz',
                      )
                      
                    if self.GenerateStarNPZ:
                        
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
                      save_file           = self.BaseDir + self.NPZfilename + 'Star' + '.npz',
                      )  
                    
                    # Training UNET model
                    if self.TrainUNET:
                            print('Training UNET model')
                            load_path = self.BaseDir + self.NPZfilename + '.npz'
        
                            (X,Y), (X_val,Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)
                            c = axes_dict(axes)['C']
                            n_channel_in, n_channel_out = X.shape[c], Y.shape[c]
                            
                            config = Config(axes, n_channel_in, n_channel_out, unet_n_depth= self.depth,train_epochs= self.epochs, train_batch_size = self.batch_size, unet_n_first = self.startfilter, train_loss = 'mse', unet_kern_size = self.kern_size, train_learning_rate = self.learning_rate, train_reduce_lr={'patience': 5, 'factor': 0.5})
                            print(config)
                            vars(config)
                            
                            model = CARE(config , name = 'UNET' + self.model_name, basedir = self.model_dir)
                            
                            
                                                     
                                 
                            if self.copy_model_dir is not None:   
                              if os.path.exists(self.copy_model_dir + 'UNET' + self.copy_model_name + '/' + 'weights_now.h5') and os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_now.h5') == False:
                                 print('Loading copy model')
                                 model.load_weights(self.copy_model_dir + 'UNET' + self.copy_model_name + '/' + 'weights_now.h5')   
                            
                            if os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_now.h5'):
                                print('Loading checkpoint model')
                                model.load_weights(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_now.h5')
                            
                            if os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_last.h5'):
                                print('Loading checkpoint model')
                                model.load_weights(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_last.h5')
                                
                            if os.path.exists(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_best.h5'):
                                print('Loading checkpoint model')
                                model.load_weights(self.model_dir + 'UNET' + self.model_name + '/' + 'weights_best.h5')    
                            
                            history = model.train(X,Y, validation_data=(X_val,Y_val))

                    if self.TrainSTAR:
                            print('Training StarDistModel model with unet backbone')
                            self.axis_norm = (0,1,2)
                            load_path = self.BaseDir + self.NPZfilename + 'Star' + '.npz'
                            
                            
                            (self.X_trn,self.Y_trn), (self.X_val, self.Y_val), axes = load_training_data(load_path, validation_split=0.1, verbose=True)
                          
                            print(Config3D.__doc__)
                            extents = calculate_extents(self.Y)
                            anisotropy = tuple(np.max(extents) / extents)
                            rays = Rays_GoldenSpiral(self.n_rays, anisotropy=anisotropy)
                            conf = Config3D (
                                  rays       = rays,
                                  anisotropy = anisotropy,
                                  backbone='resnet',
                                  train_epochs = self.epochs,
                                  train_learning_rate = self.learning_rate,
                                  resnet_n_blocks = self.depth,
                                  train_checkpoint = self.model_dir + self.model_name +'.h5',
                                  resnet_kernel_size = (self.kern_size, self.kern_size, self.kern_size),
                                  train_patch_size = (self.PatchZ, self.PatchX, self.PatchY ),
                                  train_batch_size = self.batch_size,
                                  resnet_n_filter_base = self.startfilter,
                                  train_dist_loss = 'mae',
                                  grid         = (1,1,1),#tuple(1 if a > 1.5 else 2 for a in anisotropy),
                                  use_gpu      = self.use_gpu,
                                  n_channel_in = 1
                                  )
                            
                            print(conf)
                            vars(conf)
                                 
                                
                            Starmodel = StarDist3D(conf, name=self.model_name, basedir=self.model_dir)
                            print(Starmodel._axes_tile_overlap('ZYX'), os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5'))                            
                            median_size = calculate_extents(self.Y, np.median)
                            fov = np.array(Starmodel._axes_tile_overlap('ZYX'))
                            if any(median_size > fov):
                                 print("WARNING: median object size larger than field of view of the neural network.")
                                 
                                 
                            if self.copy_model_dir is not None:   
                              if os.path.exists(self.copy_model_dir + self.copy_model_name + '/' + 'weights_now.h5') and os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5') == False:
                                 print('Loading copy model')
                                 Starmodel.load_weights(self.copy_model_dir + self.copy_model_name + '/' + 'weights_now.h5')  
                              if os.path.exists(self.copy_model_dir + self.copy_model_name + '/' + 'weights_last.h5') and os.path.exists(self.model_dir + self.model_name + '/' + 'weights_last.h5') == False:
                                 print('Loading copy model')
                                 Starmodel.load_weights(self.copy_model_dir + self.copy_model_name + '/' + 'weights_last.h5')

                              if os.path.exists(self.copy_model_dir + self.copy_model_name + '/' + 'weights_best.h5') and os.path.exists(self.model_dir + self.model_name + '/' + 'weights_best.h5') == False:
                                 print('Loading copy model')
                                 Starmodel.load_weights(self.copy_model_dir + self.copy_model_name + '/' + 'weights_best.h5')

 
                            
                            if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_now.h5'):
                                print('Loading checkpoint model')
                                Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_now.h5')
                                
                            if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_last.h5'):
                                print('Loading checkpoint model')
                                Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_last.h5')   
                                
                            if os.path.exists(self.model_dir + self.model_name + '/' + 'weights_best.h5'):
                                print('Loading checkpoint model')
                                Starmodel.load_weights(self.model_dir + self.model_name + '/' + 'weights_best.h5')     
                                 
                            Starmodel.train(self.X_trn, self.Y_trn, validation_data=(self.X_val,self.Y_val), epochs = self.epochs)
                            Starmodel.optimize_thresholds(self.X_val, self.Y_val)
        
        
                 
         
def ReadFloat(fname):

    return imread(fname).astype('float32')         
         

def ReadInt(fname):

    return imread(fname).astype('uint16')         
def no_background_patches(threshold=0.8, percentile=99.9):

    """Returns a patch filter to be used by :func:`create_patches` to determine for each image pair which patches
    are eligible for sampling. The purpose is to only sample patches from "interesting" regions of the raw image that
    actually contain a substantial amount of non-background signal. To that end, a maximum filter is applied to the target image
    to find the largest values in a region.
    Parameters
    ----------
    threshold : float, optional
        Scalar threshold between 0 and 1 that will be multiplied with the (outlier-robust)
        maximum of the image (see `percentile` below) to denote a lower bound.
        Only patches with a maximum value above this lower bound are eligible to be sampled.
    percentile : float, optional
        Percentile value to denote the (outlier-robust) maximum of an image, i.e. should be close 100.
    Returns
    -------
    function
        Function that takes an image pair `(y,x)` and the patch size as arguments and
        returns a binary mask of the same size as the image (to denote the locations
        eligible for sampling for :func:`create_patches`). At least one pixel of the
        binary mask must be ``True``, otherwise there are no patches to sample.
    Raises
    ------
    ValueError
        Illegal arguments.
    """

    (np.isscalar(percentile) and 0 <= percentile <= 100) or _raise(ValueError())
    (np.isscalar(threshold)  and 0 <= threshold  <=   1) or _raise(ValueError())

   
    def _filter(datas, patch_size, dtype=np.bool):
        #Filter the mask by ignoring zero region mask
        image = datas[0]
        
        if dtype is not None:
            image = image.astype(dtype)
        # make max filter patch_size smaller to avoid only few non-bg pixel close to image border
        patch_size = [(p//2 if p>1 else p) for p in patch_size]
        
        filtered = maximum_filter(image, patch_size)
        
        return filtered > threshold * np.percentile(image,percentile)
    return _filter



## Sample patches

def sample_patches_from_multiple_stacks(datas, patch_size, n_samples, datas_mask=None, patch_filter=None, verbose=False):
    """ sample matching patches of size `patch_size` from all arrays in `datas` """

    # TODO: some of these checks are already required in 'create_patches'
    len(patch_size)==datas[0].ndim or _raise(ValueError())

    if not all(( a.shape == datas[0].shape for a in datas )):
        raise ValueError("all input shapes must be the same: %s" % (" / ".join(str(a.shape) for a in datas)))

    if not all(( 0 < s <= d for s,d in zip(patch_size,datas[0].shape) )):
        raise ValueError("patch_size %s negative or larger than data shape %s along some dimensions" % (str(patch_size), str(datas[0].shape)))
    if patch_filter is None:
        patch_mask = np.ones(datas[0].shape,dtype=np.bool)
    else:
        patch_mask = patch_filter(datas, patch_size)
    # get the valid indices
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip(patch_size, datas[0].shape)])
    valid_inds = np.where(patch_mask[border_slices])
    n_valid = len(valid_inds[0])
    if n_valid == 0:
        
        raise ValueError("'patch_filter' didn't return any region to sample from")
    
    sample_inds = choice(range(n_valid), n_samples, replace=(n_valid < n_samples))

    # valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)] # slow for large n_valid
    # rand_inds = [v[sample_inds] for v in valid_inds]
    rand_inds = [v[sample_inds] + s.start for s, v in zip(border_slices, valid_inds)]

    # res = [np.stack([data[r[0] - patch_size[0] // 2:r[0] + patch_size[0] - patch_size[0] // 2,
    #                  r[1] - patch_size[1] // 2:r[1] + patch_size[1] - patch_size[1] // 2,
    #                  r[2] - patch_size[2] // 2:r[2] + patch_size[2] - patch_size[2] // 2,
    #                  ] for r in zip(*rand_inds)]) for data in datas]

    res = [np.stack([data[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,patch_size))] for r in zip(*rand_inds)]) for data in datas]

    return res


def choice(population, k=1, replace=True):
    ver = platform.sys.version_info
    if replace and (ver.major,ver.minor) in [(2,7),(3,5)]: # python 2.7 or 3.5
        # slow if population is large and not a np.ndarray
        return list(np.random.choice(population, k, replace=replace))
    else:
        try:
            # save state of 'random' and set seed using 'np.random'
            state = random.getstate()
            random.seed(np.random.randint(np.iinfo(int).min, np.iinfo(int).max))
            if replace:
                # sample with replacement
                return random.choices(population, k=k)
            else:
                # sample without replacement
                return random.sample(population, k=k)
        finally:
            # restore state of 'random'
            random.setstate(state)
## Create training data

def _valid_low_high_percentiles(ps):
    return isinstance(ps,(list,tuple,np.ndarray)) and len(ps)==2 and all(map(np.isscalar,ps)) and (0<=ps[0]<ps[1]<=100)

def _memory_check(n_required_memory_bytes, thresh_free_frac=0.5, thresh_abs_bytes=1024*1024**2):
    try:
        # raise ImportError
        import psutil
        mem = psutil.virtual_memory()
        mem_frac = n_required_memory_bytes / mem.available
        if mem_frac > 1:
            raise MemoryError('Not enough available memory.')
        elif mem_frac > thresh_free_frac:
            print('Warning: will use at least %.0f MB (%.1f%%) of available memory.\n' % (n_required_memory_bytes/1024**2,100*mem_frac), file=sys.stderr)
            sys.stderr.flush()
    except ImportError:
        if n_required_memory_bytes > thresh_abs_bytes:
            print('Warning: will use at least %.0f MB of memory.\n' % (n_required_memory_bytes/1024**2), file=sys.stderr)
            sys.stderr.flush()

def sample_percentiles(pmin=(1,3), pmax=(99.5,99.9)):
    """Sample percentile values from a uniform distribution.
    Parameters
    ----------
    pmin : tuple
        Tuple of two values that denotes the interval for sampling low percentiles.
    pmax : tuple
        Tuple of two values that denotes the interval for sampling high percentiles.
    Returns
    -------
    function
        Function without arguments that returns `(pl,ph)`, where `pl` (`ph`) is a sampled low (high) percentile.
    Raises
    ------
    ValueError
        Illegal arguments.
    """
    _valid_low_high_percentiles(pmin) or _raise(ValueError(pmin))
    _valid_low_high_percentiles(pmax) or _raise(ValueError(pmax))
    pmin[1] < pmax[0] or _raise(ValueError())
    return lambda: (np.random.uniform(*pmin), np.random.uniform(*pmax))


def norm_percentiles(percentiles=sample_percentiles(), relu_last=False):
    """Normalize extracted patches based on percentiles from corresponding raw image.
    Parameters
    ----------
    percentiles : tuple, optional
        A tuple (`pmin`, `pmax`) or a function that returns such a tuple, where the extracted patches
        are (affinely) normalized in such that a value of 0 (1) corresponds to the `pmin`-th (`pmax`-th) percentile
        of the raw image (default: :func:`sample_percentiles`).
    relu_last : bool, optional
        Flag to indicate whether the last activation of the CARE network is/will be using
        a ReLU activation function (default: ``False``)
    Return
    ------
    function
        Function that does percentile-based normalization to be used in :func:`create_patches`.
    Raises
    ------
    ValueError
        Illegal arguments.
    Todo
    ----
    ``relu_last`` flag problematic/inelegant.
    """
    if callable(percentiles):
        _tmp = percentiles()
        _valid_low_high_percentiles(_tmp) or _raise(ValueError(_tmp))
        get_percentiles = percentiles
    else:
        _valid_low_high_percentiles(percentiles) or _raise(ValueError(percentiles))
        get_percentiles = lambda: percentiles

    def _normalize(patches_x,patches_y, x,y,mask,channel):
        pmins, pmaxs = zip(*(get_percentiles() for _ in patches_x))
        percentile_axes = None if channel is None else tuple((d for d in range(x.ndim) if d != channel))
        _perc = lambda a,p: np.percentile(a,p,axis=percentile_axes,keepdims=True)
        patches_x_norm = normalize_mi_ma(patches_x, _perc(x,pmins), _perc(x,pmaxs))
        if relu_last:
            pmins = np.zeros_like(pmins)
        patches_y_norm = normalize_mi_ma(patches_y, _perc(y,pmins), _perc(y,pmaxs))
        return patches_x_norm, patches_y_norm

    return _normalize



def normalize(x, pmin=3, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

def create_local_patches(
        raw_data,
        patch_size,
        n_patches_per_image,
        patch_axes    = None,
        save_file     = None,
        transforms    = None,
        patch_filter  = no_background_patches(),
        normalization = norm_percentiles(),
        shuffle       = True,
        verbose       = True,
    ):
    """Create normalized training data to be used for neural network training.
    Parameters
    ----------
    raw_data : :class:`RawData`
        Object that yields matching pairs of raw images.
    patch_size : tuple
        Shape of the patches to be extraced from raw images.
        Must be compatible with the number of dimensions and axes of the raw images.
        As a general rule, use a power of two along all XYZT axes, or at least divisible by 8.
    n_patches_per_image : int
        Number of patches to be sampled/extracted from each raw image pair (after transformations, see below).
    patch_axes : str or None
        Axes of the extracted patches. If ``None``, will assume to be equal to that of transformed raw data.
    save_file : str or None
        File name to save training data to disk in ``.npz`` format (see :func:`csbdeep.io.save_training_data`).
        If ``None``, data will not be saved.
    transforms : list or tuple, optional
        List of :class:`Transform` objects that apply additional transformations to the raw images.
        This can be used to augment the set of raw images (e.g., by including rotations).
        Set to ``None`` to disable. Default: ``None``.
    patch_filter : function, optional
        Function to determine for each image pair which patches are eligible to be extracted
        (default: :func:`no_background_patches`). Set to ``None`` to disable.
    normalization : function, optional
        Function that takes arguments `(patches_x, patches_y, x, y, mask, channel)`, whose purpose is to
        normalize the patches (`patches_x`, `patches_y`) extracted from the associated raw images
        (`x`, `y`, with `mask`; see :class:`RawData`). Default: :func:`norm_percentiles`.
    shuffle : bool, optional
        Randomly shuffle all extracted patches.
    verbose : bool, optional
        Display overview of images, transforms, etc.
    Returns
    -------
    tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`, str)
        Returns a tuple (`X`, `Y`, `axes`) with the normalized extracted patches from all (transformed) raw images
        and their axes.
        `X` is the array of patches extracted from source images with `Y` being the array of corresponding target patches.
        The shape of `X` and `Y` is as follows: `(n_total_patches, n_channels, ...)`.
        For single-channel images, `n_channels` will be 1.
    Raises
    ------
    ValueError
        Various reasons.
    Example
    -------
    >>> raw_data = RawData.from_folder(basepath='data', source_dirs=['source1','source2'], target_dir='GT', axes='ZYX')
    >>> X, Y, XY_axes = create_patches(raw_data, patch_size=(32,128,128), n_patches_per_image=16)
    Todo
    ----
    - Save created patches directly to disk using :class:`numpy.memmap` or similar?
      Would allow to work with large data that doesn't fit in memory.
    """
    ## images and transforms
    if transforms is None:
        transforms = []
    transforms = list(transforms)
    if patch_axes is not None:
        transforms.append(permute_axes(patch_axes))
    if len(transforms) == 0:
        transforms.append(Transform.identity())

    if normalization is None:
        normalization = lambda patches_x, patches_y, x, y, mask, channel: (patches_x, patches_y)

    image_pairs, n_raw_images = raw_data.generator(), raw_data.size
    tf = Transform(*zip(*transforms)) # convert list of Transforms into Transform of lists
    image_pairs = compose(*tf.generator)(image_pairs) # combine all transformations with raw images as input
    n_transforms = np.prod(tf.size)
    n_images = n_raw_images * n_transforms
    n_patches = n_images * n_patches_per_image
    n_required_memory_bytes = 2 * n_patches*np.prod(patch_size) * 4

    ## memory check
    _memory_check(n_required_memory_bytes)

    ## summary
    if verbose:
        print('='*66)
        print('%5d raw images x %4d transformations   = %5d images' % (n_raw_images,n_transforms,n_images))
        print('%5d images     x %4d patches per image = %5d patches in total' % (n_images,n_patches_per_image,n_patches))
        print('='*66)
        print('Input data:')
        print(raw_data.description)
        print('='*66)
        print('Transformations:')
        for t in transforms:
            print('{t.size} x {t.name}'.format(t=t))
        print('='*66)
        print('Patch size:')
        print(" x ".join(str(p) for p in patch_size))
        print('=' * 66)

    sys.stdout.flush()

    ## sample patches from each pair of transformed raw images
    X = np.empty((n_patches,)+tuple(patch_size),dtype=np.float32)
    Y = np.empty_like(X)

    for i, (x,y,_axes,mask) in tqdm(enumerate(image_pairs),total=n_images,disable=(not verbose)):
        if i >= n_images:
            warnings.warn('more raw images (or transformations thereof) than expected, skipping excess images.')
            break
        if i==0:
            axes = axes_check_and_normalize(_axes,len(patch_size))
            channel = axes_dict(axes)['C']
        # checks
        # len(axes) >= x.ndim or _raise(ValueError())
       
        axes == axes_check_and_normalize(_axes) or _raise(ValueError('not all images have the same axes.'))
        x.shape == y.shape or _raise(ValueError())
        mask is None or mask.shape == x.shape or _raise(ValueError())
        (channel is None or (isinstance(channel,int) and 0<=channel<x.ndim)) or _raise(ValueError())
        channel is None or patch_size[channel]==x.shape[channel] or _raise(ValueError('extracted patches must contain all channels.'))

        
        _Y,_X = sample_patches_from_multiple_stacks((y,x), patch_size, n_patches_per_image, mask, patch_filter)

        s = slice(i*n_patches_per_image,(i+1)*n_patches_per_image)
        X[s], Y[s] = normalization(_X,_Y, x,y,mask,channel)
        



    if shuffle:
        shuffle_inplace(X,Y)

    axes = 'SC'+axes.replace('C','')
    if channel is None:
        X = np.expand_dims(X,1)
        Y = np.expand_dims(Y,1)
    else:
        X = np.moveaxis(X, 1+channel, 1)
        Y = np.moveaxis(Y, 1+channel, 1)


    #If the white pixel regions are less reject them
    Xfiltered = []
    Yfiltered = []
    #Testing half the volume has to be filled white
    for k in range(0, Y.shape[0]):
         if np.mean(Y[k]) > 0.5 * (Y[k].shape[0] * Y[k].shape[1] * Y[k].shape[2]):
                Yfiltered.append(Y[k])
                Xfiltered.append(X[k])     
    Xfiltered = np.asarray(Xfiltered) 
    Yfiltered = np.asarray(Yfiltered)


    if save_file is not None:
        print('Saving data to %s.' % str(Path(save_file)))
        save_training_data(save_file, Xfiltered, Yfiltered, axes)

    return Xfiltered,Yfiltered,axes   

def shuffle_inplace(*arrs,**kwargs):
    seed = kwargs.pop('seed', None)
    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed=seed)
    state = rng.get_state()
    for a in arrs:
        rng.set_state(state)
        rng.shuffle(a)
         
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
         
         
