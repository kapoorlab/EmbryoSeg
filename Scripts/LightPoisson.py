import os
import sys
import glob
import numpy as np
from pathlib import Path
sys.path.append('/home/vkapoor/EmbryoSeg/embryoseg')
from tifffile import imread, imwrite
#from embryoseg.sampling import Crops


Data_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/LightSheetTraining/CropRaw/'
Noisy_Data_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/LightSheetTraining/NoisyCropRaw/'

Path(Noisy_Data_dir).mkdir(exist_ok=True)

Raw_path = os.path.join(Data_dir, '*tif')
filesRaw = glob.glob(Raw_path)
filesRaw.sort
for fname in filesRaw:


       Name = os.path.basename(os.path.splitext(fname)[0])
       image = imread(fname)
       #Make it noisy
       noisymask = np.random.poisson(image)
       noisyimage = image + noisymask

       imwrite(Noisy_Data_dir + "/" + Name + ".tif", noisyimage.astype("float32"))


       
