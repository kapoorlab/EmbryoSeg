import os
import sys
sys.path.append('/home/vkapoor/EmbryoSeg/embryoseg')
from sampling import Crops
#from embryoseg.sampling import Crops


Data_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/LightSheetTraining/'
NPZ_filename = 'BigGuignardLabAscadianEmbryo'

PatchX = 256
PatchY = 256
PatchZ = 128
n_patches_per_image = 10


Crops(Data_dir, NPZ_filename, PatchZ, PatchY, PatchX,n_patches_per_image)
