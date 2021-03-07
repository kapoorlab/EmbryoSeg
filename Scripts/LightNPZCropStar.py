import os
import sys
sys.path.append('/home/vkapoor/EmbryoSeg/embryoseg')
from sampling import NPZCrops
#from embryoseg.sampling import Crops


Data_dir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/LightSheetTraining/'
NPZ_filename = 'GuignardLabAscadianEmbryo'



NPZCrops(Data_dir, NPZ_filename)
