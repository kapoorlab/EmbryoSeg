
import sys
sys.path.append('../../')
from sampling import Crops
#from embryoseg.sampling import Crops


Data_dir = '/home/sancere/Kepler/Lucas2Varun/1_StarWat_Training/DrosophilaMembraneData/'
NPZ_filename = 'DrosophilaMembrane'

PatchX = 384
PatchY = 256
PatchZ = 128
n_patches_per_image = 10


Crops(Data_dir, NPZ_filename, PatchZ, PatchY, PatchX,n_patches_per_image)
