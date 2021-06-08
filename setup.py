from setuptools import setup
from setuptools import find_packages

with open('README.md') as f:
    long_description = f.read()


setup(name="embryoseg",
<<<<<<< HEAD
      version='1.0.3',
=======
      version='1.2.4',
>>>>>>> 66610c015a2c23d790167dd23cda7fc6abb9dbaf
      author='Varun Kapoor',
      author_email='randomaccessiblekapoor@gmail.com',
      url='https://github.com/kapoorlab/EmbryoSeg/',
      description='SmartSeed Segmentation for Mouse Pre-implantation cells.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=["numpy", "pandas", "napari==0.4.3", "pyqt5", "natsort", "scikit-image", "scipy", "opencv-python-headless", "tifffile", "matplotlib"],
      packages=['embryoseg','embryoseg/utils','embryoseg/notebooks','embryoseg/models'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Natural Language :: English',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.7',
                   ])
