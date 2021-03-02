from setuptools import setup
from setuptools import find_packages

with open('README.md') as f:
    long_description = f.read()


setup(name="embryoseg",
      version='1.1.5',
      author='Varun Kapoor',
      author_email='randomaccessiblekapoor@gmail.com',
      url='https://github.com/kapoorlab/EmbryoSeg/',
      description='SmartSeed Segmentation for Mouse Pre-implantation cells.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=["numpy", "pandas", "napari", "pyqt5", "btrack","natsort", "scikit-image", "scipy", "opencv-python-headless", "tifffile", "matplotlib", "stardist", "csbdeep"],
      packages=['embryoseg','embryoseg/utils','embryoseg/models'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Natural Language :: English',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.9',
                   ])
