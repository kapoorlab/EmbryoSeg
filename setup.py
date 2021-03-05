from setuptools import setup
from setuptools import find_packages

with open('README.md') as f:
    long_description = f.read()


setup(name="embryoseg",
      version='1.2.0',
      author='Varun Kapoor',
      author_email='randomaccessiblekapoor@gmail.com',
      url='https://github.com/kapoorlab/EmbryoSeg/',
      description='SmartSeed Segmentation for animal embryonic cells.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      install_requires=[ "numpy", "pandas", "napari","pyqt5", "natsort", "scikit-image", "scipy", "opencv-python-headless", "tifffile", "matplotlib", "csbdeep"],
      packages=['embryoseg','embryoseg/utils','embryoseg/stardist_kapoor','embryoseg/stardist_kapoor/geometry','embryoseg/stardist_kapoor/kernels','embryoseg/stardist_kapoor/lib', 'embryoseg/stardist_kapoor/lib/qhull_src', 'embryoseg/stardist_kapoor/lib/qhull_src/src/libqhull_r', 'embryoseg/stardist_kapoor/lib/qhull_src/src/libqhullcpp','embryoseg/stardist_kapoor/models'],
      classifiers=['Development Status :: 3 - Alpha',
                   'Natural Language :: English',
                   'License :: OSI Approved :: MIT License',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3.9',
                   ])
