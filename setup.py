from setuptools import setup, find_packages

setup(name='starfish',
      version='0.1.0',
      description='Create synthetic training data using Blender',
      url='https://github.com/autognc/starfish',
      author='Kevin Black',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=[
          'numpy~=1.16.1',
          'opencv-python~=4.2.0',
          'mathutils~=2.81.2'
      ],
      zip_safe=False)
