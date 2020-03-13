from setuptools import setup, find_packages

setup(name='starfish',
      version='0.1.0',
      description='Create synthetic training data using Blender',
      url='https://github.com/autognc/starfish',
      author='Kevin Black',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy~=1.16.1',
          'scipy~=1.3.1',
          'pillow~=6.2.0',
          'mathutils~=2.81.2'
      ],
      zip_safe=False)
