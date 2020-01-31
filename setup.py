from setuptools import setup

setup(name='synthetic-spacecraft-imagery',
      version='0.1.0',
      description='Create synthetic training data using Blender',
      url='https://github.com/autognc/synthetic-spacecraft-imagery',
      author='Kevin Black',
      license='MIT',
      packages=['ssi'],
      install_requires=[
            'numpy~=1.16.1',
            'scipy~=1.3.1',
            'pillow~=6.1.0',
            'mathutils~=2.81.2'
      ],
      zip_safe=False)
