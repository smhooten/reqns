from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='reqns',
   version='0.2',
   description='A versatile, parallelized LED and laser rate equations dynamic solver',
   license="BSD-2-Clause",
   long_description=long_description,
   author='Sean Hooten',
   author_email='shooten@eecs.berkeley.edu',
   url="http://github.com/smhooten/reqns",
   packages=['reqns'],
   install_requires=['numpy', 'scipy', 'fdint', 'mpi4py'],
   zip_safe=True,
)



#package_data={'reqns':['data/*']},
