from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='reqns',
   version='0.1',
   description='A versatile, parallelized LED and laser rate equations dynamic solver',
   license="BSD-2-Clause",
   long_description=long_description,
   author='Sean Hooten',
   author_email='shooten@eecs.berkeley.edu',
   url="http://github.com/smhooten/reqns",
   packages=['reqns'],
   install_requires=['numpy', 'scipy', 'fdint', 'mpi4py'],
   zip_safe=False
   scripts=[
            'scripts/TCSPC.py'
           ]
)



#package_data={'reqns':['data/*']},
