from setuptools import setup

setup(
    name='pointnet',
    version='0.1',
    packages=['pointnet', 'pointnet.utils', 'pointnet.models', 'pointnet.sem_seg',
              'pointnet.part_seg'],
    package_dir={'': 'src'},
    url='https://github.com/charlesq34/pointnet',
    license='MIT',
    author='Tuomas Airaksinen',
    author_email='tuomas.airaksinen@vrt.fi',
    description=''
)
