from distutils.core import setup

"""
Navigate to this directory in a python terminal &
Run the below command to install

pip install --editable .

(make sure to include the trailling '.')
"""

setup(
    name='aspen',
    version='1.0.0',
    packages=['tform', 'signals'],
    url='',
    license='',
    author='jp1757',
    author_email='james.m.r.peter@gmail.com',
    description='General purpose financial model back-test suite'
)
