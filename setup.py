#!/usr/bin/env python
import os
import sys
from setuptools import setup

# Prepare and send a new release to PyPI
if "release" in sys.argv[-1]:
    os.system("python setup.py sdist")
    os.system("twine upload dist/bolides*")
    os.system("rm -rf dist/bolides*")
    sys.exit()

# Load the __version__ variable without importing the package already
exec(open('bolides/version.py').read())

setup(name='bolides',
      version=__version__,
      description="Inspect things that go KABOOM. 💥",
      long_description=open('README.rst').read(),
      author='Clemens Rumpf & Geert Barentsen',
      author_email='hello@geert.io',
      license='BSD',
      packages=['bolides'],
      include_package_data=True,
      # See http://www.sphinx-doc.org/en/stable/theming.html#distribute-your-theme-as-a-python-package
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Intended Audience :: Developers",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          ],
      )
