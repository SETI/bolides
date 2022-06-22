bolides
=======

**A package to analyze bolide data in Python.**

``bolides`` is primarily designed to work with bolide detections from the Geostationary Lightning Mapper instruments aboard GOES-16 and GOES-17 that are published at `neo-bolide.ndc.nasa.gov <https://neo-bolide.ndc.nasa.gov>`_. But ``bolides`` can also read bolide data from ZODB database files produced by the GLM bolide detection pipeline, from `US Government sensors <https://cneos.jpl.nasa.gov/fireballs/>`_, and from your own .csv files containing bolide data.

**Functionality**

``bolides`` puts bolide detections from various sources into a common BolideDataFrame format. With this, ``bolides`` can do things like:

- Automatically add metadata like lunar phase, solar time, and solar altitude to bolide detections.
- Filter the data sets by any variable.
- Search the data sets for particular bolides by time or location.
- Make histograms of bolides over time.
- Plot detections on arbitrary map projections, coloring by any categorical or quantitative variables.
- Plot GLM fields-of-view in the GOES-West and GOES-East positions, and filter bolide detections by the FOV.
- Augment one data set with data from another, automatically matching bolide detections from different sources.
- Pull corresponding bolide light curves from `neo-bolide.ndc.nasa.gov <https://neo-bolide.ndc.nasa.gov>`_ and plot them.

**Installation**

#. First, `PROJ <https://proj.org/install.html>`_ needs to be installed on your system using the installation instructions at the link. On Ubuntu this can be done with ``sudo apt install proj-bin``
#. Next, due to `known Cartopy and Shapely problems <https://github.com/SciTools/cartopy/issues/738>`_, Cartopy and Shapely need to be installed manually. This is done with: ``pip install numpy && pip install cartopy==0.18.0 shapely --no-binary cartopy --no-binary shapely``. Note that an older Cartopy version is recommended to prevent it from requiring PROJ versions not yet in common repositories.
#. ``bolides`` can be then be (hypothetically) installed with ``pip install bolides``. As the code on this branch is not on PyPI yet, it can be installed from TestPyPI with ``pip install -i https://test.pypi.org/simple/ bolides --extra-index-url https://pypi.org/simple/``

Note: if you wish to install the repository in an editable mode, you can ``git clone`` this repository, then enter the repository's directory and run ``pip install -e .``

**Tutorial**

For a usage tutorial, go to notebooks/tutorial.ipynb
