bolides
=======

**A package to analyze bolide data in Python.**

|pypi-badge| |rtd-badge|

.. |rtd-badge| image:: https://readthedocs.org/projects/bolides/badge/?version=latest
               :target: https://bolides.readthedocs.io/en/latest
               :alt: Documentation Status
.. |pypi-badge| image:: https://img.shields.io/pypi/v/bolides.svg
                :target: https://pypi.org/project/bolides/
                :alt: PyPI link

``bolides`` is a community package primarily designed to work with bolide detections from the Geostationary Lightning Mapper instruments aboard GOES-16 and GOES-17 that are published at `neo-bolide.ndc.nasa.gov <https://neo-bolide.ndc.nasa.gov>`_. But ``bolides`` can also read bolide data from ZODB database files produced by the GLM bolide detection pipeline, from `US Government sensors <https://cneos.jpl.nasa.gov/fireballs/>`_, and from your own .csv files containing bolide data.

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
- Run an `interactive webapp <https://bolides.aozerov.com>`_.

**Documentation**

All package documentation is hosted at `bolides.readthedocs.io <https://bolides.readthedocs.io>`_.

**Installation**

Steps 1 and 2 below can be skipped if you do not wish to use any plotting functions. However, you will still need to ``pip install shapely``.

#. First, `PROJ <https://proj.org/install.html>`_ needs to be installed on your system using the installation instructions at the link. `GEOS <https://libgeos.org/usage/install/>`_ also needs to be installed. On Ubuntu both can be installed with ``sudo apt install proj-bin libproj-dev libgeos-dev``.
#. Next, due to `known Cartopy and Shapely problems <https://github.com/SciTools/cartopy/issues/738>`_, Cartopy and Shapely need to be installed manually. This is done with: ``pip install numpy && pip install cartopy==0.18.0 shapely --no-binary cartopy --no-binary shapely``. Note that an older Cartopy version is recommended to prevent it from requiring PROJ versions not yet in common repositories. If you already have PROJ 8.0.0, you may omit the ``==0.18.0`` and install the latest Cartopy version.
#. ``bolides`` can be then be (hypothetically) installed with ``pip install bolides``. As the code on this branch is not on PyPI yet, it can be installed from TestPyPI with ``pip install -i https://test.pypi.org/simple/ bolides --extra-index-url https://pypi.org/simple/``

| Note: if you wish to install the repository in an editable mode, you can ``git clone`` this repository, then enter the repository's directory and run ``pip install -e .``
| As this is still not the main branch, you will have to do ``git checkout bdf-implementation`` in the repository to switch to this branch before running ``pip install -e .``

**Tutorial**

For a usage tutorial, go `here <https://bolides.readthedocs.io/en/latest/tutorials>`_.

**Historical Note**

The original version of ``bolides`` was developed by Clemens Rumpf and Geert Barentsen. It has been rewritten since then, but all of the original code and functionality is still present.

**Acknowledgments**

Work carried out for this project is supported by NASA’s Planetary Defense Coordination Office (PDCO).
Anthony Ozerov is supported through NASA Cooperative Agreement 80NSSC19M0089.
