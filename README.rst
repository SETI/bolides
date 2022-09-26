=======
bolides
=======

**A package to analyze bolide data in Python.**

|pypi-badge| |rtd-badge| |binder-badge|

.. |rtd-badge| image:: https://readthedocs.org/projects/bolides/badge/?version=latest
               :target: https://bolides.readthedocs.io/en/latest
               :alt: Documentation Status
.. |pypi-badge| image:: https://img.shields.io/pypi/v/bolides.svg
                :target: https://pypi.org/project/bolides/
                :alt: PyPI link
.. |binder-badge| image:: https://mybinder.org/badge_logo.svg
                :target: https://mybinder.org/v2/gh/jcsmithhere/bolides/master?labpath=notebooks%2Ftutorial.ipynb
                :alt: Binder link

.. image:: https://raw.githubusercontent.com/jcsmithhere/bolides/master/docs/preview.gif

``bolides`` is a community package primarily designed to work with bolide detections from the Geostationary Lightning Mapper instruments aboard GOES-16 and GOES-17 that are published at `neo-bolide.ndc.nasa.gov <https://neo-bolide.ndc.nasa.gov>`_. But ``bolides`` can also read bolide data from ZODB database files produced by the GLM bolide detection pipeline, from `US Government sensors <https://cneos.jpl.nasa.gov/fireballs/>`_, and from your own .csv files containing bolide data.

Functionality
=============

``bolides`` puts bolide detections from various sources into a common BolideDataFrame format. With this, ``bolides`` can do things like:

- Automatically add metadata like lunar phase, solar time, and solar altitude to bolide detections.
- Filter the data sets by any variable.
- Search the data sets for particular bolides by time or location.
- Make histograms of bolides over time.
- Plot detections on arbitrary map projections, coloring by any categorical or quantitative variables.
- Plot GLM fields-of-view in the GOES-West and GOES-East positions, and filter bolide detections by the FOV.
- Augment one data set with data from another, automatically matching bolide detections from different sources.
- Pull corresponding bolide light curves from `neo-bolide.ndc.nasa.gov <https://neo-bolide.ndc.nasa.gov>`_ and plot them.
- Pull meteor shower data from the `IAU Meteor Data Center <https://www.ta3.sk/IAUC22DB/MDC2007/>`_ and plot their orbits.
- Run an `interactive webapp <https://bolides.seti.org>`_.

.. end-before-here

Documentation
=============

All package documentation is hosted at `bolides.readthedocs.io <https://bolides.readthedocs.io>`_.

Installation
============

.. installation-start

We want to make installation as easy as possible. If you have any problems with the installation process, please `open an issue <https://github.com/jcsmithhere/bolides/issues/new/choose>`_.

Dependencies
------------

If using Conda, just use Conda to install cartopy: ``conda install -c conda-forge cartopy``.

If using pip:

#. First, `PROJ <https://proj.org/install.html>`_ needs to be installed on your system using the installation instructions at the link. `GEOS <https://libgeos.org/usage/install/>`_ also needs to be installed. On Ubuntu both can be installed with ``sudo apt install proj-bin libproj-dev libgeos-dev``
#. Next, due to `known Cartopy and Shapely problems <https://github.com/SciTools/cartopy/issues/738>`_, Cartopy and Shapely need to be installed manually. This is done with: ``pip install numpy && pip install cartopy==0.18.0 shapely --no-binary cartopy --no-binary shapely``. Note that an older Cartopy version is recommended to prevent it from requiring PROJ versions not yet in common repositories. If you already have PROJ 8.0.0, you may omit the ``==0.18.0`` and install the latest Cartopy version.


The package
-----------

While the package is still in development we recommend installing from source:

#. Download this repository: ``git clone https://github.com/jcsmithhere/bolides.git``
#. Move into it: ``cd bolides``
#. Install the package in editable mode: ``pip install -e .``

Once this package is on the Python Package Index, you can install via pip using ``pip install bolides``.

.. installation-end

Tutorial
========

For a usage tutorial, go `here <https://bolides.readthedocs.io/en/latest/tutorials>`_. An interactive version is hosted on `binder <https://mybinder.org/v2/gh/jcsmithhere/bolides/master?labpath=notebooks%2Ftutorial.ipynb>`_.

.. start-after-here

Historical Note
===============

The original version of ``bolides`` was developed by Clemens Rumpf and Geert Barentsen. It has been rewritten since then, but all of the original code and functionality is still present.

Acknowledgments
===============

This development is supported through NASA's Asteroid Threat Assessment Project (ATAP), which is funded through NASA's Planetary Defense Coordination Office (PDCO).
Anthony Ozerov is supported through NASA Cooperative Agreement 80NSSC19M0089.
