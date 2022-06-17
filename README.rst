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

**Tutorial**

For a usage tutorial, go to notebooks/tutorial.ipynb
