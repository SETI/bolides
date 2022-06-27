.. _api.bdf:

===============
BolideDataFrame
===============

.. currentmodule:: bolides

The `BolideDataFrame` class is the main way to work with data in the ``bolides`` package.
`BolideDataFrame` is an extension of GeoPandas' `~geopandas.GeoDataFrame` object, which extends Pandas' `~pandas.DataFrame`.

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   BolideDataFrame

Filtering
~~~~~~~~~

.. autosummary::
   :toctree: api/

    BolideDataFrame.filter_date
    BolideDataFrame.filter_boundary
    BolideDataFrame.filter_observation

Search
~~~~~~

.. autosummary::
   :toctree: api/

    BolideDataFrame.get_closest_by_time
    BolideDataFrame.get_closest_by_loc

Plotting
~~~~~~~~

.. autosummary::
   :toctree: api/

   BolideDataFrame.plot_detections
   BolideDataFrame.plot_density
   BolideDataFrame.plot_dates

Data Manipulation
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: api/

   BolideDataFrame.augment
