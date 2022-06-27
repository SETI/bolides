.. _api.bdf:

===============
BolideDataFrame
===============
.. currentmodule:: bolides

The `BolideDataFrame` class is an extension of GeoPandas' `~geopandas.GeoDataFrame` object, which extends Pandas' `~pandas.DataFrame`.

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
    BolideDataFrame.clip_boundary
    BolideDataFrame.filter_observation

Plotting
~~~~~~~~

.. autosummary::
   :toctree: api/

   BolideDataFrame.plot_detections
   BolideDataFrame.plot_density
   BolideDataFrame.plot_dates
