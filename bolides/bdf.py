import os
from datetime import datetime, timedelta
import requests
from warnings import warn, filterwarnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import pickle

from geopandas import GeoDataFrame

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lightkurve import LightCurve, LightCurveCollection

from . import API_ENDPOINT_EVENT, MPLSTYLE, ROOT_PATH
from .utils import reconcile_input
from .sources import glm_website, usg, pipeline, gmn, csv, remote

_FIRST_COLS = ['datetime', 'longitude', 'latitude', 'source', 'detectedBy',
               'confidenceRating', 'confidence', 'lightcurveStructure',
               'energy', 'energy_g16', 'energy_g17', 'brightness_g16', 'brightness_g17',
               'brightness_cat_g16', 'brightness_cat_g17', 'impact-e', 'alt', 'vel']


class BolideDataFrame(GeoDataFrame):
    """
    Subclass of GeoPandas `~geopandas.GeoDataFrame` with additional bolide-specific methods.

    Parameters
    ----------
    source : str
        Specifies the source for the initialized. Can be:

        - ``'glm'``: initialize from neo-bolide-ndc.nasa.gov data
        - ``'glm-pipeline'`` to initialize from ZODB database files from the GLM detection pipeline.
        - ``'pickle'``: initialize from a pickled GeoDataFrame
        - ``'csv'``: initialize from a .csv file
        - ``'usg'``: initialize from US Government data at cneos.jpl.nasa.gov/fireballs/
        - ``'gmn'``: initialize from Global Meteor Networ data at globalmeteornetwork.org/data/
    files : str, list

        Specifies files to be used depending on source.

        - For ``'pickle'``, specifies the filename of the pickled object.
        - For ``'csv'``, specifies the filename of the csv.
        - For ``'glm-pipeline'``, specifies the filename(s) of the database file(s)
    url : str
        Specifies remote url to be used with ``'remote'``.
    date : str, date, datetime
        Specifies a date when using Global Meteor Network data.
        str can be either yyyy-mm or yyyy-mm-dd.

    annotate : bool
        Whether or not to add additional metadata.
    rearrange : bool
        Whether or not to rearrange the columns, if coming from a CSV or Pickle.
    """
    def __init__(self, *args, **kwargs):

        if len(args)==0 and len(kwargs)==0:
            kwargs['source'] = 'glm'
        if 'source' not in kwargs:
            return super().__init__(*args, **kwargs)

        defaults = {'files': None, 'annotate': True, 'rearrange': False}
        kwargs = reconcile_input(kwargs, defaults)
        source = kwargs['source']
        files = kwargs['files']
        annotate = kwargs['annotate']
        rearrange = kwargs['rearrange']

        # input standardization
        source = source.lower()
        if type(files) is str:
            files = [files]

        # input validation
        valid_sources = ['website', 'glm', 'usg', 'pickle', 'gmn', 'csv', 'pipeline',
                         'glm-pipeline', 'usg-orbits', 'glm-orbits', 'remote']
        if source not in valid_sources:
            raise ValueError("Source \""+str(source)+"\" is unsupported. Please use one of "+str(valid_sources))

        if source in ['pickle', 'csv', 'glm-pipeline', 'pipeline'] and files is None:
            raise ValueError("Files must be specified for the given source \""+source+"\"")

        if source in ['pickle', 'csv', 'glm-pipeline', 'pipeline'] and not all([os.path.isfile(f) for f in files]):
            paths = "\n".join([os.path.abspath(f) for f in files])
            raise ValueError("At least one of the files specified does not exist.\
                             Here are their absolute paths:\n"+paths)

        if source in ['pickle', 'csv'] and len(files) > 1:
            warn("More than one file given for source \""+source+"\". Only the first one will be used.")

        # Initialize differently based on source.
        # Each if statement creates a GeoDataFrame with the EPSG:4326 CRS

        if source in ['website', 'glm']:
            source = 'glm'
            init_gdf = glm_website()

        elif source == 'usg':
            init_gdf = usg()

        elif source == 'gmn':
            if 'date' not in kwargs:
                raise ValueError('Must specify a yyyy-mm or yyyy-mm-dd date to use GMN data')
            init_gdf = gmn(kwargs['date'], loc_mode='begin')

        elif source == 'pickle':
            with open(files[0], 'rb') as pkl:
                init_gdf = pickle.load(pkl)

        elif source == 'csv':
            init_gdf = csv(files[0])

        elif source == 'remote':
            if 'url' not in kwargs:
                raise ValueError('Must specify a url with url=...')
            url = kwargs['url']
            init_gdf = remote(url)

        elif source == 'usg-orbits':
            init_gdf = remote('https://aozerov.com/data/usg-orbits.csv')
            annotate = False

        # elif source == 'glm-orbits':
        #     init_gdf = csv(file=ROOT_PATH+'/../notebooks/glm-orbits.csv')
        #     annotate = False

        elif source in ['glm-pipeline', 'pipeline']:
            source = 'glm-pipeline'
            if 'min_confidence' in kwargs:
                min_confidence = kwargs['min_confidence']
            else:
                min_confidence = 0
            init_gdf = pipeline(files=files, min_confidence=min_confidence)

        init_gdf['source'] = source

        # rearrange columns, respecting original order if csv or pickle
        if source not in ['csv', 'pickle'] or rearrange is True:
            first_cols = [col for col in _FIRST_COLS if col in init_gdf.columns]
            other_cols = [col for col in init_gdf.columns if col not in first_cols]
            init_gdf = init_gdf[first_cols + other_cols]

        # initialize the super-class (GeoDataFrame) using the created init_gdf
        super().__init__(init_gdf)

        # add additional metadata to the bolides
        if annotate:
            self.annotate()

        descriptions = pd.read_csv(ROOT_PATH+'/metadata/columns.csv', index_col='column')
        self.descriptions = descriptions[descriptions.sources.isin([source, 'all'])]

    def annotate(self):
        """Add metadata to bolide detections"""
        from .astro_utils import get_phase, get_solarhour, get_sun_alt

        # lunar phase
        self['phase'] = [get_phase(dt) for dt in self['datetime']]
        # moon fullness
        self['moon_fullness'] = -np.abs(self['phase']-0.5)*2+1
        # solar hour
        self['solarhour'] = [get_solarhour(data[0], data[1]) for data in zip(self['datetime'], self['longitude'])]
        # solar altitude
        sun_alt = np.array([get_sun_alt(dt=row['datetime'].to_pydatetime(), lat=row['latitude'], lon=row['longitude']) for _, row in self.iterrows()])
        self['sun_alt_obs'] = sun_alt[:, 0]
        self['sun_alt_app'] = sun_alt[:, 1]

        self['date_retrieved'] = datetime.now()

    def describe(self, key=None):
        """Describe a variable of the BolideDataFrame

        Parameters
        ----------
        key : str, iterable, or None
            The variables to be described.
            If str, describes that variable.
            If iteralbe, describe all variable strings in the iterable.
            If None, describe all variables in the BolideDataFrame.

        Returns
        -------
        str :
            A string describing the variable(s).
        """

        if type(key) is str:
            key = [key]
        to_describe = self.columns if key is None else key
        if 'source' in self.columns and len(self)>0:
            source = self['source'].iloc[0]
        else:
            source = ''
        sources = ['all', source]
        for column in to_describe:
            description = self.descriptions['description'][column] if column in self.descriptions.index else ""
            print(column+":", description)

    def filter_date(self, start=None, end=None, inplace=False):
        """Filter bolides by date.

        Filters the `~BolideDataFrame` using dates given in ISO format.
        `start` or `end` can be left unspecified to only bound the dates on one end.

        Parameters
        ----------
        start, end: str
            ISO-format strings that can be read by `~datetime.datetime.fromisoformat`.
        inplace: bool
            If True, the `~BolideDataFrame` of this method is altered. If False, it is not, so the returned
            `~BolideDataFrame` must be used.

        Returns
        -------
        `~BolideDataFrame`
        """
        new_bdf = self

        # drop data before start date, if specified
        if start is not None:
            to_drop = self.datetime < datetime.fromisoformat(start)
            new_bdf = self.drop(self.index[to_drop], inplace=inplace)
        if inplace:
            new_bdf = self
        # drop data after end date, if specified
        if end is not None:
            to_drop = new_bdf.datetime > datetime.fromisoformat(end)
            new_bdf = new_bdf.drop(new_bdf.index[to_drop], inplace=inplace)
        if inplace:
            new_bdf = self

        # force the class, which gets converted to GeoDataFrame at some point
        # in the drop methods above
        force_bdf_class(new_bdf)

        return new_bdf

    def get_closest_by_time(self, datestr, n=1):
        """Get the n bolides closest to a given iso-format date string

        Parameters
        ----------
        datestr : str
            ISO-format string that can be read by `~datetime.datetime.fromisoformat`.
        n : int
            Number of closest detections to return.

        Returns
        -------
        bdf : `~BolideDataFrame`
            The filtered data.

        """

        dt = datetime.fromisoformat(datestr)
        return self.iloc[(self['datetime'] - dt).abs().argsort()].head(n)

    def get_closest_by_loc(self, lon, lat, n=1):
        """Get the n bolides closest to a given longitude and latitude

        Parameters
        ----------
        lon, lat : int
            Longitude and latitude to search for bolides near.
        n : int
            Number of closest detections to return.

        Returns
        -------
        bdf : `~BolideDataFrame`
            The filtered data.

        """
        lon_diff = (self['longitude'] - lon).abs()
        lat_diff = (self['latitude'] - lat).abs()
        tot_diff = lon_diff + lat_diff
        return self.iloc[tot_diff.argsort()].head(n)

    def filter_boundary(self, boundary, intersection=False, interior=True):
        """Filter data to only points within specified boundaries.

        Parameters
        ----------
        boundary: str or list of str
            The boundary (boundaries) to filter the `~BolideDataFrame` by.
            Refer to `~bolides.fov_utils.get_boundary`.
        intersection: bool
            - True: keep detections within the union of all boundaries given.
            - False: keep detections within the intersection of all boundaries given.
        interior: bool
            - True: default behavior.
            - False: keep detections outside the union or intersection of the boundaries, instead of inside.

        Returns
        -------
        `~BolideDataFrame`
            The filtered `~BolideDataFrame`
        """

        import pyproj
        # define Azimuthal Equidistant projection
        aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs

        # put bdf into the correct CRS
        crs = self.geometry.crs
        bdf = self.to_crs(aeqd)

        from .fov_utils import get_boundary
        final_polygon = get_boundary(boundary, intersection=intersection, collection=False)

        # clip with the polygon, which is in Azimuthal Equidistant
        points_in_poly = [pt.within(final_polygon) for pt in bdf['geometry']]
        if interior:
            bdf = bdf[points_in_poly]
        else:
            bdf = bdf[~points_in_poly]
        # project bdf back to original CRS
        bdf = bdf.to_crs(crs)

        force_bdf_class(bdf)

        return bdf

    def filter_observation(self, sensors, intersection=False):
        """Filter data to only points observable (in time and space) by a given sensor.

        Parameters
        ----------
        sensors: str or list of str
            The sensor(s) to filter the `~BolideDataFrame` by:

            - ``'GLM-16'``: The Geostationary Lightning Mapper aboard GOES-16.
            - ``'GLM-17'``: The Geostationary Lightning Mapper aboard GOES-17.
              Note that the biannual yaw flips of GOES-17 are taken into account.
              The shorthand ``'G16'`` and ``'G17'`` can also be used.

        intersection: bool
            If True, filter for bolides observable by all sensors given.
            If False, filter for bolides observable by any sensors given.

        Returns
        -------
        `~BolideDataFrame`
            The filtered `~BolideDataFrame`
        """

        # input standardization and validation
        if type(sensors) is str:
            sensors = [sensors]
        if type(sensors) is not list or not all([type(s) is str for s in sensors]):
            raise ValueError("Sensors must be a string or list of strings")
        sensors = [sensor.lower() for sensor in sensors]

        valid_sensors = ['g16', 'g17', 'glm-16', 'glm-17']

        indices = []
        for num, sensor in enumerate(sensors):
            filename = sensor
            if sensor in ['g16', 'glm-16']:
                filename = ROOT_PATH + '/data/glm16_obs.csv'
            elif sensor in ['g17', 'glm-17']:
                filename = ROOT_PATH + '/data/glm17_obs.csv'
            else:
                raise ValueError("Invalid sensor \""+sensor+"\". sensors must be in "+str(valid_sensors))

            bdfs = []

            # read csv defining the observation times and FOV
            fov_df = pd.read_csv(filename)

            # for each observation time + FOV, filter_date by the observation
            # times, filter_boundary by the FOV, and append result to the list
            # of BolideDataFrames
            for i in range(len(fov_df)):
                start = None if fov_df.start.isna()[i] else fov_df.start[i]
                end = None if fov_df.end.isna()[i] else fov_df.end[i]
                boundary = fov_df.boundary[i]
                bdfs.append(self.filter_boundary([boundary]).filter_date(start=start, end=end))
            bdf = pd.concat(bdfs)
            if not intersection:
                indices += list(bdf.index)
            elif num == 0:
                indices = list(bdf.index)
            else:
                indices = [idx for idx in indices if idx in list(bdf.index)]

        # get a list of unique indices, and sort them
        indices = list(set(indices))
        indices.sort()

        # get the final filtered BolideDataFrame and return
        filtered = self.loc[indices, :]
        force_bdf_class(filtered)
        return filtered

    def filter_shower(self, shower=None, years=None, padding=1, sdf=None, exclude=False):
        """Filter data to only points observable (in time and space) by a given sensor.

        Parameters
        ----------
        shower: str, int, list of str, or list of int
            The meteor shower(s) to filter by. Can enter either IAU number,
            IAU 3-letter code, or full shower name. Refer to the IAU Meteor
            Data center at https://www.ta3.sk/IAUC22DB/MDC2007/.
        years: int or list of int
            The shower years to include. Default is to filter for all occurrences
            of the shower(s) in the BolideDataFrame
        padding: int
            The number of days to pad around the peak time.
        sdf: ShowerDataFrame
            Optionally provide a ShowerDataFrame to use for filtering
        exclude: bool
            Whether or not to exclude bolides around the given showers. Default is to include.

        Returns
        -------
        `~BolideDataFrame`
            The filtered `~BolideDataFrame`
        """
        if years is None:
            years = list(range(min(self.datetime).year-1, max(self.datetime).year+1))
        elif type(years) is int:
            years = [years]

        if sdf is None:
            if hasattr(self, '_showers'):
                sdf = self._showers
            else:
                from . import ShowerDataFrame
                sdf = ShowerDataFrame()
                self._showers = sdf

        dates = sdf.get_dates(shower, years).datetime
        date_padding = timedelta(days=padding)
        date_ranges = [[d-date_padding, d+date_padding] for d in dates]

        counts = np.sum(np.array([list(self.datetime.between(d[0], d[1])) for d in date_ranges]), axis=0)
        if not exclude:
            good_locs = counts != np.zeros(len(counts))
        else:
            good_locs = counts == np.zeros(len(counts))
        return self[good_locs]

    def plot_detections(self, crs=None,
                        category=None, coastlines=True, style=MPLSTYLE,
                        boundary=None, boundary_style={}, figsize=(8, 8),
                        **kwargs):
        """Plot detections of bolides.

        Reprojects the geometry of bdf to the crs given, and scatters the points
        on a cartopy map. kwargs are passed through to matplotlib's scatter.

        Parameters
        ----------
        crs : `~cartopy.crs.CRS`
            The map projection to use. Refer to
            https://scitools.org.uk/cartopy/docs/latest/reference/projections.html.
        boundary : str or list of str
            The boundaries to plot.
            Refer to `~bolides.fov_utils.get_boundary`.
        category : str
            The name of a categorical column in the `~BolideDataFrame`
        **kwargs :
            Keyword arguments passed through to `~matplotlib.pyplot.scatter`.

        Other Parameters
        ----------------
        coastlines: bool
            Whether or not to draw coastlines.
        style : str
            The matplotlib style to use. Refer to
            https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        boundary_style : dict
            The kwargs to use when plotting the boundary.
            Refer to `~cartopy.mpl.geoaxes.GeoAxes.add_geometries`.
        figsize : tuple
            The size (width, height) of the plotted figure.

        Returns
        -------
        fig : `~matplotlib.pyplot.figure`
        ax : `~cartopy.mpl.geoaxes.GeoAxesSubplot`
        """
        from .crs import DefaultCRS
        if crs is None:
            crs = DefaultCRS()

        # The cartopy library used by plot_detections currently has many
        # warnings about the shapely library deprecating things...
        # This code suppresses those warnings
        filterwarnings("ignore", message="__len__ for multi-part")
        filterwarnings("ignore", message="Iteration over multi-part")

        import matplotlib.cm as cmx

        # get geopandas projection and reproject dataframe points
        crs_proj4 = crs.proj4_init
        bdf_proj = self.to_crs(crs_proj4)
        # filter out rows with no geometry

        # default parameters put into kwargs if not specified by user
        defaults = {'marker': '.', 'color': 'red', 'cmap': plt.get_cmap('viridis')}
        if 'c' in kwargs:
            del defaults['color']
        kwargs = reconcile_input(kwargs, defaults)

        good_locs = ~bdf_proj.geometry.is_empty
        x = np.empty(len(bdf_proj))
        y = np.empty(len(bdf_proj))
        points = bdf_proj[good_locs]['geometry']
        x[good_locs] = np.array([p.x for p in points])
        y[good_locs] = np.array([p.y for p in points])

        # using the given style,
        with plt.style.context(style):

            # generate Figure and GeoAxes with the given proejction
            fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=figsize)

            ax.stock_img()  # plot background map

            # scatter points, passing arguments through

            if category is None:  # if there is no categorical variable specified
                cb = plt.scatter(x, y, **kwargs)
                # if color is determined by a quantitative variable, we add a colorbar
                if 'c' in kwargs:
                    plt.colorbar(cb, label=kwargs['c'].name)

            else:  # if there is a categorical variable specified, color points using it
                unique = self[category].unique()  # get the unique values of the variable
                import matplotlib.colors as colors
                hot = plt.get_cmap('tab10')
                cNorm = colors.Normalize(vmin=0, vmax=len(unique))
                scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

                del kwargs['color']  # color kwarg being overridden by categorical variable
                s = kwargs['s'] if 's' in kwargs else None

                # for each unique category, scatter the data with the right color
                for num, label in enumerate(unique):
                    idx = self[category] == label
                    if s is not None and hasattr(s, '__getitem__'):
                        kwargs['s'] = s[idx]
                    ax.scatter(x[idx], y[idx], color=scalarMap.to_rgba(num), label=label, **kwargs)
                plt.legend()

            if coastlines:
                ax.coastlines()  # plot coastlines

            # plot sensor FOV
            from .fov_utils import add_boundary
            if boundary:
                add_boundary(ax, boundary, boundary_style)

        return fig, ax

    def plot_interactive(self, *args, **kwargs):
        """Plot an interactive map of bolide detections.

        Parameters
        ----------
        mode : str
            Either ``'earth'`` or ``'radiant'``.
            ``'earth'`` plots locations on the Earth,
            ``'radiant'`` plots locations in the sky.
        projection : str
            The map projection to use. Here is the complete list:
            [``'eckert4'``, ``'goes-e'``, ``'goes-w'``, ``'fy4a'``,
            ``'airy'``, ``'aitoff'``, ``'albers'``, ``'albers usa'``, ``'august'``,
            ``'azimuthal equal area'``, ``'azimuthal equidistant'``, ``'baker'``,
            ``'bertin1953'``, ``'boggs'``, ``'bonne'``, ``'bottomley'``, ``'bromley'``,
            ``'collignon'``, ``'conic conformal'``, ``'conic equal area'``, ``'conic equidistant'``,
            ``'craig'``, ``'craster'``, ``'cylindrical equal area'``,
            ``'cylindrical stereographic'``, ``'eckert1'``, ``'eckert2'``,
            ``'eckert3'``, ``'eckert4'``, ``'eckert5'``, ``'eckert6'``, ``'eisenlohr'``,
            ``'equirectangular'``, ``'fahey'``, ``'foucaut'``, ``'foucaut sinusoidal'``,
            ``'ginzburg4'``, ``'ginzburg5'``, ``'ginzburg6'``,
            ``'ginzburg8'``, ``'ginzburg9'``, ``'gnomonic'``, ``'gringorten'``,
            ``'gringorten quincuncial'``, ``'guyou'``, ``'hammer'``, ``'hill'``,
            ``'homolosine'``, ``'hufnagel'``, ``'hyperelliptical'``,
            ``'kavrayskiy7'``, ``'lagrange'``, ``'larrivee'``, ``'laskowski'``,
            ``'loximuthal'``, ``'mercator'``, ``'miller'``, ``'mollweide'``, ``'mt flat polar parabolic'``,
            ``'mt flat polar quartic'``, ``'mt flat polar sinusoidal'``,
            ``'natural earth'``, ``'natural earth1'``, ``'natural earth2'``,
            ``'nell hammer'``, ``'nicolosi'``, ``'orthographic'``,
            ``'patterson'``, ``'peirce quincuncial'``, ``'polyconic'``,
            ``'rectangular polyconic'``, ``'robinson'``, ``'satellite'``, ``'sinu mollweide'``,
            ``'sinusoidal'``, ``'stereographic'``, ``'times'``,
            ``'transverse mercator'``, ``'van der grinten'``, ``'van der grinten2'``,
            ``'van der grinten3'``, ``'van der grinten4'``,
            ``'wagner4'``, ``'wagner6'``, ``'wiechel'``, ``'winkel tripel'``,
            ``'winkel3'``]
        boundary : str or list of str
            The boundaries to plot.
            Refer to `~bolides.fov_utils.get_boundary`.
        color : str
            The name of a column in the `~BolideDataFrame`
        logscale : bool
            Whether or not to use a logarithmic scale for the colors.
        **kwargs :
            Keyword arguments passed through to `~plotly.express.scatter_geo`.

        Other Parameters
        ----------------
        coastlines: bool
            Whether or not to draw coastlines.
        style : str
            The matplotlib style to use. Refer to
            https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        boundary_style : dict
            The kwargs to use when plotting the boundary.
            Refer to `~cartopy.mpl.geoaxes.GeoAxes.add_geometries`.
        figsize : tuple
            The size (width, height) of the plotted figure.
        culture : str
            When mode is ``'radiant'``, the asterism source culture to use. Here is the complete list:
            [``'anutan'``, ``'aztec'``, ``'belarusian'``, ``'boorong'``,
             ``'chinese'``, ``'chinese_contemporary'``, ``'chinese_medieval'``,
             ``'egyptian'``, ``'hawaiian_starlines'``, ``'indian'``,
             ``'inuit'``, ``'japanese_moon_stations'``, ``'korean'``,
             ``'lokono'``, ``'macedonian'``, ``'maori'``, ``'maya'``,
             ``'mongolian'``, ``'navajo'``, ``'norse'``, ``'northern_andes'``,
             ``'romanian'``, ``'russian_siberian'``, ``'sami'``,
             ``'sardinian'``, ``'tongan'``, ``'tukano'``, ``'tupi'``,
             ``'western'``, ``'western_SnT'``, ``'western_hlad'``,
             ``'western_rey'``]


        Returns
        -------
        fig : `~matplotlib.pyplot.figure`
        ax : `~cartopy.mpl.geoaxes.GeoAxesSubplot`
        """
        from .plotting import plot_interactive
        return plot_interactive(self, *args, **kwargs)

    def plot_density(self, crs=None,
                     bandwidth=5, coastlines=True, style=MPLSTYLE,
                     boundary=None, boundary_style={},
                     kde_params={}, lat_resolution=100, lon_resolution=50,
                     n_levels=100, figsize=(8, 8), **kwargs):
        """Plot bolide detection density.

        Density is computed using scikit-learn's `~sklearn.neighbors.KernelDensity`
        using the haversine distance metric (as the data is in longitude and latitude)
        and gaussian kernel by default.
        It is then gridded, projected, and plotted.

        Parameters
        ----------
        crs : `~cartopy.crs.CRS`
            The map projection to use. Refer to
            https://scitools.org.uk/cartopy/docs/latest/reference/projections.html.
        bandwidth : float
            The bandwidth of the Kernel Density Estimator, in degrees.
        boundary : str or list of str
            The boundaries to plot and clip the density by.
            Refer to `~bolides.fov_utils.get_boundary`.
        n_levels : int
            Number of discrete density levels to plot.
        lat_resolution, lon_resolution : ints
            The number of discrete latitude and longitude levels when gridding the density.
        **kwargs :
            Keyword arguments passed through to `~cartopy.mpl.geoaxes.GeoAxes.contourf`.

        Other Parameters
        ----------------
        coastlines : bool
            Whether or not to draw coastlines.
        style : str
            The matplotlib style to use. Refer to
            https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        boundary_style : dict
            The kwargs to use when plotting the boundary.
            Refer to `~cartopy.mpl.geoaxes.GeoAxes.add_geometries`.
        kde_params : dict
            The kwargs to pass to `~sklearn.neighbors.KernelDensity`.
            Note that 'metric' is not allowed to be specified, as haversine
            is the only valid metric.
        figsize : tuple
            The size (width, height) of the plotted figure.

        Returns
        -------
        fig : `~matplotlib.pyplot.figure`
        ax : `~cartopy.mpl.geoaxes.GeoAxesSubplot`
        """
        from .crs import DefaultCRS
        if crs is None:
            crs = DefaultCRS()

        from cartopy import crs as ccrs
        # The cartopy library used by plot_density currently has many
        # warnings about the shapely library deprecating things...
        # This code suppresses those warnings
        filterwarnings("ignore", message="__len__ for multi-part")
        filterwarnings("ignore", message="Iteration over multi-part")

        # filter to only detections containing both latitude and longitude
        bdf = self[(~self.latitude.isnull()) & (~self.longitude.isnull())]

        # get numpy array in the format that KernelDensity likes
        data = np.vstack([np.radians(bdf.latitude), np.radians(bdf.longitude)]).T

        # set kde_params and validate input
        if 'kernel' not in kde_params:
            kde_params['kernel'] = 'gaussian'
        if 'metric' in kde_params:
            del kde_params['metric']
            warn('Please do not specify metric. Any metric other than haversine (default)\
                 will lead to invalid results.')

        from math import radians
        from sklearn.neighbors import KernelDensity
        # create and fit a KDE
        kde = KernelDensity(bandwidth=radians(bandwidth), metric="haversine", **kde_params)
        kde.fit(data)

        # create grid of latitudes and longitudes
        from math import pi
        X, Y = np.meshgrid(np.linspace(-pi, pi, lat_resolution),
                           np.linspace(-pi/2, pi/2, lon_resolution))
        xy = np.vstack([Y.ravel(), X.ravel()]).T

        # compute density at gridpoints
        density_per_steradian = np.exp(kde.score_samples(xy))

        # convert density per steradian to bolides per sqkm
        steradian_per_sqdeg = 1/3282.80635
        earth_sqkm = 510 * 10**6
        earth_sqdeg = 41252.96
        sqdeg_per_sqkm = earth_sqdeg / earth_sqkm
        num_bolides = len(self)
        bolides_per_steradian = num_bolides * density_per_steradian
        bolides_per_sqkm = bolides_per_steradian * steradian_per_sqdeg * sqdeg_per_sqkm

        # prepare for plotting
        Z = bolides_per_sqkm
        Z = Z.reshape(X.shape)
        levels = np.linspace(0, Z.max(), n_levels)
        x = np.degrees(X)
        y = np.degrees(Y)
        z = Z

        # get mask given the boundary
        from .fov_utils import get_mask
        mask = get_mask(np.degrees(xy), boundary).reshape(X.shape)

        # default parameters put into kwargs if not specified by user
        default_cmap = plt.get_cmap('viridis').copy()
        default_cmap.set_under('none')
        defaults = {'alpha': 1, 'antialiased': False, 'cmap': default_cmap}
        kwargs = reconcile_input(kwargs, defaults)

        # using the given style,
        with plt.style.context(style):

            # generate Figure and GeoAxes with the given proejction
            fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=figsize)

            ax.stock_img()  # plot background map

            # plot contour, passing arguments through
            filled_c = ax.contourf(x, y, z*mask, levels=levels[1:],
                                   transform=ccrs.PlateCarree(), **kwargs)

            # make lines invisible
            for c in filled_c.collections:
                c.set_edgecolor('none')
                c.set_linewidth(0.000000000001)

            if coastlines:
                ax.coastlines()  # plot coastlines

            # plot sensor FOV
            from .fov_utils import add_boundary
            if boundary:
                add_boundary(ax, boundary, boundary_style)

            plt.colorbar(filled_c, alpha=kwargs['alpha'],
                         label='bolide density (km$^{-2}$)')

        return fig, ax

    def plot_dates(self, freq='1D', logscale=False,
                   start=None, end=None,
                   figsize=(10, 3), style=MPLSTYLE,
                   showers=None, line_style={}, **kwargs):
        """Plot the number of bolides over time.

        Parameters
        ----------
        freq : str
            The binning frequency. Can be strings like ``'2D'``, ``'1M'``, etc.
            Refer to https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        start, end: strs
            Optional arguments to filter by date.
            Must be ISO-format strings that can be read by `~datetime.datetime.fromisoformat`.
        logscale : bool
            Whether or not to use a log-scale for the y-axis
        **kwargs :
            Keyword arguments passed through to `~matplotlib.axes.Axes.bar`.

        Other Parameters
        ----------------
        style : str
            The matplotlib style to use. Refer to
            https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
        showers : ShowerDataFrame
            A ShowerDataFrame to use for the shower data. By default, showers are pulled
            from the established showers list at the IAU meter data center.
        figsize : tuple
            The size (width, height) of the plotted figure.
        line_style : dict
            The matplotlib style arguments for the vertical shower lines.

        Returns
        -------
        fig : `~matplotlib.pyplot.figure`
        ax : `~matplotlib.axes.Axes`
        """
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()

        # filter date to given start and end times
        bdf = self.filter_date(start, end, inplace=False)
        datetimes = bdf.datetime

        # get counts according to the given frequency
        index = bdf.groupby(pd.Grouper(key='datetime', freq=freq)).count().index

        if showers is not None:

            # a ShowerDataFrame is stored in the _showers attribute, so that
            # users can make multiple plots without constantly recreating it
            if hasattr(self, '_showers'):
                sdf = self._showers
            else:
                from . import ShowerDataFrame
                sdf = ShowerDataFrame()
                self._showers = sdf
            date_info = sdf.get_dates(showers, range(min(bdf.datetime).year-1, max(bdf.datetime).year+1))
            shower_names = date_info['shower name']
            shower_dates = date_info.datetime

        # with the given matplotlib style,
        with plt.style.context(style):

            # initialize figure and set major, minor ticks and formats
            fig, ax = plt.subplots(figsize=figsize, dpi=300)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))

            # make minor ticks (months) small, pad major ticks (years)
            plt.tick_params(axis='x', which='minor', labelsize=5, pad=0, labeltop=True)
            plt.tick_params(axis='x', which='major', pad=3, labeltop=True)

            # set x-label, adding year data if data is only within one year
            # (and hence no year data on x-axis)
            ax.set_xlabel("Date")
            if min(bdf.datetime).year == max(bdf.datetime).year:
                ax.set_xlabel("Date (month in "+str(min(bdf.datetime).year)+")")
            ax.set_ylabel("# Events")

            plt.hist(datetimes, bins=index, **kwargs)

            # set x-limits to data limits
            x_min = datetime.fromisoformat(start) if start is not None else min(bdf.datetime)
            x_max = datetime.fromisoformat(end) if end is not None else max(bdf.datetime)
            plt.xlim(x_min, x_max)

            if showers is not None:
                names = list(shower_names)
                unique = list(shower_names.unique())
                import matplotlib.cm as cm
                cmap = cm.Set2

                defaults = {'linestyle': 'dashed', 'linewidth': 1,
                            'alpha': 0.5}
                line_kwargs = reconcile_input(line_style, defaults)

                for name, date in zip(shower_names, shower_dates):
                    plt.axvline(x=date, color=cmap(unique.index(name)/8),
                                label=name, **line_kwargs)
                # remove extra labels for repeat showers
                for i, p in enumerate(ax.get_lines()):
                    if p.get_label() in names[:i]:
                        p.set_label('_' + p.get_label())
                plt.legend(loc='upper right')

            # make y-axis logscale if specified
            if logscale:
                ax.set_yscale('log')

        return fig, ax

    def add_website_data(self, ids=None):
        """Pull light curve data from neo-bolide.ndc.nasa.gov.

        Downloads light curve data from neo-bolide.ndc.nasa.gov, placing it
        into the BolideDataFrame as a column of `~lightkurve.LightCurveCollection` objects.
        The column is named ``'lightcurves'``

        Parameters
        ----------
        ids : list of str
            Optional list of strings representing the bolide ID's that light curves are wanted for.
            If not used, a light curve is added for every bolide in the BolideDataFrame.
        """

        lclist = []
        for num, row in tqdm(self.iterrows(), "Downloading data", total=len(self)):  # for each bolide

            # if a subset of ids was specified that excludes this row, skip.
            if ids is not None and row['_id'] not in ids:
                continue

            # pull data from website
            data = requests.get(API_ENDPOINT_EVENT + row['_id']).json()['data'][0]['attachments']
            row_lcs = []

            # create a LightCurve for each attachment in the data
            for attachment in data:
                geodata = attachment['geoData']
                flux = [point['energy'] for point in geodata]
                time = [point['time']/1000 for point in geodata]
                from astropy.time import Time
                time_obj = Time(time, format='unix')
                lc = LightCurve(time=time_obj, flux=flux)
                lc.meta['MISSION'] = attachment['platformId']
                lc.meta['LABEL'] = attachment['platformId']
                row_lcs.append(lc)
            lclist.append(LightCurveCollection(row_lcs))
        self['lightcurves'] = lclist

    # TODO: match on a column other than _id?
    def augment(self, new_data, time_limit=300, score_limit=5, intersection=False, outer=False):
        """Augment BolideDataFrame with data from another source

        Parameters
        ----------
        new_data : `~BolideDataFrame`
            The source of the new data.
        time_limit : int
            Maximum time difference (seconds) for two bolides to possibly be the same
        score_limit : int
            Maximum score (time_limit * difference in latlon) for two bolides to
            possibly be the same.
        intersection : bool
            - True: only keep detections that appear in both sets of data.
            - False: keep all detections that appear in the first set.
        outer : bool
            - True: keep detections from both sets of data.
            - False: keep all detections that appear in the first set.

        Returns
        -------
        bdf : `~BolideDataFrame`
        """

        # make _id column if doesn't exist
        if "_id" not in self.columns:
            self["_id"] = np.arange(len(self))

        # iterate through rows of self, computing distance and identifying
        # detections with detections in the other data if certain closeness
        # criteria based on time and distance are met
        ids = [""] * len(new_data)
        for num, row in tqdm(self.iterrows(), "Augmenting data", total=len(self)):
            deltas = row['datetime']-new_data['datetime']
            s_deltas = np.abs([delta.total_seconds() for delta in deltas])
            closest = np.argmin(s_deltas)
            if s_deltas[closest] < time_limit:
                lat_diff = abs(row['latitude']-new_data['latitude'].iloc[closest])
                lon_diff = abs(row['longitude'] % 360 - new_data['longitude'].iloc[closest] % 360)
                geo_diff = lat_diff+lon_diff
                s_diff = s_deltas[closest]
                score = geo_diff * s_diff
                if score < score_limit:
                    ids[closest] = row['_id']
        new_data['_id'] = ids
        # merge the data. If taking the intersection, only keep rows in
        # original BolideDataFrame that have a corresponding detection in
        # the other source
        if intersection:
            merged = self.merge(new_data, 'inner', on='_id')
        elif outer:
            merged = self.merge(new_data, 'outer', on='_id')
        else:
            merged = self.merge(new_data, 'left', on='_id')

        # rename the required columns
        for col in merged.columns:
            if col.endswith('_x') and col[:-2]+"_y" in merged.columns:
                merged[col[:-2]] = merged[col]
                del merged[col]

        # force class
        merged.__class__ = BolideDataFrame

        return merged

    def __setattr__(self, attr, val):
        # since BolideDataFrame can have non-column attributes, we can ignore
        # Pandas' warnings about setting attributes
        filterwarnings("ignore", message="Pandas doesn't allow columns to be created via a new attribute name")
        return super().__setattr__(attr, val)

    # override GeoPandas' __getitem__ to force the class to remain BolideDataFrame
    def __getitem__(self, key):
        result = super().__getitem__(key)
        force_bdf_class(result)
        return result

    @property
    def _constructor(self):
        return BolideDataFrame

    # define how the BolideDataFrame renders in an IPython notebook
    def _repr_html_(self):

        # want to show all columns
        with pd.option_context('display.max_columns', None):
            # get the default representation from Pandas
            df_rep = super()._repr_html_()

        good_sources = ['glm', 'usg', 'glm-pipeline']

        attribution = ""

        # get the attribution HTML file
        if 'source' in self.columns and len(self) > 0:
            source = self['source'].iloc[0]
            if source in good_sources:
                with open(ROOT_PATH + '/metadata/' + source + '.html', 'r') as f:
                    attribution = f.read()
            if 'source_y' in self.columns and self['source_y'].iloc[0] in good_sources:
                source = self['source_y'].iloc[0]
                attribution += "Augmented with "
                with open(ROOT_PATH + '/metadata/' + source + '.html', 'r') as f:
                    attribution += f.read()
        return attribution + df_rep


def force_bdf_class(bdf):
    if isinstance(bdf, GeoDataFrame):
        bdf.__class__ = BolideDataFrame
