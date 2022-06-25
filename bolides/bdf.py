import requests
from datetime import datetime
from warnings import warn, filterwarnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import pickle

from geopandas import GeoDataFrame
from cartopy import crs as ccrs

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lightkurve import LightCurve, LightCurveCollection

from . import API_ENDPOINT_EVENTLIST, API_ENDPOINT_EVENT, MPLSTYLE, ROOT_PATH
from .constants import GLM_STEREO_MIDPOINT
from .utils import make_points


class BolideDataFrame(GeoDataFrame):
    """
    Subclass of GeoPandas `~geopandas.GeoDataFrame` with additional bolide-specific methods.

    Parameters
    ----------
    source : str
        Specifies the source for the initialized. Can be either ``'website'``
        to initialize from neo-bolide-ndc.nasa.gov data, ``'pickle'`` to
        initialize from a pickled GeoDataFrame, ``'csv'`` to initialize from a
        .csv file, ``'usg'`` to initialize from US Government data at
        cneos.jpl.nasa.gov/fireballs/, or ``'pipeline'`` to initialize from
        ZODB database files from the pipeline.
    files : str, list
        For ``'pickle'``, specifies the filename of the pickled object.
        For ``'csv'``, specifies the filename of the csv.
        For ``'pipeline'``, specifies the filename(s) of the database file(s)
    """
    def __init__(self, source='website', files=None):

        # Initialize differently based on source.
        # Each if statement creates a GeoDataFrame with the EPSG:4326 CRS

        if source == 'website':
            init_gdf = get_df_from_website()
            init_gdf['source'] = 'website'

        elif source == 'usg':
            init_gdf = get_df_from_usg()
            init_gdf['source'] = 'usg'

        elif source == 'pickle':
            if type(files) is not list:
                files = [files]
            if len(files) > 1:
                warn("More than one file given. Only the first is used.")
            with open(files[0], 'rb') as pkl:
                init_gdf = pickle.load(pkl)

        elif source == 'csv':
            if type(files) is not list:
                files = [files]
            if len(files) > 1:
                warn("More than one file given. Only the first is used.")
            init_gdf = pd.read_csv(files[0], index_col=0, parse_dates=['datetime'], keep_default_na=False)
            lats = init_gdf['latitude']
            lons = init_gdf['longitude']
            points = make_points(lons, lats)
            init_gdf = GeoDataFrame(init_gdf, geometry=points, crs="EPSG:4326")

        elif source == 'pipeline':
            init_gdf = get_df_from_pipeline(files)
            init_gdf['source'] = 'pipeline'

        else:
            raise('Unknown source '+str(source))

        cols = list(init_gdf.columns.values)
        first_cols = ['datetime', 'longitude', 'latitude']
        first_cols.reverse()
        for col in first_cols:
            col_idx = cols.index(col)
            del cols[col_idx]
            cols.insert(0, col)
        init_gdf = init_gdf[cols]

        # initialize the super-class (GeoDataFrame) using the created init_gdf
        super().__init__(init_gdf)

        # add additional metadata to the bolides
        self.annotate()

        from configparser import ConfigParser
        config = ConfigParser()
        config.read(ROOT_PATH+'/desc.ini')
        self.descriptions = config['neo-bolide']

    def annotate(bdf):
        """Add metadata to bolide detections"""
        from .astro_utils import get_phase, get_solarhour, get_sun_alt

        # lunar phase
        bdf['phase'] = [get_phase(dt) for dt in bdf['datetime']]
        # moon fullness
        bdf['moon_fullness'] = -np.abs(bdf['phase']-0.5)*2+1
        # solar hour
        bdf['solarhour'] = [get_solarhour(data[0], data[1]) for data in zip(bdf['datetime'], bdf['longitude'])]
        # solar altitude
        bdf['sun_alt'] = [get_sun_alt(row) for _, row in bdf.iterrows()]

        bdf['date_retrieved'] = datetime.now()

    def describe(self, key=None):
        if type(key) is str:
            key = [key]
        to_describe = self.columns if key is None else key
        for column in to_describe:
            description = self.descriptions[column] if column in self.descriptions else ""
            print(column, ":", description)

    def filter_date(self, start=None, end=None, inplace=False):
        """Filter bolides by date.

        Filters the BolideDataFrame using dates given in ISO format.

        Parameters
        ----------
        start, end: str
            ISO-format strings that can be read by `~datetime.datetime.fromisoformat`.
        inplace: bool
            If True, the BolideDataFrame of this method is altered. If False, it is not, so the returned
            BolideDataFrame must be used.

        Returns
        -------
        new_bdf: BolideDataFrame"""
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
        """Get the n bolides closest to a given iso-format date string"""
        dt = datetime.fromisoformat(datestr)
        return self.iloc[(self['datetime'] - dt).abs().argsort()].head(n)

    def get_closest_by_loc(self, lon, lat, n=1):
        """Get the n bolides closest to a given longitude and latitude"""
        lon_diff = (self['longitude'] - lon).abs()
        lat_diff = (self['latitude'] - lat).abs()
        tot_diff = lon_diff + lat_diff
        return self.iloc[tot_diff.argsort()].head(n)

    def clip_boundary(self, boundary=None, intersection=False, interior=True):
        """Filter data to only points within specified boundaries"""
        assert boundary is not None

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

    def filter_observation(self, sensors=None):

        indices = []
        for sensor in sensors:
            filename = sensor
            if sensor == 'glm16':
                filename = ROOT_PATH + '/data/glm16_obs.csv'
            if sensor == 'glm17':
                filename = ROOT_PATH + '/data/glm17_obs.csv'
            bdfs = []
            fov_df = pd.read_csv(filename)
            for i in range(len(fov_df)):
                start = None if fov_df.start.isna()[i] else fov_df.start[i]
                end = None if fov_df.end.isna()[i] else fov_df.end[i]
                boundary = fov_df.boundary[i]
                bdfs.append(self.clip_boundary([boundary]).filter_date(start=start, end=end))
            bdf = pd.concat(bdfs)
            indices += list(bdf.index)

        indices = list(set(indices))
        indices.sort()
        filtered = self.loc[indices, :]
        force_bdf_class(filtered)
        return filtered

    def plot_detections(self, crs=ccrs.AlbersEqualArea(central_longitude=GLM_STEREO_MIDPOINT),
                        category=None, coastlines=True, style=MPLSTYLE,
                        boundary=None, boundary_style={}, figsize=(8, 8),
                        **kwargs):
        """Plot detections of bolides.

        Reprojects the geometry of bdf to the crs given, and scatters the points
        on a cartopy map. kwargs are passed through to matplotlib's scatter.

        Parameters
        ----------
        crs : cartopy Coordinate Reference System
        category: str
            The name of a categorical column in the BolideDataFrame
        coastlines: bool
        style : str

        Returns
        -------
        fig : Matplotlib Figure
        ax : Cartopy GeoAxesSubplot
        """
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
            kwargs['c'] = kwargs['c'][~bdf_proj.geometry.is_empty]
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        bdf_proj = bdf_proj[~bdf_proj.geometry.is_empty]

        # get x,y lists
        points = bdf_proj['geometry']
        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])

        # using the given style,
        with plt.style.context(style):

            # generate Figure and GeoAxes with the given proejction
            fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(8, 8))

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
                    if s is not None:
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

    def plot_density(self, crs=ccrs.AlbersEqualArea(central_longitude=GLM_STEREO_MIDPOINT),
                     bandwidth=5, coastlines=True, style=MPLSTYLE,
                     boundary=None, boundary_style={},
                     kde_params={}, lat_resolution=100, lon_resolution=50,
                     n_levels=100, figsize=(8, 8), **kwargs):

        # The cartopy library used by plot_density currently has many
        # warnings about the shapely library deprecating things...
        # This code suppresses those warnings
        filterwarnings("ignore", message="__len__ for multi-part")
        filterwarnings("ignore", message="Iteration over multi-part")

        from sklearn.neighbors import KernelDensity
        from math import pi

        bdf = self[~self.latitude.isnull()]

        data = np.vstack([np.radians(bdf.latitude), np.radians(bdf.longitude)]).T
        if 'kernel' not in kde_params:
            kde_params['kernel'] = 'gaussian'

        from math import radians
        kde = KernelDensity(bandwidth=radians(bandwidth), metric="haversine", **kde_params)
        kde.fit(data)

        X, Y = np.meshgrid(np.linspace(-pi, pi, lat_resolution),
                           np.linspace(-pi/2, pi/2, lon_resolution))
        xy = np.vstack([Y.ravel(), X.ravel()]).T

        density_per_steradian = np.exp(kde.score_samples(xy))

        # convert density per steradian to bolides per sqkm
        steradian_per_sqdeg = 1/3282.80635
        earth_sqkm = 510 * 10**6
        earth_sqdeg = 41252.96
        sqdeg_per_sqkm = earth_sqdeg / earth_sqkm
        num_bolides = len(self)
        bolides_per_steradian = num_bolides * density_per_steradian
        bolides_per_sqkm = bolides_per_steradian * steradian_per_sqdeg * sqdeg_per_sqkm

        Z = bolides_per_sqkm
        Z = Z.reshape(X.shape)
        levels = np.linspace(0, Z.max(), n_levels)

        x = np.degrees(X)
        y = np.degrees(Y)
        z = Z

        from .fov_utils import get_mask
        mask = get_mask(np.degrees(xy), boundary).reshape(X.shape)

        # default parameters put into kwargs if not specified by user
        default_cmap = plt.get_cmap('viridis').copy()
        default_cmap.set_under('none')
        defaults = {'alpha': 1, 'antialiased': False, 'cmap': default_cmap}
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

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

    def plot_dates(self, freq='1D', logscale=False, style=MPLSTYLE, start=None, end=None, **kwargs):
        """Plots the number of bolides over time."""
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()

        # filter date to given start and end times
        bdf = self.filter_date(start, end, inplace=False)

        # get counts according to the given frequency
        counts = bdf.groupby(pd.Grouper(key='datetime', freq=freq)).count()

        # with the given matplotlib style,
        with plt.style.context(style):

            # initialize figure and set major, minor ticks and formats
            fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))

            # make minor ticks (months) small, pad major ticks (years)
            plt.tick_params(axis='x', which='minor', labelsize=5, pad=0)
            plt.tick_params(axis='x', which='major', pad=3)

            # set x-label, adding year data if data is only within one year
            # (and hence no year data on x-axis)
            ax.set_xlabel("Date")
            if min(bdf.datetime).year == max(bdf.datetime).year:
                ax.set_xlabel("Date (month in "+str(min(bdf.datetime).year)+")")
            ax.set_ylabel("# Events")

            # adjust width according to number of bars, if not specified
            if 'width' not in kwargs:
                kwargs['width'] = max(100/len(counts), 1)

            # make bar plot, passing kwargs through
            ax.bar(counts.index, counts.iloc[:, 0], **kwargs)
            # set x-limits to data limits
            plt.xlim(min(bdf.datetime), max(bdf.datetime))

            # make y-axis logscale if specified
            if logscale:
                ax.set_yscale('log')

        return fig, ax

    def add_website_data(self, ids=None):
        # import json
        lclist = []
        for num, row in self.iterrows():  # for each bolide

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

    def to_pickle(self, filename):
        """Dump BolideDataFrame to a pickle at specified filename"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # override GeoPandas' __getitem__ to force the class to remain BolideDataFrame
    def __getitem__(self, key):
        result = super().__getitem__(key)
        force_bdf_class(result)
        return result

    # TODO: match on a column other than _id?
    def augment(self, new_data, time_limit=300, score_limit=5, intersection=False):
        """Augment BolideDataFrame with data from another source"""

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
                lat_diff = abs(row['latitude']-new_data['latitude'][closest])
                lon_diff = abs(row['longitude'] % 360 - new_data['longitude'][closest] % 360)
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


def get_df_from_website():

    # load data from website
    json = requests.get(API_ENDPOINT_EVENTLIST).json()

    # create DataFrame using JSON data
    df = pd.DataFrame(json['data'])
    df["datetime"] = df["datetime"].astype("datetime64")

    # add bolide energy data
    energies_g16 = []
    energies_g17 = []
    for ats in df.attachments:
        e_g16 = 0
        e_g17 = 0
        for at in ats:
            platform = at['platformId']
            energy = at['energy']
            if platform == 'G16':
                e_g16 += energy
            if platform == 'G17':
                e_g17 += energy
        if e_g16 == 0:
            e_g16 = np.nan
        if e_g17 == 0:
            e_g17 = np.nan
        energies_g16.append(e_g16)
        energies_g17.append(e_g17)
    df['energies_g16'] = energies_g16
    df['energies_g17'] = energies_g17

    # create a list to be used as a geometry column
    lats = df['latitude']
    lons = df['longitude']
    points = make_points(lons, lats)

    # create GeoDataFrame using DataFrame and the geometry.
    # EPSG:4326 because data is in lon-lat format.
    gdf = GeoDataFrame(df, geometry=points, crs="EPSG:4326")

    return gdf


def get_df_from_usg():

    # load data from website
    json = requests.get('https://ssd-api.jpl.nasa.gov/fireball.api').json()
    data = json['data']
    cols = json['fields']

    # create DataFrame
    df = pd.DataFrame(data, columns=cols)
    df['latitude'] = df['lat'].astype(float) * ((df['lat-dir'] == 'N') * 2 - 1)
    df['longitude'] = df['lon'].astype(float) * ((df['lon-dir'] == 'E') * 2 - 1)
    del df['lat'], df['lon'], df['lat-dir'], df['lon-dir']
    df['datetime'] = [datetime.fromisoformat(date) for date in df['date']]
    del df['date']
    df['energy'] = df['energy'].astype(float)
    df['vel'] = df['vel'].astype(float)

    # create a list to be used as a geometry column
    lats = df['latitude']
    lons = df['longitude']
    points = make_points(lons, lats)

    gdf = GeoDataFrame(df, geometry=points, crs="EPSG:4326")

    return gdf


def get_df_from_pipeline(files, min_confidence=0):

    from .pipeline_utils import dict_from_zodb
    dict_of_lists = dict_from_zodb(files=files, min_confidence=min_confidence)

    # create Point objects
    lon = dict_of_lists['avgLon']
    lat = dict_of_lists['avgLat']
    points = make_points(lon, lat)
    bdf = GeoDataFrame(dict_of_lists, geometry=points, crs="EPSG:4326")
    column_translation = {'avgLon': 'longitude', 'avgLat': 'latitude', 'bolideTime': 'datetime',
                          'timeDuration': 'duration', 'goesSatellite': 'detectedBy'}
    bdf = bdf.rename(columns=column_translation)

    return bdf


def force_bdf_class(bdf):
    if isinstance(bdf, GeoDataFrame):
        bdf.__class__ = BolideDataFrame
