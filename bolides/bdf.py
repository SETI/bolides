import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from geopandas import GeoDataFrame
from shapely.geometry import Point
import numpy as np
from warnings import warn
import warnings
from cartopy import crs as ccrs
from datetime import datetime
from . import API_ENDPOINT_EVENTLIST, API_ENDPOINT_EVENT, MPLSTYLE
from lightkurve import LightCurve, LightCurveCollection
import pickle
from math import degrees
import ephem


class BolideDataFrame(GeoDataFrame):
    """
    Subclass of GeoPandas `~geopandas.GeoDataFrame` with additional bolide-specific methods.

    Parameters
    ----------
    source : str
        Specifies the source for the initialized. Can be either ``'website'``
        to initialize from neo-bolide-ndc.nasa.gov data, ``'pickle'`` to
        initialize from a pickled GeoDataFrame, or ``'pipeline'`` to initialize
        from ZODB database files from the pipeline.
    files : str, list
        For ``'pickle'``, specifies the filename of the pickled object.
        For ``'pipeline'``, specifies the filename(s) of the database file(s)
    """
    def __init__(self, source='website', files=None):

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
            coords = zip(lons, lats)
            points = [Point(coord[0], coord[1]) for coord in coords]
            init_gdf = GeoDataFrame(init_gdf, geometry=points, crs="EPSG:4326")

        elif source == 'pipeline':
            init_gdf = get_df_from_pipeline(files)
            init_gdf['source'] = 'pipeline'

        else:
            raise('Unknown source')

        super().__init__(init_gdf)

        self.annotate_bdf()

        from configparser import ConfigParser
        config = ConfigParser()
        config.read('bolides/desc.ini')
        self.descriptions = config['neo-bolide']

    def annotate_bdf(bdf):
        bdf['phase'] = [get_phase(dt) for dt in bdf['datetime']]
        bdf['moon_fullness'] = -np.abs(bdf['phase']-0.5)*2+1
        bdf['solarhour'] = [get_solarhour(data[0], data[1]) for data in zip(bdf['datetime'], bdf['longitude'])]

        sun_alt = []
        for num, row in bdf.iterrows():
            obs = ephem.Observer()
            obs.lon = str(row['longitude'])
            obs.lat = str(row['latitude'])
            obs.date = row['datetime']
            sun = ephem.Sun()
            sun.compute(obs)
            sun_alt.append(degrees(sun.alt))
        bdf['sun_alt'] = sun_alt

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
        if start is not None:
            to_drop = self.datetime < datetime.fromisoformat(start)
            new_bdf = self.drop(self.index[to_drop], inplace=inplace)
        if inplace:
            new_bdf = self
        if end is not None:
            to_drop = new_bdf.datetime > datetime.fromisoformat(end)
            new_bdf = new_bdf.drop(new_bdf.index[to_drop], inplace=inplace)
        if inplace:
            new_bdf = self
        if isinstance(new_bdf, GeoDataFrame):
            new_bdf.__class__ = BolideDataFrame
        return new_bdf

    def get_closest_by_time(self, datestr, n=1):
        dt = datetime.fromisoformat(datestr)
        return self.iloc[(self['datetime'] - dt).abs().argsort()].head(n)

    def get_closest_by_loc(self, lon, lat, n=1):
        lon_diff = (self['longitude'] - lon).abs()
        lat_diff = (self['latitude'] - lat).abs()
        tot_diff = lon_diff + lat_diff
        return self.iloc[tot_diff.argsort()].head(n)

    def clip_boundary(self, boundary=[], intersection=False):

        # put bdf into the correct CRS
        crs = self.geometry.crs
        bdf = self.to_crs("epsg:4236")
        from netCDF4 import Dataset
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        import pyproj

        # define Azimuthal Equidistant projection
        aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs

        polygons = []

        # for each item in the boundary, add it to the list of polygons in the
        # Azimuthal Equidistant CRS
        if 'goes-e' in boundary:

            # get data and create polygon
            fov = Dataset("data/GLM_FOV_edges.nc", "r", format="NETCDF4")
            lats = fov.variables['G16_fov_lat'][0]
            lons = fov.variables['G16_fov_lon'][0]
            goes_e = Polygon(zip(lons, lats))

            # use geopandas to re-project it
            gdf = GeoDataFrame(geometry=[goes_e], crs='epsg:4236')
            gdf = gdf.to_crs(aeqd)
            polygons.append(gdf.geometry[0])

        # for the GOES-W position, we can combine the regular and inverted FOV
        if 'goes-w' in boundary:

            # get data and create polygons
            fov = Dataset("data/GLM_FOV_edges.nc", "r", format="NETCDF4")
            lats = fov.variables['G17_fov_lat'][0]
            lons = fov.variables['G17_fov_lon'][0]
            goes_w = Polygon(zip(lons, lats))
            lats = fov.variables['G17_fov_lat_inverted'][0]
            lons = fov.variables['G17_fov_lon_inverted'][0]
            goes_w_i = Polygon(zip(lons, lats))

            # use geopandas to re-project them
            gdf = GeoDataFrame(geometry=[goes_w, goes_w_i], crs='epsg:4236')
            gdf = gdf.to_crs(aeqd)
            polygons.append(unary_union([gdf.geometry[0], gdf.geometry[1]]))

        # either take intersection of FOVs or the union to get a final polygon
        if intersection:
            final_polygon = polygons[0]
            for polygon in polygons:
                final_polygon = final_polygon.intersection(polygon)
        else:
            final_polygon = unary_union(polygons)

        # project bdf to Azimuthal Equidistant CRS
        bdf = bdf.to_crs(aeqd)
        # clip with the polygon, which is in Azimuthal Equidistant
        bdf = bdf.clip(final_polygon)
        # project bdf back to original CRS
        bdf = bdf.to_crs(crs)

        if isinstance(bdf, GeoDataFrame):
            bdf.__class__ = BolideDataFrame

        return bdf

    def plot_detections(self, crs=ccrs.AlbersEqualArea(central_longitude=-100), category=None,
                        coastlines=True, style=MPLSTYLE, boundary=['goes-w', 'goes-e'], boundary_style={}, **kwargs):
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
        warnings.filterwarnings("ignore", message="__len__ for multi-part")
        warnings.filterwarnings("ignore", message="Iteration over multi-part")

        import matplotlib.cm as cmx

        # default parameters put into kwargs if not specified by user
        defaults = {'marker': '.', 'color': 'red', 'cmap': plt.get_cmap('viridis')}
        if 'c' in kwargs:
            del defaults['color']
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        # get geopandas projection and reproject dataframe points
        crs_proj4 = crs.proj4_init
        bdf_proj = self.to_crs(crs_proj4)
        # filter out rows with no geometry
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
            if boundary:
                add_boundary(ax, boundary, boundary_style)

        return fig, ax

    def plot_dates(self, freq='1D', logscale=False, style=MPLSTYLE, start=None, end=None, **kwargs):
        """Plots the number of bolides over time."""
        from pandas.plotting import register_matplotlib_converters

        bdf = self.filter_date(start, end, inplace=False)

        register_matplotlib_converters()
        counts = bdf.groupby(pd.Grouper(key='datetime', freq=freq)).count()

        with plt.style.context(style):
            fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
            ax.xaxis.set_major_locator(mdates.YearLocator())
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))

            plt.tick_params(axis='x', which='minor', labelsize=5, pad=0)
            plt.tick_params(axis='x', which='major', pad=3)

            ax.set_xlabel("Date")
            if min(bdf.datetime).year == max(bdf.datetime).year:
                ax.set_xlabel("Date (month in "+str(min(bdf.datetime).year)+")")
            ax.set_ylabel("# Events")
            if 'width' not in kwargs:
                kwargs['width'] = max(100/len(counts), 1)
            ax.bar(counts.index, counts.iloc[:, 0], **kwargs)
            plt.xlim(min(bdf.datetime), max(bdf.datetime))
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
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # override GeoPandas' __getitem__ to force the class to remain BolideDataFrame
    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, GeoDataFrame):
            result.__class__ = BolideDataFrame
        return result

    # currently only supports augmenting GLM data with USG data
    def augment(self, source, files=None, intersection=False):
        new_data = BolideDataFrame(source=source, files=files)
        new_data['_id'] = ""
        for num, row in self.iterrows():
            deltas = row['datetime']-new_data['datetime']
            s_deltas = np.abs([delta.total_seconds() for delta in deltas])
            closest = np.argmin(s_deltas)
            if s_deltas[closest] < 300:
                lat_diff = abs(row['latitude']-new_data['latitude'][closest])
                lon_diff = abs(row['longitude'] % 360 - new_data['longitude'][closest] % 360)
                geo_diff = lat_diff+lon_diff
                s_diff = s_deltas[closest]
                score = geo_diff * s_diff
                if score < 5:
                    new_data['_id'][closest] = row['_id']
        if intersection:
            merged = self.merge(new_data, 'inner', on='_id')
        else:
            merged = self.merge(new_data, 'left', on='_id')
        merged['geometry'] = merged['geometry_x']
        merged['datetime'] = merged['datetime_x']
        merged['latitude'] = merged['latitude_x']
        merged['longitude'] = merged['longitude_x']
        merged = merged[[c for c in merged.columns if not c.endswith('_x')]]
        merged = merged[[c for c in merged.columns if not c.endswith('_y')]]
        merged.__class__ = BolideDataFrame
        merged.annotate_bdf()
        return merged


def get_df_from_website():

    # load data from website
    json = requests.get(API_ENDPOINT_EVENTLIST).json()

    # create DataFrame using JSON data
    df = pd.DataFrame(json['data'])
    df["datetime"] = df["datetime"].astype("datetime64")

    # create a list to be used as a geometry column
    lats = df['latitude']
    lons = df['longitude']
    coords = zip(lons, lats)
    points = [Point(coord[0], coord[1]) for coord in coords]

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
    df['datetime'] = [datetime.fromisoformat(date) for date in df['date']]
    df['energy'] = df['energy'].astype(float)

    # create a list to be used as a geometry column
    lats = df['latitude']
    lons = df['longitude']
    coords = zip(lons, lats)
    points = [Point(coord[0], coord[1]) for coord in coords]

    gdf = GeoDataFrame(df, geometry=points, crs="EPSG:4326")

    return gdf


def get_feature(feature, bDispObj):
    return [getattr(disp.features, feature) for disp in bDispObj.bolideDispositionProfileList]


def get_df_from_pipeline(files):
    import bolide_dispositions.BolideDispositions.from_bolideDatabase as bDisp_from_db
    if type(files) is str:
        bDispObj = bDisp_from_db(files, verbosity=True, useRamDisk=False)
    else:
        bDispObj = bDisp_from_db(files[0], verbosity=True, useRamDisk=False)
        for i in range(1, len(files)):
            bDispObj = bDisp_from_db(files[i], extra_bolideDispositionProfileList=bDispObj,
                                     verbosity=True, useRamDisk=False)

    lon = get_feature('avgLon', bDispObj)
    lat = get_feature('avgLat', bDispObj)
    sat = get_feature('goesSatellite', bDispObj)
    dur = get_feature('timeDuration', bDispObj)
    dat = get_feature('bolideTime', bDispObj)
    confidence = [disp.machineOpinions[0].bolideBelief for disp in bDispObj.bolideDispositionProfileList]

    coords = zip(lon, lat)
    points = [Point(coord[0], coord[1]) for coord in coords]
    bdf = GeoDataFrame({'detectedBy': sat, 'latitude': lat, 'longitude': lon, 'datetime': dat,
                        'duration': dur, 'confidence': confidence}, geometry=points, crs="EPSG:4326")

    return bdf


def get_phase(datetime):
    date = ephem.Date(datetime)
    nnm = ephem.next_new_moon(date)
    pnm = ephem.previous_new_moon(date)

    lunation = (date-pnm)/(nnm-pnm)
    return lunation


def get_solarhour(datetime, lon):
    o = ephem.Observer()
    o.date = datetime
    from math import pi
    o.long = lon/180 * pi
    sun = ephem.Sun()
    sun.compute(o)
    hour_angle = o.sidereal_time() - sun.ra
    solarhour = ephem.hours(hour_angle+ephem.hours('12:00')).norm/(2*pi) * 24
    return solarhour


def add_boundary(ax, boundary, boundary_style):

    # for filename in boundary:
    #     with open('data/'+filename+'.pkl', 'rb') as f:
    #         b = pickle.load(f)
    #     ax.add_geometries([b[0]], crs=b[1], facecolor='none',
    #                       edgecolor='k', alpha=1, linewidth=3)

    boundary_defaults = {"facecolor": "none"}
    for key, value in boundary_defaults.items():
        if key not in boundary_style:
            boundary_style[key] = value
    from netCDF4 import Dataset
    from shapely.geometry import LinearRing
    lines = []
    if 'goes-e' in boundary:
        fov = Dataset("data/GLM_FOV_edges.nc", "r", format="NETCDF4")
        lats = fov.variables['G16_fov_lat'][0]
        lons = fov.variables['G16_fov_lon'][0]
        lines.append(LinearRing(zip(lons, lats)))
    if 'goes-w' in boundary:
        fov = Dataset("data/GLM_FOV_edges.nc", "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat'][0]
        lons = fov.variables['G17_fov_lon'][0]
        lines.append(LinearRing(zip(lons, lats)))
        lats = fov.variables['G17_fov_lat_inverted'][0]
        lons = fov.variables['G17_fov_lon_inverted'][0]
        lines.append(LinearRing(zip(lons, lats)))
    ax.add_geometries(lines, crs=ccrs.PlateCarree(), **boundary_style)


# def make_point(point):
#     lon = point['location']['coordinates'][0]
#     lat = point['location']['coordinates'][1]
#     return Point(lon, lat)

# def get_gdf(geodata):
#     points = [make_point(point) for point in geodata]
#     lat = [point['location']['coordinates'][1] for point in geodata]
#     energy = [point['energy'] for point in geodata]
#     time = [point['time'] for point in geodata]
#     gdf = GeoDataFrame({'time':time,'energy':energy}, geometry=points, crs='epsg:4326')
#     return gd
