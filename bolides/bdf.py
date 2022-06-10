import requests
from bolides.bolidelist import BolideList
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from geopandas import GeoDataFrame
from shapely.geometry import Point
import numpy as np
import warnings
from cartopy import crs as ccrs
from datetime import datetime
from . import API_ENDPOINT_EVENT
from lightkurve import LightCurve, LightCurveCollection
import pickle


class BolideDataFrame(GeoDataFrame):
    def __init__(self, source='website', files=None):
        if source == 'website':
            init_gdf = get_df_from_website()
        elif source == 'pickle':
            if type(files) is not list:
                files = [files]
            if len(files) > 1:
                warnings.warn("More than one file given. Only the first is used.")
            with open(files[0], 'rb') as pkl:
                init_gdf = pickle.load(pkl)
        elif source == 'pipeline':
            init_gdf = get_df_from_pipeline(files)
        else:
            raise('Unknown source')

        super().__init__(init_gdf)

        self.annotate_bdf()

    def annotate_bdf(bdf):
        bdf['phase'] = [get_phase(dt) for dt in bdf['datetime']]
        bdf['moon_fullness'] = -np.abs(bdf['phase']-0.5)*2+1
        bdf['solarhour'] = [get_solarhour(data[0], data[1]) for data in zip(bdf['datetime'], bdf['longitude'])]

    def filter_date_after(self, datestring):
        to_drop = self.datetime >= datetime.fromisoformat(datestring)
        self.drop(self.index[~to_drop], inplace=True)

    def filter_date_before(self, datestring):
        to_drop = self.datetime <= datetime.fromisoformat(datestring)
        self.drop(self.index[~to_drop], inplace=True)

    def filter_date_between(self, start, end):
        to_drop = self.datetime.between(datetime.fromisoformat(start), datetime.fromisoformat(end))
        self.drop(self.index[~to_drop], inplace=True)

    def plot_detections(self, crs=ccrs.LambertCylindrical(central_longitude=-100), **kwargs):
        """Plot detections of bolides.

        Reprojects the geometry of bdf to the crs given, and scatters the points
        on a cartopy map. args and kwargs are passed through to matplotlib's
        scatter.

        Parameters
        ----------
        crs : cartopy Coordinate Reference System
        bdf : GeoDataFrame
            GeoDataFrame containing bolides as used throughout this package.

        Returns
        -------
        fig : Matplotlib figure
        """

        # get geopandas projection and reproject dataframe points
        crs_proj4 = crs.proj4_init
        bdf_proj = self.to_crs(crs_proj4)

        fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(15, 15))

        ax.stock_img()  # plot background map

        # get x,y lists
        points = bdf_proj['geometry']
        x = [p.x for p in points]
        y = [p.y for p in points]

        defaults = {'marker': '.', 'color': 'red'}
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value

        # scatter points, passing arguments through
        ax.scatter(x, y, **kwargs)
        return fig, ax

    def plot_dates(self, freq='1D', **kwargs):
        """Plots the number of bolides over time."""
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        counts = self.groupby(pd.Grouper(key='datetime', freq=freq)).count()
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=90)
        ax.set_xlabel("date")
        ax.set_ylabel("# Events")
        ax.bar(counts.index, counts._id, **kwargs)
        return fig, ax

    def add_website_data(self, ids=None):
        # import json
        lclist = []
        for num, row in self.iterrows():

            # if a subset of ids was specified that excludes this row, skip.
            if ids is not None and row['_id'] not in ids:
                continue
            # with open('test_data/'+row['_id']+'.json') as f:
            #     data = json.load(f)['data'][0]['attachments']
            data = requests.get(API_ENDPOINT_EVENT + row['_id']).json()['data'][0]['attachments']
            row_lcs = []
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

    def __getitem__(self, key):
        result = super().__getitem__(key)
        if isinstance(result, GeoDataFrame):
            result.__class__ = BolideDataFrame
        return result


def get_df_from_website():
    bl = BolideList()
    bdf = bl.to_pandas()
    lats = bdf['latitude']
    lons = bdf['longitude']
    coords = zip(lons, lats)
    points = [Point(coord[0], coord[1]) for coord in coords]
    bdf = GeoDataFrame(bdf, geometry=points, crs="EPSG:4326")

    return bdf


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
    import ephem
    date = ephem.Date(datetime)
    nnm = ephem.next_new_moon(date)
    pnm = ephem.previous_new_moon(date)

    lunation = (date-pnm)/(nnm-pnm)
    return lunation


def get_solarhour(datetime, lon):
    import ephem
    o = ephem.Observer()
    o.date = datetime
    from math import pi
    o.long = lon/180 * pi
    sun = ephem.Sun()
    sun.compute(o)
    hour_angle = o.sidereal_time() - sun.ra
    solarhour = ephem.hours(hour_angle+ephem.hours('12:00')).norm/(2*pi) * 24
    return solarhour


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
