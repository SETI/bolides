from datetime import datetime

import numpy as np
import pandas as pd
from pytz import timezone
from geopandas import GeoDataFrame
import pyproj
import cartopy.crs as ccrs
from shapely.geometry import Point, Polygon, LinearRing

from . import GLM_FOV_PATH, ROOT_PATH
from .utils import reconcile_input

# all times in this package are UTC; used to localize naive datetimes
utc = timezone('UTC')

# the GOES satellites carrying a GLM instrument that this package knows about
GOES_SATELLITES = [16, 17, 18, 19]


def add_boundary(ax, boundary=None, boundary_style={}):
    """Add a boundary to Cartopy GeoAxesSubplot"""

    assert boundary is not None

    # define defaults and add them to the user-defined style,
    # without overriding any user-specified parameters
    boundary_defaults = {"facecolor": "none"}
    boundary_style = reconcile_input(boundary_style, boundary_defaults)

    # get Polygons representing the boundaries
    polygons = get_boundary(boundary)

    # if polygons is not a list, make it a list.
    if type(polygons) is not list:
        polygons = [polygons]

    # define the crs of the polygons
    crs = ccrs.AzimuthalEquidistant(central_latitude=90)
    # add the polygons to the GeoAxesSubplot, passing the boundary style in
    ax.add_geometries(polygons, crs=crs, **boundary_style)


def change_crs(polygons, current, new):
    single_item = False
    if type(polygons) is not list:
        polygons = [polygons]
        single_item = True

    gdf = GeoDataFrame(geometry=polygons, crs=current)
    gdf = gdf.to_crs(new)

    if single_item:
        return gdf.geometry[0]
    else:
        return list(gdf.geometry)


def get_boundary(boundary, collection=True, intersection=False, crs=None):
    """Get specified boundary polygons.

    GOES GLM FOV obtained from a netCDF4 file kindly provided by Katrina Virts
    (NASA/MSFC; katrina.virts@uah.edu)
    based on information from Doug Mach and Clem Tillier.
    Fengyun-4A FOV obtained by eye from several papers.
    Global Meteor Network FOV obtained from KML files at
    https://globalmeteornetwork.org/data/.

    Possible values in boundary:

    - ``'goes'``: Combined FOV of the GLM aboard GOES-16 and GOES-17
    - ``'goes-w'``: GOES-West position GLM FOV, currently corresponding to GOES-17.
      Note that this combines the inverted and non-inverted FOVs.
    - ``'goes-e'``: GOES-East position GLM FOV, currently corresopnding to GOES-16.
    - ``'goes-w-normal'``: GOES-West position GLM FOV, when GOES-17 is not inverted (summer).
    - ``'goes-w-inverted'``: GOES-West position GLM FOV, when GOES-17 is inverted (winter).
    - ``'goes-17-checkout-orbit'``: GOES-17 GLM FOV when it was in its checkout orbit.
    - ``'fy4a'``: Combined FOV of the Fengyun-4A LMI, in both North and South configurations.
    - ``'fy4a-n'``: Fengyun-4A LMI FOV when in the North configuration (summer).
    - ``'fy4a-s'``: Fengyun-4A LMI FOV when in the South configuration (winter).
    - ``'gmn-25km'``: Combined FOV of all Global Meteor Network stations at 25km detection altitude.
    - ``'gmn-70km'``: Combined FOV of all Global Meteor Network stations at 70km detection altitude.
    - ``'gmn-100km'``: Combined FOV of all Global Meteor Network stations at 100km detection altitude.

    Parameters
    ----------
    boundary: str or iterable with multiple strings
        Specifies the boundaries desired. Currently supports:
    collection: bool
        If boundary contains multiple boundaries,
        ``True`` will return a list of boundaries,
        while ``False`` will return their combination (according to the intersection argument)
    intersection: bool
        If boundary contains multiple strings and collection is ``False``, when True will return
        the intersection of the fields-of-view, when False will return their union.

    Returns
    -------
    `~shapely.geometry.Polygon` or list of `~shapely.geometry.Polygon` in the Azimuthal Equidistant
    Coordinate Reference System with a central latitude of 90, central longitude of 0.
    """
    from netCDF4 import Dataset
    from shapely.ops import unary_union

    if type(boundary) is str:
        boundary = boundary.lower()

    valid_boundaries = ['goes-w-normal', 'goes-w-inverted', 'goes-w', 'goes-e', 'goes',
                        'fy4a-s', 'fy4a-n', 'fy4a', 'goes-17-checkout-orbit',
                        'gmn-100km', 'gmn-70km', 'gmn-25km']

    if type(boundary) is str and boundary not in valid_boundaries:
        raise ValueError("unknown boundary \""+boundary+"\". Valid boundaries are: \n"+str(valid_boundaries))

    if type(boundary) is not str and not hasattr(boundary, '__iter__'):
        raise ValueError("boundary must be a string or list")

    if crs is not None:
        polygons = get_boundary(boundary, collection, intersection, crs=None)
        aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
        return change_crs(polygons, aeqd, crs)

    if boundary == 'goes-w-normal':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat'][0]
        lons = fov.variables['G17_fov_lon'][0]
        polygon = Polygon(zip(lons, lats))
        return aeqd_from_lonlat(polygon)

    elif boundary == 'goes-w-inverted':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat_inverted'][0]
        lons = fov.variables['G17_fov_lon_inverted'][0]
        polygon = Polygon(zip(lons, lats))
        return aeqd_from_lonlat(polygon)

    elif boundary == 'goes-17-checkout-orbit':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat'][0]
        lons = np.array(fov.variables['G17_fov_lon'][0]) + (-89.5-(-137.2))
        polygon = Polygon(zip(lons, lats))
        return aeqd_from_lonlat(polygon)

    elif boundary == 'goes-w':
        return get_boundary(['goes-w-normal', 'goes-w-inverted'], collection=False)

    elif boundary == 'goes':
        return get_boundary(['goes-w', 'goes-e'], collection=False)

    elif boundary == 'goes-e':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G16_fov_lat'][0]
        lons = fov.variables['G16_fov_lon'][0]
        polygon = Polygon(zip(lons, lats))
        return aeqd_from_lonlat(polygon)

    elif boundary == 'fy4a-n':
        lons = [55, 157.4, 127, 81.75]
        lats = [56.25, 56.25, 14.92, 14.92]
        polygon = fy4a_corners_to_boundary(lons, lats)
        return aeqd_from_lonlat(polygon)

    elif boundary == 'fy4a-s':
        lons = [81.75, 127, 157.4, 55]
        lats = [-14.92, -14.92, -56.25, -56.25]
        polygon = fy4a_corners_to_boundary(lons, lats)
        return aeqd_from_lonlat(polygon)

    elif boundary == 'fy4a':
        return get_boundary(['fy4a-s', 'fy4a-n'], collection=False)

    elif boundary == 'gmn-100km':
        return load_gmn_shp('100km')

    elif boundary == 'gmn-70km':
        return load_gmn_shp('70km')

    elif boundary == 'gmn-25km':
        return load_gmn_shp('25km')

    polygons = []
    for b in boundary:
        new_boundary = get_boundary(b)
        if type(new_boundary) is list:
            polygons += new_boundary
        else:
            polygons.append(new_boundary)

    if collection:
        return polygons
    # either take intersection of FOVs or the union to get a final polygon
    if intersection:
        final_polygon = polygons[0]
        for polygon in polygons:
            final_polygon = final_polygon.intersection(polygon)
    else:
        final_polygon = unary_union(polygons)

    return final_polygon


def aeqd_from_lonlat(polygon):
    # define Azimuthal Equidistant projection
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    gdf = GeoDataFrame(geometry=[polygon], crs='epsg:4326')
    gdf = gdf.to_crs(aeqd)
    return gdf.geometry[0]


def load_gmn_shp(distance):
    # load Global Meteor Network shapefiles
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    from geopandas import read_file
    path = ROOT_PATH+'/data/gmn/'+distance+'.shp'
    gdf = read_file(path)
    gdf = gdf.to_crs(aeqd)

    # combine them into one polygon, buffering to resolve shape problems
    from shapely.ops import unary_union
    polygons = [p.buffer(0) for p in gdf.geometry]
    return unary_union(polygons)


# get corner points of fy4a FOV and return boundary
def fy4a_corners_to_boundary(lons, lats):

    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    gdf = GeoDataFrame(geometry=points, crs='epsg:4326')
    geos = pyproj.Proj(proj='geos', ellps='WGS84', datum='WGS84', h=35785831.0, lon_0=105).srs
    gdf = gdf.to_crs(geos)
    points = gdf.geometry
    linestring = LinearRing(zip([p.x for p in points], [p.y for p in points]))
    length = linestring.length
    points = [linestring.interpolate(x) for x in np.linspace(0, length, 1000)]
    polygon = Polygon(zip([p.x for p in points], [p.y for p in points]))
    gdf = GeoDataFrame(geometry=[polygon], crs=geos)
    gdf = gdf.to_crs('epsg:4326')
    return gdf.geometry[0]


def get_mask(xy, boundary=None, intersection=False):
    # define Azimuthal Equidistant projection
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs

    if boundary is None:
        return np.ones(len(xy))

    polygon = get_boundary(boundary, intersection=intersection, collection=False)

    points = [Point(p[1], p[0]) for p in xy]
    gdf = GeoDataFrame(geometry=points, crs='epsg:4326')
    gdf = gdf.to_crs(aeqd)
    points = gdf.geometry
    points_in_poly = [p.within(polygon) for p in points]
    return np.array(points_in_poly)


def _normalize_satellite(satellite):
    """Normalize a GOES satellite identifier to its integer number.

    Accepts an int (e.g. ``16``) or a string in any of the common forms
    (``'16'``, ``'g16'``, ``'glm-16'``, ``'goes-16'``, case-insensitive) and
    returns the integer 16/17/18/19. Raises ``ValueError`` otherwise.
    """
    if isinstance(satellite, (int, np.integer)):
        num = int(satellite)
    elif isinstance(satellite, str):
        s = satellite.lower().strip()
        # strip the common textual prefixes, longest first so 'g' does not
        # swallow the leading letter of 'glm'/'goes'
        for prefix in ('glm-', 'glm', 'goes-', 'goes', 'g'):
            if s.startswith(prefix):
                s = s[len(prefix):]
                break
        try:
            num = int(s)
        except ValueError:
            raise ValueError('unknown satellite "' + str(satellite) + '". Valid '
                             'satellites are: ' + str(GOES_SATELLITES))
    else:
        raise ValueError('satellite must be an int or string, got ' + str(type(satellite)))

    if num not in GOES_SATELLITES:
        raise ValueError('unknown satellite "' + str(satellite) + '". Valid '
                         'satellites are: ' + str(GOES_SATELLITES))
    return num


def get_obs_intervals(satellite):
    """Get the operational intervals and FOV orientations of a GOES satellite's GLM.

    Reads the packaged ``data/glm##_obs.csv`` file that records when the given
    satellite was active and, for GOES-17, which yaw-flip orientation it was in.

    Parameters
    ----------
    satellite: int or str
        A GOES satellite identifier accepted by `_normalize_satellite`
        (e.g. ``16``, ``'g16'``, ``'glm-16'``, ``'goes-16'``).

    Returns
    -------
    `~pandas.DataFrame`
        A frame with at least the columns ``boundary``, ``start`` and ``end``.
        ``start``/``end`` are ISO-format date strings (``end`` may be empty,
        meaning "still active"); ``boundary`` is a name understood by
        `get_boundary`.
    """
    num = _normalize_satellite(satellite)
    path = ROOT_PATH + '/data/glm' + str(num) + '_obs.csv'
    # the files carry a leading '#' comment line (hence header=1) and are
    # whitespace-aligned for readability: skipinitialspace handles the spaces
    # after each comma, and the strip below removes any trailing spaces so the
    # date strings parse cleanly (e.g. "2023-01-03   " -> "2023-01-03")
    df = pd.read_csv(path, header=1, skipinitialspace=True)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def _to_utc(time):
    """Coerce a time (ISO string, datetime, or pandas/numpy datetime) to UTC-aware."""
    if isinstance(time, str):
        dt = datetime.fromisoformat(time)
    elif isinstance(time, datetime):
        dt = time
    else:
        # pandas Timestamp, numpy datetime64, etc.
        dt = pd.Timestamp(time).to_pydatetime()
    # if the timezone is not given, assume UTC (as elsewhere in the package)
    if dt.tzinfo is None:
        dt = utc.localize(dt)
    return dt


def _parse_window(value):
    """Parse a start/end cell from an obs csv into a UTC-aware datetime or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return None
    return _to_utc(str(value).strip())


def _boundary_position(boundary):
    """Return (position, inverted) describing a GLM FOV boundary name."""
    position = 'east' if boundary == 'goes-e' else 'west'
    inverted = boundary == 'goes-w-inverted'
    return position, inverted


def get_satellites(lat, lon, time, numbers_only=False):
    """Determine which GOES satellites' GLM could observe a spatiotemporal point.

    Given a point (or several points) in space and time, returns the GOES
    satellites (16, 17, 18, 19) whose Geostationary Lightning Mapper was both
    operational and had the point within its field of view at that time. This is
    the per-point inverse of `~bolides.BolideDataFrame.filter_observation`, and
    does not require a `~bolides.BolideDataFrame`.

    The operational windows and FOV orientations come from the packaged
    ``data/glm##_obs.csv`` files (see `get_obs_intervals`); the FOV polygons come
    from `get_boundary`. GOES-18 reuses the GOES-West FOV and GOES-19 the
    GOES-East FOV.

    Parameters
    ----------
    lat, lon: float or array-like of float
        Latitude and longitude of the point(s), in degrees (EPSG:4326).
    time: str, `~datetime.datetime`, or array-like
        The time(s) of the point(s). Strings are parsed with
        `~datetime.datetime.fromisoformat`; naive times are assumed to be UTC.
        Must be the same length as ``lat`` and ``lon`` (a single point may use
        scalars for all three).
    numbers_only: bool
        If ``False`` (default), return structured per-satellite information. If
        ``True``, return only the satellite numbers.

    Returns
    -------
    `~pandas.DataFrame` or list
        - If ``numbers_only`` is ``False`` (default): a `~pandas.DataFrame` with
          one row per observing (point, satellite), with columns ``latitude``,
          ``longitude``, ``time``, ``satellite``, ``position``
          (``'east'``/``'west'``), ``boundary``, ``inverted``,
          ``'Instrument Operational'`` and ``'Instrument Disabled'`` (the latter
          is ``'Still In Operation'`` for satellites that have not been retired).
          For array input a leading ``point`` column gives the 0-based index into
          the inputs.
        - If ``numbers_only`` is ``True``: for a single point, a sorted list of
          observing satellite integers; for array input, a list of such lists,
          one per point.
    """
    # determine whether the caller passed a single point (scalars) or arrays,
    # to decide the shape of the return value
    scalar = np.ndim(lat) == 0
    lats = np.atleast_1d(np.asarray(lat, dtype=float))
    lons = np.atleast_1d(np.asarray(lon, dtype=float))

    # normalize time into a list, then to UTC-aware datetimes
    if isinstance(time, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        times = [_to_utc(t) for t in time]
    else:
        times = [_to_utc(time)]

    if not (len(lats) == len(lons) == len(times)):
        raise ValueError('lat, lon and time must have the same length (got '
                         + str(len(lats)) + ', ' + str(len(lons)) + ', '
                         + str(len(times)) + ')')

    # project the points to the Azimuthal Equidistant CRS the FOV polygons use
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    gdf = GeoDataFrame(geometry=[Point(lo, la) for lo, la in zip(lons, lats)], crs='epsg:4326')
    points = list(gdf.to_crs(aeqd).geometry)

    # preload each satellite's observation windows and the FOV polygons, so each
    # csv is read and each boundary projected only once
    obs = {}
    poly_cache = {}
    for num in GOES_SATELLITES:
        df = get_obs_intervals(num)
        rows = []
        for _, row in df.iterrows():
            boundary = row['boundary']
            rows.append((boundary, _parse_window(row['start']), _parse_window(row['end'])))
            if boundary not in poly_cache:
                poly_cache[boundary] = get_boundary(boundary)
        obs[num] = rows

    # for each point, find every satellite that was operational and viewing it
    records = []
    per_point_nums = []
    for i, (point, t) in enumerate(zip(points, times)):
        nums_here = []
        for num in GOES_SATELLITES:
            for boundary, start, end in obs[num]:
                in_time = (start is None or t >= start) and (end is None or t <= end)
                if in_time and point.within(poly_cache[boundary]):
                    nums_here.append(num)
                    position, inverted = _boundary_position(boundary)
                    # show a human-readable label rather than NaT when the
                    # satellite has no retirement date (still operating)
                    disabled = 'Still In Operation' if end is None else end
                    records.append({'point': i,
                                    'latitude': lats[i], 'longitude': lons[i], 'time': t,
                                    'satellite': num, 'position': position,
                                    'boundary': boundary, 'inverted': inverted,
                                    'Instrument Operational': start,
                                    'Instrument Disabled': disabled})
                    # the satellite observes this point; its windows are disjoint
                    # so there is no need to check the rest of them
                    break
        per_point_nums.append(sorted(nums_here))

    if numbers_only:
        return per_point_nums[0] if scalar else per_point_nums

    columns = ['point', 'latitude', 'longitude', 'time', 'satellite', 'position',
               'boundary', 'inverted', 'Instrument Operational', 'Instrument Disabled']
    result = pd.DataFrame(records, columns=columns)
    # for a single point the 'point' index column is just noise
    if scalar:
        result = result.drop(columns=['point'])
    return result
