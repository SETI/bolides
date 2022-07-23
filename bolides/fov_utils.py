import numpy as np
from geopandas import GeoDataFrame
import pyproj
import cartopy.crs as ccrs
from shapely.geometry import Point, Polygon, LinearRing

from . import GLM_FOV_PATH, ROOT_PATH
from .utils import reconcile_input


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
    - ``'goes-w-ni'``: GOES-West position GLM FOV, when GOES-17 is not inverted (summer).
    - ``'goes-w-i'``: GOES-West position GLM FOV, when GOES-17 is inverted (winter).
    - ``'goes-17-89.5'``: GOES-17 GLM FOV when it was in its checkout orbit.
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

    valid_boundaries = ['goes-w-ni', 'goes-w-i', 'goes-w', 'goes-e', 'goes',
                        'fy4a-s', 'fy4a-n', 'fy4a', 'goes-17-89.5',
                        'gmn-100km', 'gmn-70km', 'gmn-25km']

    if type(boundary) is str and boundary not in valid_boundaries:
        raise ValueError("unknown boundary \""+boundary+"\". Valid boundaries are: \n"+str(valid_boundaries))

    if type(boundary) is not str and not hasattr(boundary, '__iter__'):
        raise ValueError("boundary must be a string or list")

    if crs is not None:
        polygons = get_boundary(boundary, collection, intersection, crs=None)
        aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
        return change_crs(polygons, aeqd, crs)

    if boundary == 'goes-w-ni':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat'][0]
        lons = fov.variables['G17_fov_lon'][0]
        polygon = Polygon(zip(lons, lats))
        return aeqd_from_lonlat(polygon)

    elif boundary == 'goes-w-i':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat_inverted'][0]
        lons = fov.variables['G17_fov_lon_inverted'][0]
        polygon = Polygon(zip(lons, lats))
        return aeqd_from_lonlat(polygon)

    elif boundary == 'goes-17-89.5':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat'][0]
        lons = np.array(fov.variables['G17_fov_lon'][0]) + (-89.5-(-137.2))
        polygon = Polygon(zip(lons, lats))
        return aeqd_from_lonlat(polygon)

    elif boundary == 'goes-w':
        return get_boundary(['goes-w-ni', 'goes-w-i'], collection=False)

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
