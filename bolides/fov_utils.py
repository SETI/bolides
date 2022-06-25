import numpy as np
from geopandas import GeoDataFrame
import pyproj
import cartopy.crs as ccrs
from shapely.geometry import Point, Polygon, LinearRing

from . import GLM_FOV_PATH


def add_boundary(ax, boundary=None, boundary_style={}):

    assert boundary is not None

    # for filename in boundary:
    #     with open('data/'+filename+'.pkl', 'rb') as f:
    #         b = pickle.load(f)
    #     ax.add_geometries([b[0]], crs=b[1], facecolor='none',
    #                       edgecolor='k', alpha=1, linewidth=3)

    boundary_defaults = {"facecolor": "none"}
    for key, value in boundary_defaults.items():
        if key not in boundary_style:
            boundary_style[key] = value

    polygons = get_boundary(boundary)

    crs = ccrs.AzimuthalEquidistant(central_latitude=90)
    if not hasattr(polygons, '__iter__'):
        polygons = [polygons]
    ax.add_geometries(polygons, crs=crs, **boundary_style)


def get_boundary(boundary, intersection=False, collection=True):
    from netCDF4 import Dataset
    from shapely.ops import unary_union

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
        return get_boundary(['goes-w-ni','goes-w-i'], collection=False)

    elif boundary == 'goes':
        return get_boundary(['goes-w','goes-e'], collection=False)

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
        return get_boundary(['fy4a-s','fy4a-n'], collection=False)

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
