import numpy as np
from geopandas import GeoDataFrame
import cartopy.crs as ccrs

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

    polygons = [get_boundary(b) for b in boundary]

    crs = ccrs.AzimuthalEquidistant(central_latitude=90)
    ax.add_geometries(polygons, crs=crs, **boundary_style)


def get_boundary(boundary):
    from netCDF4 import Dataset
    from shapely.geometry import Polygon

    if boundary == 'goes-w-ni':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat'][0]
        lons = fov.variables['G17_fov_lon'][0]
        polygon = Polygon(zip(lons, lats))

    elif boundary == 'goes-w-i':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat_inverted'][0]
        lons = fov.variables['G17_fov_lon_inverted'][0]
        polygon = Polygon(zip(lons, lats))

    elif boundary == 'goes-17-89.5':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G17_fov_lat'][0]
        lons = np.array(fov.variables['G17_fov_lon'][0]) + (-89.5-(-137.2))
        polygon = Polygon(zip(lons, lats))

    elif boundary == 'goes-w':
        from shapely.ops import unary_union

        # get data and create polygons
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        goes_w_ni = get_boundary('goes-w-ni')
        goes_w_i = get_boundary('goes-w-i')
        return unary_union([goes_w_ni, goes_w_i])

    elif boundary == 'goes-e':
        fov = Dataset(GLM_FOV_PATH, "r", format="NETCDF4")
        lats = fov.variables['G16_fov_lat'][0]
        lons = fov.variables['G16_fov_lon'][0]
        polygon = Polygon(zip(lons, lats))

    elif boundary == 'fy4a-n':
        lons = [55, 157.4, 127, 81.75]
        lats = [56.25, 56.25, 14.92, 14.92]
        polygon = fy4a_corners_to_boundary(lons, lats)

    elif boundary == 'fy4a-s':
        lons = [81.75, 127, 157.4, 55]
        lats = [-14.92, -14.92, -56.25, -56.25]
        polygon = fy4a_corners_to_boundary(lons, lats)

    return aeqd_from_lonlat(polygon)


def aeqd_from_lonlat(polygon):
    import pyproj
    # define Azimuthal Equidistant projection
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    gdf = GeoDataFrame(geometry=[polygon], crs='epsg:4236')
    gdf = gdf.to_crs(aeqd)
    return gdf.geometry[0]


# get corner points of fy4a FOV and return boundary
def fy4a_corners_to_boundary(lons, lats):
    import pyproj
    from shapely.geometry import Point, Polygon, LinearRing

    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    gdf = GeoDataFrame(geometry=points, crs='epsg:4236')
    geos = pyproj.Proj(proj='geos', ellps='WGS84', datum='WGS84', h=35785831.0, lon_0=105).srs
    gdf = gdf.to_crs(geos)
    points = gdf.geometry
    linestring = LinearRing(zip([p.x for p in points], [p.y for p in points]))
    length = linestring.length
    points = [linestring.interpolate(x) for x in np.linspace(0, length, 1000)]
    polygon = Polygon(zip([p.x for p in points], [p.y for p in points]))
    gdf = GeoDataFrame(geometry=[polygon], crs=geos)
    gdf = gdf.to_crs('epsg:4236')
    return gdf.geometry[0]
