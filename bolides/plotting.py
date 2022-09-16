import pandas as pd
import numpy as np
from .utils import reconcile_input
from . import ROOT_PATH


def plot_interactive(df, mode='earth', projection="eckert4",
                     boundary=None, color=None, logscale=False,
                     culture='western', reference_plane='ecliptic',
                     **kwargs):
    import plotly.express as px
    import plotly.graph_objects as go
    df = df.copy()
    hover_columns = ['datetime']

    if mode == 'earth':
        loc_cols = ['latitude', 'longitude']
    elif mode == 'radiant':
        loc_cols = ['dec', 'ra']

    if len(df) == 0 or any([col not in df.columns for col in loc_cols]):
        values = [[]] * len(loc_cols)
        df = pd.DataFrame(dict(list(zip(loc_cols, values))))

    if mode == 'radiant' and reference_plane == 'ecliptic':
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        lats = []
        lons = []
        for ra, dec in zip(df['ra'], df['dec']):
            c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
            lats.append(c.barycentrictrueecliptic.lat.value)
            lons.append(c.barycentrictrueecliptic.lon.value)
        df['lat_ecliptic'] = lats
        df['lon_ecliptic'] = lons
        df['lon_ecliptic_neg'] = -df['lon_ecliptic']
        loc_cols = ['lat_ecliptic', 'lon_ecliptic_neg']

    elif mode == 'radiant':
        df['ra_neg'] = -df['ra']
        loc_cols = ['dec', 'ra_neg']

    too_long = ['otherInformation', 'reason', 'description', 'otherDetectingSources', 'status',
                'lastModifiedBy', 'enteredBy', 'submittedBy', 'publishedBy', 'platform', 'rejectedBy',
                'rejectedDate', 'date_retrieved', '__v', 'Remarks', 'References']
    if color not in df.columns:
        color = None

    import numbers

    hover_columns = [col for col in hover_columns if col in df.columns]

    if len(df) > 0:
        for col in df.columns:
            if type(df[col].iloc[0]) in [list, dict]:
                continue
            if all([isinstance(x, numbers.Number) for x in df[col]]) and (col not in too_long):
                hover_columns.append(col)
            elif len(df[col].unique()) < 20 and (col not in too_long):
                hover_columns.append(col)
        for col in hover_columns:
            if col not in loc_cols + [color]:
                if all([isinstance(i, numbers.Number) for i in df[col]]):
                    df[col] = ['%g' % num for num in df[col]]
                else:
                    df[col] = df[col].fillna('')

    hover_columns = [col for col in hover_columns if not col.endswith('_sd')]

    if len(hover_columns) > 30:
        hover_columns = ['datetime', 'longitude', 'latitude', 'ra', 'dec', 'lon_ecliptic', 'lat_ecliptic']
    hover_columns = [col for col in hover_columns if col in df.columns]

    defaults = {'hover_data': hover_columns}
    kwargs = reconcile_input(kwargs, defaults)

    projection = projection.lower()
    proj_name = projection
    if projection in ['goes-e', 'goes-w', 'fy4a']:
        proj_name = 'satellite'
    if logscale and color is not None and all([isinstance(x, numbers.Number) for x in df[color]]):
        fig = px.scatter_geo(df, lat=loc_cols[0], lon=loc_cols[1],
                             color=np.log(df[color]),
                             projection=proj_name, **kwargs)
        fig.update_layout(coloraxis_colorbar={'title': 'log('+color+')'})
    else:
        fig = px.scatter_geo(df, lat=loc_cols[0], lon=loc_cols[1],
                             color=color,
                             projection=proj_name, **kwargs)
    from .constants import GLM_STEREO_MIDPOINT, GOES_E_LON, GOES_W_LON, FY4A_LON
    if len(df) > 0 and df['source'].iloc[0] == 'website' and mode == 'earth':
        fig.update_geos(projection_rotation={'lon': GLM_STEREO_MIDPOINT})
    if projection in ['satellite', 'goes-e', 'goes-w', 'fy4a']:
        distance = 42164/6378  # geostationary orbit distance
        rotation = 0
        if projection == 'goes-e':
            rotation = GOES_E_LON
        elif projection == 'goes-w':
            rotation = GOES_W_LON
        elif projection == 'fy4a':
            rotation = FY4A_LON
        fig.update_geos(projection={'type': 'satellite', 'distance': distance},
                        projection_rotation={'lon': rotation})

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=900)
    fig.update_geos(landcolor="White", lataxis_showgrid=True, lonaxis_showgrid=True)
    if mode == 'radiant':
        fig.update_geos(landcolor='Black', lakecolor='Black',
                        coastlinecolor='Black', oceancolor='Black', showocean=True,)
    fig.update_traces(marker=dict(size=8))
    if mode == 'earth':
        from bolides.fov_utils import get_boundary
        import pyproj
        import shapely
        from geopandas import GeoDataFrame
        if boundary is None:
            boundary = []
        if type(boundary) is str:
            boundary = [boundary]
        polygons = get_boundary(boundary)
        if type(polygons) is not list:
            polygons = [polygons]
        aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
        gdf = GeoDataFrame(geometry=polygons, crs=aeqd)
        gdf = gdf.to_crs('epsg:4326')
        polygons = gdf.geometry
        for num, polygon in enumerate(polygons):
            if type(polygon) is shapely.geometry.MultiPolygon:
                polygon_group = list(polygon)
            else:
                polygon_group = [polygon]

            from plotly.colors import qualitative
            for i, p in enumerate(polygon_group):
                lons, lats = p.exterior.coords.xy
                lons = np.array(lons)
                lats = np.array(lats)
                if boundary[num] in ['goes', 'goes-w', 'goes-w-i', 'goes-w-ni']:
                    lons = lons - (lons > 50) * 360
                fig.add_trace(go.Scattergeo(mode="lines", lon=lons, lat=lats,
                                            name=boundary[num], opacity=0.7,
                                            line=dict(color=qualitative.Pastel[num]),
                                            legendgroup=str(num),
                                            showlegend=(i == 0)))
    if mode == 'radiant':
        constellations = pd.read_csv(ROOT_PATH+'/data/constellations/'+culture+'.csv')

        # convert strings to lists
        from .utils import str_to_list
        for col in ['ra', 'dec']:
            constellations[col] = [str_to_list(x) for x in constellations[col]]

        for num, row in constellations.iterrows():
            name = row['name']
            data_x = np.array(row['ra']).astype(float)
            data_y = np.array(row['dec']).astype(float)
            if reference_plane == 'ecliptic':
                from astropy import units as u
                from astropy.coordinates import SkyCoord
                lats = []
                lons = []
                for ra, dec in zip(row['ra'], row['dec']):
                    try:
                        c = SkyCoord(ra=float(ra)*u.degree, dec=float(dec)*u.degree, frame='icrs')
                        ecliptic_coords = c.barycentrictrueecliptic
                        lats.append(ecliptic_coords.lat.value)
                        lons.append(ecliptic_coords.lon.value)
                    except ValueError as e:
                        print(e)
                data_x = np.array(lons)
                data_y = np.array(lats)
            data_x = -data_x
            for n in range(int(len(row['ra'])/2)):
                fig.add_trace(go.Scattergeo(mode="lines", name=name,
                                            hoverinfo='name', hoverlabel={'namelength': -1},
                                            lon=data_x[n*2:(n+1)*2],
                                            lat=data_y[n*2:(n+1)*2],
                                            line=dict(color='White'), showlegend=False))

    return fig
