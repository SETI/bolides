# This module contains high-level plotting tools for use with the BolieDataFrame class, 
# but also accessible for general use by other packages.

import pandas as pd
import numpy as np
from warnings import warn, filterwarnings

import matplotlib.pyplot as plt

from geopandas import GeoDataFrame
from shapely.geometry import Point

from . import MPLSTYLE, ROOT_PATH
from .utils import reconcile_input

def plot_scatter(
        latitude, 
        longitude, 
        crs=None,
        category=None, 
        boundary=None, 
        coastlines=True, 
        style=MPLSTYLE,
        boundary_style={}, 
        figsize=(10, 5),
        fig=None,
        ax=None,
        **kwargs):
    """Plot detections of bolides.

    Reprojects the geometry of bdf to the crs given, and scatters the points
    on a cartopy map. kwargs are passed through to matplotlib's scatter.

    Parameters
    ----------
    latitude : np.array
        The array of latitudes (degrees) to plot the density.
        Using EPSG:4326 - WGS 84, latitude/longitude coordinate system
        sized N
    longitude : np.array
        The array of longitudes (degrees) to plot the density.
        Using EPSG:4326 - WGS 84, latitude/longitude coordinate system
        sized N
    crs : `~cartopy.crs.CRS`
        The map projection to use. Refer to
        https://scitools.org.uk/cartopy/docs/latest/reference/projections.html.
    category : str list
        The category label to color each datum in the scatter plot
        sized N
    boundary : str or list of str
        The boundaries to plot.
        Refer to `~bolides.fov_utils.get_boundary`.
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
    c : np.array or list
        A ana array to use for the mappint to color each point
    color : str
        A single matplotlib color string
    fig : `~matplotlib.pyplot.figure`
        If passed then use this figure, otherwise, create one
        Use genearate_plot to return the fig and ax
        Note: Both fig and ax must be passed or neither
    ax : `~cartopy.mpl.geoaxes.GeoAxesSubplot`
        If passed then use this axis, otherwise, create one
        Use genearate_plot to return the fig and ax
        Note: Both fig and ax must be passed or neither

    Returns
    -------
    fig : `~matplotlib.pyplot.figure`
    ax : `~cartopy.mpl.geoaxes.GeoAxesSubplot`
    """
    
    # Both fig and ax must be passed or neither
    if fig is None != ax is None:
        # XOR gate
        'Both fig and ax must be passed or neither'
    elif fig is None and ax is None:
        fig_and_ax_passed = False
    else:
        fig_and_ax_passed = True

    # Project the latitudes and longitudes to the correct projection
    points = [Point(lon, lat) for lon, lat in zip(longitude, latitude)]
    gdf = GeoDataFrame(geometry=points, crs='epsg:4326')
    
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
    gdf_proj = gdf.to_crs(crs_proj4)
    # filter out rows with no geometry

    # default parameters put into kwargs if not specified by user
    assert not ('c' in kwargs and 'color' in kwargs), "Either 'c' or 'color' may be passed but not both"
    defaults = {'marker': '.', 'color': 'red', 'cmap': plt.get_cmap('viridis')}
    kwargs = reconcile_input(kwargs, defaults)
    if 'c' in kwargs:
        del kwargs['color']
    else:
        del kwargs['cmap']  # We are using a solid color, no color mapping

    good_locs = ~gdf_proj.geometry.is_empty
    x = np.empty(len(gdf_proj))
    y = np.empty(len(gdf_proj))
    points = gdf_proj[good_locs]['geometry']
    x[good_locs] = np.array([p.x for p in points])
    y[good_locs] = np.array([p.y for p in points])

    # using the given style,
    with plt.style.context(style):

        # generate Figure and GeoAxes with the given projection
        if not fig_and_ax_passed:
            fig, ax = generate_plot(crs=crs, figsize=figsize, style=style)
            ax.stock_img()  # plot background map

        # scatter points, passing arguments through

        if category is None:  # if there is no categorical variable specified
            cb = plt.scatter(x, y, **kwargs)
            # if color is determined by a quantitative variable, we add a colorbar
            if 'c' in kwargs:
                plt.colorbar(cb, label=kwargs['c'].name)

        else:  # if there is a categorical variable specified, color points using it
            assert len(category) == len(gdf), 'Variable <category> should be an array of strings the size of the input Lat and Lon'
            unique = category.unique()  # get the unique values of the variable
            import matplotlib.colors as colors
            hot = plt.get_cmap('tab10')
            cNorm = colors.Normalize(vmin=0, vmax=len(unique))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)

            del kwargs['color']  # color kwarg being overridden by categorical variable
            s = kwargs['s'] if 's' in kwargs else None

            # for each unique category, scatter the data with the right color
            for num, label in enumerate(unique):
                idx = category == label
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

def plot_density(
        latitude, 
        longitude, 
        crs=None,
        bandwidth=5, 
        boundary=None,
        n_levels=100, 
        lat_resolution=200, 
        lon_resolution=100,
        coastlines=True, 
        style=MPLSTYLE,
        boundary_style={},
        kde_params={}, 
        figsize=(10, 5), 
        title=None,
        fig=None,
        ax=None,
        **kwargs):
    """Plot event density on an Earth projection.

    Density is computed using scikit-learn's `~sklearn.neighbors.KernelDensity`
    using the haversine distance metric (as the data is in longitude and latitude)
    and gaussian kernel by default.
    It is then gridded, projected, and plotted.

    Parameters
    ----------
    latitude : np.array
        The array of latitudes (degrees) to plot the density.
        Using EPSG:4326 - WGS 84, latitude/longitude coordinate system
    longitude : np.array
        The array of longitudes (degrees) to plot the density.
        Using EPSG:4326 - WGS 84, latitude/longitude coordinate system
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
    title : str
        The title for the figure, if desired
    fig : `~matplotlib.pyplot.figure`
        If passed then use this figure, otherwise, create one
        Use genearate_plot to return the fig and ax
        Note: Both fig and ax must be passed or neither
    ax : `~cartopy.mpl.geoaxes.GeoAxesSubplot`
        If passed then use this axis, otherwise, create one
        Use genearate_plot to return the fig and ax
        Note: Both fig and ax must be passed or neither

    Returns
    -------
    fig : `~matplotlib.pyplot.figure`
    ax : `~cartopy.mpl.geoaxes.GeoAxesSubplot`

    """
    # Both fig and ax must be passed or neither
    if fig is None != ax is None:
        # XOR gate
        'Both fig and ax must be passed or neither'
    elif fig is None and ax is None:
        fig_and_ax_passed = False
    else:
        fig_and_ax_passed = True

    assert len(latitude) == len(longitude), "latitude and longitude should be the same length."
    num_bolides = len(latitude)

    from .crs import DefaultCRS
    if crs is None:
        crs = DefaultCRS()

    from cartopy import crs as ccrs
    # The cartopy library used by plot_density currently has many
    # warnings about the shapely library deprecating things...
    # This code suppresses those warnings
    filterwarnings("ignore", message="__len__ for multi-part")
    filterwarnings("ignore", message="Iteration over multi-part")

    # get numpy array in the format that KernelDensity likes
    data = np.vstack([np.radians(latitude), np.radians(longitude)]).T

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
        if not fig_and_ax_passed:
            fig, ax = generate_plot(crs=crs, figsize=figsize, style=style)
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

        if title is not None:
            plt.title(title)

    return fig, ax

def plot_interactive(df, mode='earth', projection="eckert4",
                     boundary=None, color=None, logscale=False,
                     culture='western', reference_plane='ecliptic',
                     **kwargs):

    # TODO: write header and documentation throughout this function

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

#*************************************************************************************************************
# Helper functions

def generate_plot(crs=None, figsize=(10,5), style=MPLSTYLE):
    """ Helper function to generate the plot figure. 

    This can be used to generate a single figure then call the plotting
    functions in this module repeatedly to superimpose data on the figure.

    Parameters
    ----------
    crs : `~cartopy.crs.CRS`
        The map projection to use. Refer to
        https://scitools.org.uk/cartopy/docs/latest/reference/projections.html.
    figsize : tuple
        The size (width, height) of the plotted figure.
    style : str
        The matplotlib style to use. Refer to
        https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html


    Returns
    -------
    fig : `~matplotlib.pyplot.figure`
    ax : `~cartopy.mpl.geoaxes.GeoAxesSubplot`
    """
    from .crs import DefaultCRS
    if crs is None:
        crs = DefaultCRS()

    with plt.style.context(style):
        fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=figsize)
        ax.stock_img()  # plot background map

    return fig, ax
