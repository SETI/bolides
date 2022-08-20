import pandas as pd
import requests
import numpy as np
from . import ROOT_PATH


class ShowerDataFrame(pd.DataFrame):
    """
    Subclass of Pandas `~pandas.DataFrame` with additional meteor-shower-specific methods.

    Parameters
    ----------
    source : str
        Specifies the source for the initialized. Can be:

        - ``'established', 'all', 'working'``: initialize from different sets
          of data offered at the IAU Meteor Data Center, https://www.ta3.sk/IAUC22DB/MDC2007/
        - ``'csv'``: initialize from a .csv file

    file : str
        Specifies files to be used if the source is ``'csv'``
    """
    def __init__(self, *args, **kwargs):
        if len(args)==0 and len(kwargs)==0:
            kwargs['source'] = 'established'
        if 'source' not in kwargs:
            return super().__init__(*args, **kwargs)
        if 'file' not in kwargs:
            kwargs['file'] = None
        source = kwargs['source']
        file = kwargs['file']
        if source == 'csv':
            df = pd.read_csv(file, index_col=0, keep_default_na=False, na_values='')
        else:
            url = 'https://www.ta3.sk/IAUC22DB/MDC2007/Etc/stream'+source+'data.txt'
            r = requests.get(url)
            start_line = 0
            for num, line in enumerate(r.text.splitlines()):
                if line[0] not in [':', '+']:
                    start_line = num
                    break
            column_line = r.text.splitlines()[start_line-3]
            import re
            columns = re.split(r" {2,}", column_line)[1:-1]
            data_lines = r.text.splitlines()[start_line:]
            data = '\n'.join(data_lines)
            import io
            csv_io = io.StringIO(data)
            df = pd.read_csv(csv_io, sep="|", header=None)
            df.columns = columns
            for col in df.columns:
                if df[col].dtype == 'O':
                    df[col] = df[col].str.strip()

        column_translation = {'inc': 'i', 'Ra': 'ra', 'De': 'dec'}
        df = df.rename(columns=column_translation)
        df['source'] = 'iau'

        numeric_cols = ['a', 'e', 'peri', 'node', 'i']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        super().__init__(df)

    def plot_orbits(self, date='2000-01-01T12:00:00',
                    use_3d=False, interactive=False, num_points=150):
        """Plot an interactive map of shower orbits.

        Parameters
        ----------
        date : str
            The date at which to plot the data, in ISO format.
        use_3d : bool
            Whether or not to make a 3D plot using Plotly
        interactive : bool
            Whether or not to make an interactive plot using Plotly

        Returns
        -------
        plotter : `~poliastro.plotting.core.OrbitPlotter3D` or `~poliastro.plotting.core.OrbitPlotter2D`
        """

        if use_3d:
            interactive = True

        from poliastro.bodies import Sun
        from astropy import units as u
        from poliastro.twobody import Orbit

        from datetime import datetime
        dt = datetime.fromisoformat(date)
        from astropy.time import Time
        epoch = Time(dt, scale='tdb').tdb

        from poliastro.plotting.misc import _plot_bodies
        from poliastro.plotting.interactive import OrbitPlotter2D, OrbitPlotter3D
        from poliastro.plotting.static import StaticOrbitPlotter

        from poliastro.frames import Planes

        if use_3d:
            plotter = OrbitPlotter3D(plane=Planes.EARTH_ECLIPTIC)
        elif interactive:
            plotter = OrbitPlotter2D(plane=Planes.EARTH_ECLIPTIC)
        else:
            plotter = StaticOrbitPlotter(plane=Planes.EARTH_ECLIPTIC)

        from warnings import simplefilter
        import erfa
        simplefilter("ignore", erfa.core.ErfaWarning)

        plotter._num_points = num_points
        plotter.set_attractor(Sun)

        _plot_bodies(plotter, epoch=epoch)

        has_data = all([col in self.columns for col in ['a', 'e', 'i', 'node', 'peri']])

        if has_data:
            sdf = self.dropna(subset=['a', 'e', 'i', 'node', 'peri'])

            import warnings
            if use_3d and len(sdf) > 50:
                warnings.warn('This will plot a lot of orbits and may crash your browser.\n'
                              'If you are afraid of this happening, please do now use show()')


            for num, row in sdf.iterrows():
                a = row['a'] * u.AU
                ecc = row['e'] * u.one
                inc = row['i'] * u.deg
                raan = row['node'] * u.deg
                argp = row['peri'] * u.deg
                nu = 5 * u.deg
                try:
                    orb = Orbit.from_classical(Sun, a, ecc, inc, raan, argp, nu, plane=Planes.EARTH_ECLIPTIC)
                except ValueError:
                    continue
                plotter.plot(orb, label=row['shower name'])

        if use_3d:
            fig = plotter._figure
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=900)
            axis_dict = {'showgrid': False, 'zeroline': False, 'visible': False}
            fig.update_layout(scene={'xaxis': axis_dict, 'yaxis': axis_dict, 'zaxis': axis_dict})

            # replace spheres with points and shorten labels
            import plotly.graph_objects as go
            good_data = []
            planets = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
            i = 0
            sun_added = False
            while(i < len(fig.data)):
                data = fig.data[i]
                orig_name = data.name
                sun = data.name == 'Sun'
                if not sun:
                    data.name = " (".join(data.name.split(' (')[1:])[:-1]

                planet = any([data.name.__contains__(p) for p in planets])
                if planet and type(data) is go.Surface:
                    name = orig_name
                    x = np.mean(data.x)
                    y = np.mean(data.y)
                    z = np.mean(data.z)
                    color = data.colorscale[0][1]
                    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], name=name))
                    fig.update_traces(marker=dict(color=color, size=4),
                                      selector=dict(name=name, type='scatter3d'),
                                      showlegend=False)
                if not planet:
                    fig.update_traces(line_dash='dot', selector=dict(name=data.name, type='scatter3d'))
                if type(data) is not go.Surface and (not sun or not sun_added):
                    good_data.append(data)
                    if sun:
                        sun_added = True
                i += 1
            fig.data = good_data

            min_x = min([min(d.x) for d in fig.data])
            min_y = min([min(d.y) for d in fig.data])
            min_z = min([min(d.z) for d in fig.data])
            max_x = max([max(d.x) for d in fig.data])
            max_y = max([max(d.y) for d in fig.data])
            max_z = max([max(d.z) for d in fig.data])

            a_x = max_x-min_x
            a_y = max_y-min_y
            a_z = max_z-min_z
            c_x = min_x + a_x/2
            c_y = min_y + a_y/2
            c_z = min_z + a_z/2

            longest = max([a_x, a_y, a_z])
            fig.update_scenes(aspectratio=dict(x=a_x/longest, y=a_y/longest, z=a_z/longest),
                              aspectmode="manual")

            x = (0-c_x)/a_x * a_x/longest
            y = (0-c_y)/a_y * a_y/longest
            z = (0-c_z)/a_z * a_z/longest
            fig.update_layout(scene_camera=dict(up=dict(x=0, y=0, z=1),
                              center=dict(x=x, y=y, z=z)))

        return plotter

    def get_dates(self, showers, years):
        if type(years) is int:
            years = [years]
        if type(showers) in [str, int]:
            showers = [showers]
        showers = [str(s) for s in showers]
        if showers[0].isdigit():
            col = 'IAUNo'
            showers = [int(s) for s in showers]
        elif len(showers[0]) == 3:
            col = 'Code'
        else:
            col = 'shower name'
        sdf = self[self[col].isin(showers)]
        import warnings
        if len(sdf) == 0:
            warnings.warn('No showers with '+col+'in'+str(showers)+'.')

        sdf = sdf[sdf.activity == 'annual']
        sdf = sdf.dropna(subset=['LaSun'])
        num = len(sdf)
        sdf = pd.concat([sdf]*len(years), ignore_index=True)
        years = np.repeat(years, num)

        lons = sdf.LaSun
        from .astro_utils import sol_lon_to_datetime
        dts = []
        for lon, year in zip(lons, years):
            dts.append(sol_lon_to_datetime(lon, year))
        return pd.DataFrame({'Code': sdf.Code,
                             'shower name': sdf['shower name'],
                             'IAUNo': sdf.IAUNo,
                             'datetime': dts})

    def __setattr__(self, attr, val):
        from warnings import filterwarnings
        filterwarnings("ignore", message="Pandas doesn't allow columns to be created via a new attribute name")
        return super().__setattr__(attr, val)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        force_showers_class(result)
        return result

    @property
    def _constructor(self):
        return ShowerDataFrame

    def _repr_html_(self):
        with pd.option_context('display.max_columns', None):
            return super()._repr_html_()


def force_showers_class(showers):
    if isinstance(showers, pd.DataFrame):
        showers.__class__ = ShowerDataFrame
