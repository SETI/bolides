import pandas as pd
import numpy as np
import requests

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
    def __init__(self, source='established', file=None):
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
        numeric_cols = ['a', 'e', 'peri', 'node', 'inc']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        super().__init__(df)

    def plot_orbits(self, date='2000-01-01T12:00:00', use_3d=False, interactive=False):
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

        from poliastro.bodies import Sun, Earth
        from astropy import units as u
        from poliastro.twobody import Orbit

        df = self.dropna(subset=['a','e','inc','node','peri'])

        import warnings
        if use_3d and len(df)>50:
            warnings.warn('This will plot a lot of orbits and may crash your browser.\n\
                           If you are afraid of this happening, please do now use show()')

        from datetime import datetime
        dt = datetime.fromisoformat(date)
        from astropy.time import Time
        epoch = Time(dt, scale='tdb')

        from poliastro.plotting.misc import plot_solar_system
        plotter = plot_solar_system(use_3d=use_3d, interactive=interactive, epoch=epoch)
        plotter.set_attractor(Sun)

        from tqdm import tqdm
        for num, row in tqdm(df.iterrows()):
            a = row['a'] * u.AU
            ecc = row['e'] * u.one
            inc = row['inc'] * u.deg
            raan = row['node'] * u.deg
            argp = row['peri'] * u.deg
            nu = 5 * u.deg
            try:
                orb = Orbit.from_classical(Sun, a, ecc, inc, raan, argp, nu)
            except ValueError:
                continue
            plotter.plot(orb, label=row['shower name'])
        if use_3d:
            fig = plotter._figure
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=900)
            axis_dict = {'showgrid': False, 'zeroline': False, 'visible': False}
            fig.update_layout(scene={'xaxis':axis_dict, 'yaxis':axis_dict, 'zaxis':axis_dict})
        return plotter

    def __setattr__(self, attr, val):
        from warnings import filterwarnings
        filterwarnings("ignore", message="Pandas doesn't allow columns to be created via a new attribute name")
        return super().__setattr__(attr, val)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        force_showers_class(result)
        return result

    def _repr_html_(self):
        with pd.option_context('display.max_columns', None):
            return super()._repr_html_()

def force_showers_class(showers):
    if isinstance(showers, pd.DataFrame):
        showers.__class__ = ShowerDataFrame
