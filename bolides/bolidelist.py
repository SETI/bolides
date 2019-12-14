import os
import requests

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from . import Bolide
from . import API_ENDPOINT_EVENTLIST


class BolideList():

    def __init__(self):
        self.json = self._load_json()

    def _load_json(self):
        """Returns a dictionary containing all events."""
        r = requests.get(API_ENDPOINT_EVENTLIST)
        return r.json()

    @property
    def ids(self):
        return [event["_id"] for event in self.json['data']]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return Bolide(self.ids[idx])

    def to_pandas(self):
        """Returns a pandas DataFrame summarizing all bolides."""
        df = pd.DataFrame(self.json['data'])
        df["datetime"] = df["datetime"].astype("datetime64")
        return df

    def plot_dates(self, year=2019):
        """Plots the number of bolides over time."""
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        df = self.to_pandas()
        mask = (df.datetime > f'{year}-01-01') & (df.datetime < f'{year+1}-01-01')
        counts = df[mask].groupby(pd.Grouper(key='datetime', freq='1D')).count()
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.set_xlabel(year)
        ax.set_ylabel("# Events")
        ax.bar(counts.index, counts._id);
        return ax

    def plot_map(self):
        """Plots the spatial distribution of bolides using basemap."""
        from mpl_toolkits.basemap import Basemap
        fig = plt.figure(figsize=(9*1.618, 9))
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.89])
        ax.set_facecolor('white')
        m = Basemap(projection='kav7',
                    lon_0=-90, lat_0=0,
                    resolution="l", fix_aspect=False)
        m.drawcountries(color='#7f8c8d', linewidth=0.8)
        m.drawstates(color='#bdc3c7', linewidth=0.5)
        m.drawcoastlines(color='#7f8c8d', linewidth=0.8)
        m.fillcontinents('#ecf0f1', zorder=0)
        df = self.to_pandas()
        x, y = m(df.longitude.values, df.latitude.values)
        m.scatter(x, y, marker="o", color="red", edgecolor='black',
                lw=0.4, s=15, zorder=999)
        plt.show()




class AMSBolideList():

    def __init__(self, year=2019):
        self.json = self._load_json(year)

    def _load_json(self, year):
        key = os.environ.get('AMS_API_KEY')
        url = f"https://www.amsmeteors.org/members/api/open_api/get_events?api_key={key}&year={year}&format=json"
        r = requests.get(url)
        return r.json()

    @property
    def events(self):
        return [self.json['result'][key] for key in self.json['result']]

    def to_pandas(self):
        """Returns a pandas DataFrame summarizing all bolides."""
        import pandas as pd
        df = pd.DataFrame(self.json['result']).transpose()
        df["avg_date_utc"] = df["avg_date_utc"].astype("datetime64")
        return df
