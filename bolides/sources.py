import pandas as pd
import numpy as np
from geopandas import GeoDataFrame
import requests
from io import StringIO
from tqdm import tqdm

from . import API_ENDPOINT_EVENTLIST
from .utils import make_points
from datetime import datetime


def glm_website():

    # load data from website
    json = download(API_ENDPOINT_EVENTLIST, json=True)

    # create DataFrame using JSON data
    df = pd.DataFrame(json['data'])
    df["datetime"] = df["datetime"].astype("datetime64")

    # add bolide energy data
    energy_g16 = []
    energy_g17 = []
    for ats in df.attachments:
        e_g16 = 0
        e_g17 = 0
        for at in ats:
            platform = at['platformId']
            energy = at['energy']
            if platform == 'G16':
                e_g16 += energy
            if platform == 'G17':
                e_g17 += energy
        if e_g16 == 0:
            e_g16 = np.nan
        if e_g17 == 0:
            e_g17 = np.nan
        energy_g16.append(e_g16)
        energy_g17.append(e_g17)
    df['energy_g16'] = energy_g16
    df['energy_g17'] = energy_g17

    # add bolide brightness data
    brightness_cat_g16 = []
    brightness_g16 = []
    brightness_cat_g17 = []
    brightness_g17 = []
    val_cols = {'GLM-16': brightness_g16, 'GLM-17': brightness_g17}
    cat_cols = {'GLM-16': brightness_cat_g16, 'GLM-17': brightness_cat_g17}
    for brightness in df.brightness:
        for sat in ['GLM-16', 'GLM-17']:
            if sat in brightness:
                cat_cols[sat].append(brightness[sat]['category'])
                val_cols[sat].append(brightness[sat]['value'])
            else:
                cat_cols[sat].append("")
                val_cols[sat].append(np.nan)
    df['brightness_cat_g16'] = brightness_cat_g16
    df['brightness_g16'] = brightness_g16
    df['brightness_cat_g17'] = brightness_cat_g17
    df['brightness_g17'] = brightness_g17
    del df['brightness']

    gdf = add_geometry(df)

    return gdf


def usg():

    # load data from website
    json = download('https://ssd-api.jpl.nasa.gov/fireball.api?vel-comp=true', json=True)
    data = json['data']
    cols = json['fields']

    # create DataFrame
    df = pd.DataFrame(data, columns=cols)
    df['latitude'] = df['lat'].astype(float) * ((df['lat-dir'] == 'N') * 2 - 1)
    df['longitude'] = df['lon'].astype(float) * ((df['lon-dir'] == 'E') * 2 - 1)
    del df['lat'], df['lon'], df['lat-dir'], df['lon-dir']
    df['datetime'] = [datetime.fromisoformat(date) for date in df['date']]
    del df['date']
    numeric_cols = ['energy', 'impact-e', 'alt', 'vel', 'vx', 'vy', 'vz']
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    # compute ra and dec from velocity components
    df['ra'] = np.nan
    df['dec'] = np.nan
    from .astro_utils import vel_to_radiant
    # ignore SettingWithCopyWarning
    with pd.option_context('mode.chained_assignment', None):
        for num, row in df.dropna(subset=['vx', 'vy', 'vz']).iterrows():
            ra, dec = vel_to_radiant(row['datetime'], row['vx'], row['vy'], row['vz'])
            df['ra'][num] = ra
            df['dec'][num] = dec

    first_cols = ['datetime', 'longitude', 'latitude', 'source', 'energy',
                  'impact-e', 'alt', 'vel', 'source']
    first_cols = [col for col in first_cols if col in df.columns]
    other_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + other_cols]

    gdf = add_geometry(df)

    return gdf


def pipeline(files, min_confidence=0):

    from .pipeline_utils import dict_from_zodb
    dict_of_lists = dict_from_zodb(files=files, min_confidence=min_confidence)

    df = pd.DataFrame(dict_of_lists)

    column_translation = {'avgLon': 'longitude', 'avgLat': 'latitude', 'bolideTime': 'datetime',
                          'timeDuration': 'duration', 'goesSatellite': 'detectedBy'}
    df = df.rename(columns=column_translation)

    gdf = add_geometry(df)

    return gdf


def gmn(date, loc_mode='begin'):
    input_date = date
    month = False
    if type(date) is str:
        if len(date) == 7:
            month = True
        elif len(date) == 10:
            pass
        else:
            raise ValueError(f'invalid GMN date {date}')
        datestr = date.replace("-", "")
    else:
        datestr = date.strftime('%Y%m%d')

    base_path = 'https://globalmeteornetwork.org/data/traj_summary_data/'
    if month:
        data_path = f'{base_path}monthly/traj_summary_monthly_{datestr}.txt'
    else:
        # load index page
        from bs4 import BeautifulSoup
        archive_url = "https://globalmeteornetwork.org/data/traj_summary_data/daily/"
        r = requests.get(archive_url)
        soup = BeautifulSoup(r.content, 'html5lib')
        links = soup.findAll('a')

        csv_links = [archive_url + link['href'] for link in links if link['href'].startswith(f'traj_summary_{datestr}')]
        data_path = csv_links[0]
        print(data_path)

    data = download(data_path)
    if data.__contains__('404 Not Found'):
        unit_string = 'monthly' if month else 'daily'
        raise ValueError(f'{input_date} not found in GMN {unit_string} data.')
    buf = StringIO(data)
    buf.readline()
    header = buf.readline().split(';')
    header = [''.join(h.split()) for h in header]
    header[1] = 'jd'
    header[2] = 'datetime'
    header[3] = 'iau_no'
    header[4] = 'iau_code'
    for idx, col in enumerate(header):
        if col == '+/-':
            header[idx] = header[idx-1]+'_sd'
    buf.readline()
    buf.readline()
    df = pd.read_csv(buf, sep=';')
    df.columns = header

    df['datetime'] = pd.to_datetime(df['datetime'])
    if loc_mode == 'begin':
        df['latitude'] = df.LatBeg
        df['longitude'] = df.LonBeg
    else:
        df['latitude'] = df.LatEnd
        df['longitude'] = df.LonEnd

    column_translation = {'RAgeo': 'ra', 'DECgeo': 'dec', 'RAgeo_sd': 'ra_sd',
                          'DECgeo_sd': 'dec_sd'}
    df = df.rename(columns=column_translation)

    # non_numeric = ['source', '#Uniquetrajectory', 'iau_code', 'Begin', 'Endin', 'Participating']
    # dates = ['datetime', 'date-retrieved']
    # numeric_cols = [col for col in df.columns if col not in non_numeric+dates]
    # for col in numeric_cols:
    #    df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.convert_dtypes()

    gdf = add_geometry(df)

    return gdf


def csv(file):
    df = pd.read_csv(file, index_col=0,
                     parse_dates=['datetime'],
                     keep_default_na=False,
                     na_values='')

    gdf = add_geometry(df)
    return gdf


def remote(url):
    data = download(url)
    buf = StringIO(data)

    return csv(buf)


def download(url, chunk_size=1024, json=False):
    buf = StringIO()
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with tqdm(desc='downloading', total=total, unit='iB', unit_scale=True, unit_divisor=1024) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = buf.write(data.decode())
            bar.update(size)
    buf.seek(0)
    if json:
        import json
        return json.loads(buf.read())
    return buf.read()


def add_geometry(df, lat_col='latitude', lon_col='longitude'):
    lats = df[lat_col]
    lons = df[lon_col]
    points = make_points(lons, lats)

    gdf = GeoDataFrame(df, geometry=points, crs="EPSG:4326")
    return gdf
