from math import degrees, pi, floor, radians, sin
from datetime import datetime
from pytz import timezone
from subprocess import Popen, PIPE
import numpy as np
import ephem
import astropy.units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.time import Time
from scipy.special import comb

def get_phase(datetime):
    """Get lunar phase (0.01=new moon just happened, 0.99=new moon about to happen)"""
    date = ephem.Date(datetime)
    nnm = ephem.next_new_moon(date)
    pnm = ephem.previous_new_moon(date)

    lunation = (date-pnm)/(nnm-pnm)
    return lunation


def get_solarhour(datetime, lon):
    """Get the hour in solar time given a datetime and longitude"""
    o = ephem.Observer()
    o.date = datetime
    o.long = lon/180 * pi
    sun = ephem.Sun()
    sun.compute(o)
    hour_angle = o.sidereal_time() - sun.ra
    solarhour = ephem.hours(hour_angle+ephem.hours('12:00')).norm/(2*pi) * 24
    return solarhour


def get_sun_alt(dt, lat, lon):
    """Get the solar altitude given a date and location"""
    obs = ephem.Observer()
    obs.lon = str(lon)
    obs.lat = str(lat)
    obs.date = dt
    sun = ephem.Sun()
    sun.compute(obs)
    observed = degrees(sun.alt)
    obs.pressure = 0
    sun.compute(obs)
    apparent = degrees(sun.alt)
    return np.array([observed, apparent])


def vel_to_radiant(dt, vx, vy, vz):
    """Input velocity in ITRS frame, output (uncorrected) radiant in ICRS frame"""

    time = Time(dt)
    # input negatives of coordinates because we want the direction they're coming from
    c = SkyCoord(x=-vx, y=-vy, z=-vz, representation_type='cartesian', frame='itrs', obstime=time)
    radec = c.transform_to(ICRS)
    return radec.ra.value, radec.dec.value


def geocentric_to_ecliptic(ra, dec):
    """Given ra and dec, compute ecliptic latitude and longitude"""

    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    lat = c.barycentrictrueecliptic.lat.value
    lon = c.barycentrictrueecliptic.lon.value
    return lat, lon


def calc_orbit(dt, v, vx, vy, vz, lat, lon, alt, wmpl_path='python'):
    """
    Input velocity vector and position in ITRS frame, output orbital elements.

    wmpl_path must be a callable Python instance which has the
    WesternMeteorPyLib installed.
    """

    keys = ['ra', 'dec', 'LaSun', 'a', 'e', 'i', 'peri', 'node', 'Pi', 'b', 'q', 'f', 'M', 'Q', 'n', 'T']

    time = Time(dt)
    # input negatives of coordinates because we want the direction they're coming from
    c = SkyCoord(x=-vx, y=-vy, z=-vz, representation_type='cartesian', frame='itrs', obstime=time)
    radec = c.transform_to(ICRS)
    ra = radec.ra.value
    dec = radec.dec.value
    datestr = dt.strftime('%Y%m%d-%H%M%S.0')
    args = f'-r {ra} -d {dec} -v {v} -t {datestr} -a {lat} -o {lon} -e {alt} -s'.split()
    process = Popen([wmpl_path, "-m", "wmpl.Trajectory.Orbit"]+args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode()
    contains_data = stdout.__contains__('Orbit:')
    if not contains_data:
        data = [np.nan]*len(keys)
    else:
        lines = stdout.split('\n')
        new_radiant_line = np.argmax([line.__contains__('Radiant (geocentric, J2000)') for line in lines])
        ra = float(lines[new_radiant_line+1].split()[2].strip('+'))
        dec = float(lines[new_radiant_line+2].split()[2].strip('+'))
        data = [ra, dec]

        orbit_line = np.argmax([line.__contains__('Orbit:') for line in lines])
        for i in range(14):
            if i == 0:
                idx = 3
            else:
                idx = 2
            data.append(float(lines[orbit_line+1+i].split()[idx]))
    data_dict = dict(zip(keys, data))
    return data_dict


def sol_lon_to_datetime(lon, year):
    """Given solar longitude and year, compute datetime"""

    JD = sol_lon_to_jd(lon, year)
    t = Time(JD, format='jd', scale='utc')
    dt = t.datetime

    # make the UTC datetime timezone-aware
    # if the timezone is not given, assume UTC (as elsewhere in the package)
    if dt.tzinfo is None:
        utc = timezone('UTC')
        dt = utc.localize(dt)

    return dt


def sol_lon_to_jd(lon, year):
    """Get the Julian Day given a solar longitude.

    An algorithm for computing the Julian Day given a solar longitude,
    as described in:
    Low-Precision Formulae for Calculating Julian Day from Solar Longitude,
    E. Ofek, WGN 2000.
    https://ui.adsabs.harvard.edu/abs/2000JIMO...28..176O.

    This algorithm is approximate (see paper for details)
    but is good enough for plotting and filtering
    """

    lon = radians(lon)
    Y = year
    M = floor(lon/360 * 12)+3
    D = 1
    dt = datetime(Y, M, D, 0, 0, 0)
    t = Time(dt, format='datetime', scale='utc')
    JD_approx = t.jd

    M0 = 2451182.24736
    M1 = 365.25963575
    A1 = 1.94330
    phi1 = -1.798135
    A2 = 0.013053
    phi2 = 2.634232
    B1 = 78.1927
    B2 = 58.13165
    P2 = -0.0000089408


    N = year - 2000
    JD0 = M0 + M1*N
    TDelta = A1*sin(lon+phi1) + A2*sin(2*lon+phi2) + B1 + B2*lon + P2*(JD_approx - 2451545)
    if abs(JD_approx-JD0-TDelta) > 50:
        TDelta += 365.2596
    JD = JD0 + TDelta
    return JD

def haversine(lat1, lon1, lat2, lon2):
        """
        Calculates the haversine distance between two points in km.
        Inputs are in degrees.

        Parameters
        ----------
        lat1, lon1 : float
            Latitude and longitude of the first point in degrees.
        lat2, lon2 : float
            Latitude and longitude of the second point in degrees.

        Returns
        -------
        float
            The haversine distance between the two points in kilometers.
        """
        R = 6371  # Earth radius in km

        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

def _distance_metric(x, y, fov_goes, total_time, n):
    """
    Measurement of how improbable it is for two bolides to be a distance apart in time and space

    Parameters
    ----------
    x, y: tuple
        Each tuple contains (time_seconds, latitude, longitude). They are rows from the BolideDataFrame.
        time_seconds is the time in seconds since the start of the dataset.
        latitude and longitude are in degrees.
    fov_goes: float
        The area of the GOES field of view in km^2.
    total_time: float
        The total time in seconds over which the bolides were observed.
    n: int
        The number of bolides in the dataset.

    Returns
    -------
    float
        The computed distance between the two points, considering both spatial and
        temporal dimensions. This estimates the probability that two bolide events would
        randomly be found as close together in space and time as the two being compared,
        under a uniform random distribution.

        np.pi*spatial_dist**2
            This is the area of a circle with radius equal to the spatial distance
            between the two points (in km²).

        / fov_goes
            Divides by the total field-of-view area (in km²) of the GOES sensor.
            This gives the probability that two random points would be within spatial_dist
            of each other, assuming uniform distribution over the field of view.

        (2*dt/total_time)
            dt is the absolute time difference between the two events (in seconds).
            total_time is the total time span of the dataset (in seconds).
            This gives the probability that two random events would be within dt of each other in time.

        comb(n, 2, exact=False)
            This is the number of unique pairs you can form from n events.
            It scales the probability to account for all possible pairs in the dataset.

        This gives the expected number of pairs (out of all possible pairs) that would be at least as
        close in space and time as the two points being compared, under a random (uniform) distribution.

        Lower values mean the pair is unusually close in space and time (less likely by chance).
        Higher values mean the pair is not unusually close (more likely by chance).

        Used as the DistanceMetric64 "distance" between the two points for the function get_closest().
    """
    
    t1, x1, y1 = x
    t2, x2, y2 = y
    spatial_dist = haversine(x1, y1, x2, y2) 
    dt = abs(t1 - t2)

    return (np.pi*spatial_dist**2/fov_goes)*(2*dt/total_time) * comb(n, 2, exact=False)
