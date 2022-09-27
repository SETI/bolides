from math import degrees, pi, floor, radians, sin
from datetime import datetime
from subprocess import Popen, PIPE
import numpy as np
import ephem
import astropy.units as u
from astropy.coordinates import ICRS, SkyCoord
from astropy.time import Time


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
    return t.datetime


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
