import ephem
from math import degrees


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
    from math import pi
    o.long = lon/180 * pi
    sun = ephem.Sun()
    sun.compute(o)
    hour_angle = o.sidereal_time() - sun.ra
    solarhour = ephem.hours(hour_angle+ephem.hours('12:00')).norm/(2*pi) * 24
    return solarhour


def get_sun_alt(row):
    """Get the solar altitude given a row of the BolideDataFrame"""
    obs = ephem.Observer()
    obs.lon = str(row['longitude'])
    obs.lat = str(row['latitude'])
    obs.date = row['datetime']
    sun = ephem.Sun()
    sun.compute(obs)
    return degrees(sun.alt)


# TODO: THIS IS INCORRECT AND AN APPROXIMATION
# NEED TO CHECK APPROXIMATION
def get_observed_alts(apparent_alts):
    """Get the observed altitude given an apparent altitude"""

    # from astropy.coordinates.builtin_frames.itrs_observed_transforms import add_refraction
    for alt in apparent_alts:
        pass
    return apparent_alts  # apparent_alts - correction


def vel_to_radiant(dt, vx, vy, vz):
    """Input velocity in ITRS frame, output (uncorrected) radiant in ICRS frame"""
    from astropy.coordinates import ICRS, SkyCoord
    from astropy.time import Time

    time = Time(dt)
    # input negatives of coordinates because we want the direction they're coming from
    c = SkyCoord(x=-vx, y=-vy, z=-vz, representation_type='cartesian', frame='itrs', obstime=time)
    radec = c.transform_to(ICRS)
    return radec.ra.value, radec.dec.value


def geocentric_to_ecliptic(ra, dec):
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    lat = c.barycentrictrueecliptic.lat.value
    lon = c.barycentrictrueecliptic.lon.value
    return lat, lon


def calc_usg_orbit(dt, v, vx, vy, vz, lat, lon, alt, wmpl_path='python'):
    """Input velocity in ITRS frame, output orbital elements"""
    from subprocess import Popen,  PIPE
    from astropy.coordinates import ICRS, SkyCoord
    from astropy.time import Time

    time = Time(dt)
    # input negatives of coordinates because we want the direction they're coming from
    c = SkyCoord(x=-vx, y=-vy, z=-vz, representation_type='cartesian', frame='itrs', obstime=time)
    radec = c.transform_to(ICRS)
    ra = radec.ra.value
    dec = radec.dec.value
    import subprocess
    datestr = dt.strftime('%Y%m%d-%H%M%S.0')
    args = f'-r {ra} -d {dec} -v {v} -t {datestr} -a {lat} -o {lon} -e {alt} -s'.split()
    process = Popen([wmpl_path, "-m", "wmpl.Trajectory.Orbit"]+args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    stdout = stdout.decode()
    contains_data = stdout.__contains__('Orbit:')
    keys = ['ra', 'dec', 'LaSun', 'a', 'e', 'i', 'peri', 'node', 'Pi', 'b', 'q', 'f', 'M', 'Q', 'n', 'T']
    import numpy as np
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
    JD = sol_lon_to_jd(lon, year)
    from astropy.time import Time
    t = Time(JD, format='jd', scale='utc')
    return t.datetime


def sol_lon_to_jd(lon, year):
    """Get the Julian Day given a solar longitude.

    An algorithm for computing the Julian Day given a solar longitude,
    as described in:
    Low-Precision Formulae for Calculating Julian Day from Solar Longitude,
    E. Ofek, WGN 2000.
    https://ui.adsabs.harvard.edu/abs/2000JIMO...28..176O.
    """

    from math import floor
    from math import radians
    lon = radians(lon)
    Y = year
    M = floor(lon/360 * 12)+3
    D = 1
    from datetime import datetime
    dt = datetime(Y, M, D, 0, 0, 0)
    from astropy.time import Time
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

    from math import sin

    N = year - 2000
    JD0 = M0 + M1*N
    TDelta = A1*sin(lon+phi1) + A2*sin(2*lon+phi2) + B1 + B2*lon + P2*(JD_approx - 2451545)
    if abs(JD_approx-JD0-TDelta) > 50:
        TDelta += 365.2596
    JD = JD0 + TDelta
    return JD
