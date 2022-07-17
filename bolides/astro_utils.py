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
    M = floor(lon/360 * 12)+1
    print(M)
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
