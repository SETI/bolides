import ephem
from math import degrees


def get_phase(datetime):
    date = ephem.Date(datetime)
    nnm = ephem.next_new_moon(date)
    pnm = ephem.previous_new_moon(date)

    lunation = (date-pnm)/(nnm-pnm)
    return lunation


def get_solarhour(datetime, lon):
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
    obs = ephem.Observer()
    obs.lon = str(row['longitude'])
    obs.lat = str(row['latitude'])
    obs.date = row['datetime']
    sun = ephem.Sun()
    sun.compute(obs)
    return degrees(sun.alt)
