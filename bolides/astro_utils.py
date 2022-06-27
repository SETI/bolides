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
