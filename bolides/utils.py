from shapely.geometry import Point


class Wrapper():
    """Wrapper class for loading from pickled pipeline data"""
    def __init__(self):
        pass


def make_points(lons, lats):
    """Make Shapely Point objects from lists of longitudes and latitudes"""
    coords = zip(lons, lats)
    points = [Point(coord[0], coord[1]) for coord in coords]
    return points


def reconcile_input(user_input, defaults):
    for key, value in defaults.items():
        if key not in user_input:
            user_input[key] = value

    return user_input


def str_to_list(string):
    string = string.strip('[]')
    if string == '':
        return []
    return string.split(', ')


def youtube_photometry(video):
    """Returns a LightCurve object containing the total grayscale intensity
    of a youtube video over time."""
    # These imports are not used anywhere else, so let's have them be optional/local:
    import cv2
    import numpy as np
    import lightkurve as lk
    from pytube import YouTube

    url = f'https://youtu.be/{video}'
    path = YouTube(url).streams.first().download()
    cap = cv2.VideoCapture(path)
    flux = []
    status = True
    while status:
        status, frame = cap.read()
        if status:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flux.append(np.nansum(gray))
    return lk.LightCurve(flux=flux)
