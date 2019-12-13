import cv2
import numpy as np
import lightkurve as lk
from pytube import YouTube


def youtube_photometry(video):
    """Returns a LightCurve object containing the total grayscale intensity
    of a youtube video over time."""
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
