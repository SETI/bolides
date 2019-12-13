import requests

from . import API_ENDPOINT_EVENT

import simplekml
import numpy as np


class Bolide():

    def __init__(self, eventid):
        self.eventid = eventid
        self.json = self._load_json(eventid)['data'][0]

    def _load_json(self, eventid):
        url = f"{API_ENDPOINT_EVENT}/{eventid}"
        r = requests.get(url)
        return r.json()
    
    @property
    def latitude(self):
        return self.json['latitude']

    @property
    def longitude(self):
        return self.json['longitude']

    @property
    def datetime(self):
        return self.json['datetime']

    def get_geodata(self, idx=0):
        return self.json['attachments'][idx]['geoData']
    
    def save_kml(self, file_name = ''):

        data_count = len(self.latitude)
        alts = np.linspace(80e3,30e3,data_count)
        lats = self.longitude[0]
        lons = self.latitude[0]
        
        if not file_name:
            file_name = './kml_file.kml'
        
        kml = simplekml.Kml()
        
        for n, point in enumerate(zip(lons, lats, alts)):
            pnt = kml.newpoint(coords=[point])
            pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
            pnt.style.iconstyle.color = simplekml.Color.blue
            pnt.style.iconstyle.scale = 1.5
            pnt.altitudemode = simplekml.AltitudeMode.relativetoground
        
        kml.save(file_name)
