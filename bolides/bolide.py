import requests

from . import API_ENDPOINT_EVENT


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
