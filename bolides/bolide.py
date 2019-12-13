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

    @property
    def attachments(self):
        return self.json['attachments']

    @property
    def geodata(self):
        return [self.json['attachments'][idx]['geoData']
                for idx in range(len(self.json['attachments']))]

    @property
    def longitudes(self):
        return [
                    [x['location']['coordinates'][0]
                    for x in self.geodata[idx]]
                for idx in range(len(self.geodata))]

    @property
    def latitudes(self):
        return [
                    [x['location']['coordinates'][1]
                    for x in self.geodata[idx]]
                for idx in range(len(self.geodata))]

    @property
    def times(self):
        return [[x['time'] for x in self.geodata[idx]]
                for idx in range(len(self.geodata))]

    @property
    def energies(self):
        return [[x['energy'] for x in self.geodata[idx]]
                for idx in range(len(self.geodata))]