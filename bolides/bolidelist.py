import requests

from . import Bolide
from . import API_ENDPOINT_EVENTLIST


class BolideList():

    def __init__(self):
        self.json = self._load_json()

    def _load_json(self):
        r = requests.get(API_ENDPOINT_EVENTLIST)
        return r.json()

    @property
    def ids(self):
        return [event["_id"] for event in self.json['data']]

    def __getitem__(self, idx):
        return Bolide(self.ids[idx])
