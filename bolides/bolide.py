import requests

from . import API_ENDPOINT_EVENT

import simplekml
import numpy as np


class Bolide():
    """Represent a bright fireball reported at https://neo-bolide.ndc.nasa.gov

    Parameters
    ----------
    eventid : str
        Unique identifier of the event as used by https://neo-bolide.ndc.nasa.gov.
    """
    def __init__(self, eventid):
        self.eventid = eventid
        self.json = self._load_json(eventid)['data'][0]
        self.nSatellites = len(self.json['attachments'])

    def _load_json(self, eventid):
        """Returns a dictionary containing the data for the bolide."""
        url = f"{API_ENDPOINT_EVENT}/{eventid}"
        r = requests.get(url)
        return r.json()

    @property
    def detectedBy(self):
        return self.json['detectedBy']

    @property
    def howFound(self):
        return self.json['howFound']

    @property
    def confidenceRating(self):
        return self.json['confidenceRating']

    @property
    def _id(self):
        return self.json['_id']

    @property
    def attachments(self):
        return self.json['attachments']

    @property
    def netCDFFilename(self):
        # There appears to be a hex number in front of each filename
        # Strip this off
        filenames = []
        for idx in range(self.nSatellites):
            realFilenameIdx = self.json['attachments'][idx]['netCdfFilename'].find('_') + 1
            filenames.append(self.json['attachments'][idx]['netCdfFilename'][realFilenameIdx:])
        return filenames

    @property
    def satellite(self):
        return [self.json['attachments'][idx]['platformId']
                for idx in range(self.nSatellites)]

    @property
    def platformId(self):
        return self.satellite

    @property
    def geodata(self):
        return [self.json['attachments'][idx]['geoData']
                for idx in range(self.nSatellites)]

    @property
    def longitudes(self):
        return [
                    [x['location']['coordinates'][0]
                    for x in self.geodata[idx]]
                for idx in range(self.nSatellites)]

    @property
    def latitudes(self):
        return [
                    [x['location']['coordinates'][1]
                    for x in self.geodata[idx]]
                for idx in range(self.nSatellites)]

    @property
    def times(self):
        return [[x['time'] for x in self.geodata[idx]]
                for idx in range(self.nSatellites)]

    @property
    def energies(self):
        return [[x['energy'] for x in self.geodata[idx]]
                for idx in range(self.nSatellites)]
    
    def to_lightcurve(self, idx=0):
        """Returns the energies as a LightCurve object."""
        import lightkurve as lk
        return lk.LightCurve(time=self.times[idx],
                             flux=self.energies[idx],
                             targetid=self.eventid[idx])

    def save_kml(self, file_name = ''):

        data_count = len(self.latitudes[0])
        alts = np.linspace(80e3,30e3,data_count)
        lats = self.latitudes[0]
        lons = self.longitudes[0]
        energies = self.energies[0]
        log_energy_mean = np.log10(np.mean(energies))
        log_energies = np.log10(energies)

        self.sizing_scales = (log_energies / log_energy_mean)**5.0

        if not file_name:
            file_name = './kml_file.kml'
        
        kml = simplekml.Kml()
        
        for n, point in enumerate(zip(lons, lats, alts)):
            pnt = kml.newpoint(coords=[point])
            pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
            pnt.style.iconstyle.color = simplekml.Color.red
            pnt.style.iconstyle.scale = self.sizing_scales[n]
            pnt.altitudemode = simplekml.AltitudeMode.relativetoground
        
        kml.save(file_name)
