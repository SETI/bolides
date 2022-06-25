from cartopy.crs import Geostationary
from . import GOES_E_LON, GOES_W_LON, FY4A_LON


class GOES_E(Geostationary):
    def __init__(self):
        super().__init__(central_longitude=GOES_E_LON)


class GOES_W(Geostationary):
    def __init__(self):
        super().__init__(central_longitude=GOES_W_LON)


class FY4A(Geostationary):
    def __init__(self):
        super().__init__(central_longitude=FY4A_LON)
