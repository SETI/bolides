from cartopy.crs import Geostationary, EqualEarth
from .constants import GOES_E_LON, GOES_W_LON, GLM_STEREO_MIDPOINT, FY4A_LON

class DefaultCRS(EqualEarth):
    """The default CoordinateReferenceSystem.

    An Equal-Earth projection from the point between GOES-W and GOES-E
    """
    def __init__(self):
        super().__init__(central_longitude=GLM_STEREO_MIDPOINT)

class GOES_E(Geostationary):
    """Cartopy CoordinateReferenceSystem representing the GOES-East perspective"""
    def __init__(self):
        super().__init__(central_longitude=GOES_E_LON)


class GOES_W(Geostationary):
    """Cartopy CoordinateReferenceSystem representing the GOES-West perspective"""
    def __init__(self):
        super().__init__(central_longitude=GOES_W_LON)


class FY4A(Geostationary):
    """Cartopy CoordinateReferenceSystem representing the Fengyun-4A perspective"""
    def __init__(self):
        super().__init__(central_longitude=FY4A_LON)
