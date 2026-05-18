API_ENDPOINT_EVENTLIST = "https://neo-bolide.ndc.nasa.gov/service/event/public"
API_ENDPOINT_EVENT = "https://neo-bolide.ndc.nasa.gov/service/event/"
MPLSTYLE = "ggplot"

from pathlib import Path
ROOT_PATH = str(Path(__file__).parent)
GLM_FOV_PATH = ROOT_PATH + '/data/GLM_FOV_edges.nc'

from .bolide import *
from .bolidelist import *
from .bdf import BolideDataFrame
from .sdf import ShowerDataFrame
from .utils import youtube_photometry
from .constants import GOES_W_LON, GOES_E_LON, GLM_STEREO_MIDPOINT, FY4A_LON

__all__ = ["BolideDataFrame"]
