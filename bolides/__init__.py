API_ENDPOINT_EVENTLIST = "https://neo-bolide.ndc.nasa.gov/service/event/public"
API_ENDPOINT_EVENT = "https://neo-bolide.ndc.nasa.gov/service/event/"
MPLSTYLE = "ggplot"

import pkg_resources
ROOT_PATH = pkg_resources.resource_filename('bolides', '.')
GLM_FOV_PATH = ROOT_PATH + '/data/GLM_FOV_edges.nc'

from .bolide import *
from .bolidelist import *
from .bdf import *
from .utils import *
