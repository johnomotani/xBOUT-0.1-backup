from .grid import open_grid

from . import geometries
from .geometries import register_geometry

from .boutdataset import BoutDatasetAccessor, open_boutdataset
from .boutdataarray import BoutDataArrayAccessor

from .plotting.animate import animate_imshow
from .plotting.utils import plot_separatrix

from ._version import __version__
