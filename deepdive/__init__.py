__version__ = '1.0.26'

__citation__ = """
Cooper RB, Flannery-Sutherland JT, Silvestro D. 
DeepDive: estimating global biodiversity patterns through time using deep learning. 
Nature Communications. 2024 May 17;15(1):4199"""

from . import bd_simulator
from .bd_simulator import *

from . import fossil_simulator
from .fossil_simulator import *

from . import plots
from .plots import *

from . import feature_extraction
from .feature_extraction import *

from . import rnn_builder
from .rnn_builder import *

from . import compare_time_series
from .compare_time_series import *

from . import utilities
from .utilities import *

from . import simulation_utilities
from .simulation_utilities import *

from . import deepdiver_utilities
from .deepdiver_utilities import *

from . import div_rates
from .div_rates import *