import sklearn
from packaging import version

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")

import tensorflow as tf

assert version.parse(tf.__version__) >= version.parse("2.8.0")

from . import data, models, plotting, stats
from .data import *
from .models import *
from .plotting import *
from .stats import *

plotting.set_plot_style()

__version__ = "0.1.0"
