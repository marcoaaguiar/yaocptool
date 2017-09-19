import sys
# sys.path.append(r"C:\casadi-py27-np1.9.1-v3.1.1")
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.2.3")

sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")

from casadi import SX,DM, inf, repmat, vertcat, log, substitute, \
                    sum1, dot, collocation_points, vec, Function, linspace, gradient
import casadi
#__all__ = 'config', 'problems', 'methods'

import config
import problems
import modelling_classes
import methods
from NMPCScheme import NMPCScheme