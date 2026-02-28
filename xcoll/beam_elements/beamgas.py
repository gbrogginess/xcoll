# copyright ################################# #
# This file is part of the Xfields Package.   #
# Copyright (c) CERN, 2021.                   #
# ########################################### #

import xobjects as xo
import xtrack as xt
import numpy as np

class BeamGasScattering(xt.BeamElement):

    _xofields = {
        '_p0c': xo.Float64,
        # others to be passed to the C-kernel
    }

    def _configure(self):
        pass