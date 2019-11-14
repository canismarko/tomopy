#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015-2019, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2015-2019. UChicago Argonne, LLC. This software was produced  #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################


import os
import unittest
from unittest import mock

import numpy as np

import tomopy
from tomopy.misc import phantom
from tomopy.recon.wrappers import astra, astra_rec_cpu, astra_rec_cuda


__author__ = "Mark Wolfman"
__copyright__ = "Copyright (c) 2019, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class AstraWrapperTestCase(unittest.TestCase):
    """Does the imported astra module get passed around properly."""
    def test_astra_module(self):
        # Prepare some sample test data
        sim = phantom.shepp2d(128)
        recon = np.empty_like(sim)
        theta = np.linspace(0, np.pi, num=100)
        tomo = tomopy.sim.project.project(sim, theta=theta, sinogram_order=True)
        # Run the CPU reconstruction with mocked astra toolbox
        astra_mod = mock.MagicMock()
        options = dict(method='SIRT')
        astra(tomo, center=[64], recon=recon, theta=theta, astra_mod=astra_mod,
                      num_gridx=128, num_gridy=1, options=options)
        # Check that astra was called properly
        astra_mod.algorithm.run.assert_called()
        # Run the GPU reconstruction with mocked astra toolbox
        astra_mod = mock.MagicMock()
        options = dict(method='FBP', proj_type='cuda')
        astra(tomo, center=[64], recon=recon, theta=theta, astra_mod=astra_mod,
                      num_gridx=128, num_gridy=1, options=options)
        # Check that astra was called properly
        astra_mod.algorithm.run.assert_called()


class AstraCPUTestCase(unittest.TestCase):
    @unittest.expectedFailure
    def test_rot_center(self):
        # Prepare some sample test data
        sim = phantom.shepp2d(128)
        recon = np.empty_like(sim)
        theta = np.linspace(0, np.pi, num=100)
        tomo = tomopy.sim.project.project(sim, theta=theta, sinogram_order=True)
        # Call the cpu recon with a mocked astra toolbox and a default rotation center
        astra_mod = mock.MagicMock()
        options = dict(method='FBP')
        astra_rec_cpu(tomo, center=[64], recon=recon, theta=theta,
                      vol_geom=None, niter=1, proj_type='linear',
                      opts=options, astra_mod=astra_mod)


class AstraGPUTestCase(unittest.TestCase):
    @unittest.expectedFailure
    def test_rot_center(self):
        # Prepare some sample test data
        sim = phantom.shepp2d(128)
        recon = np.empty_like(sim)
        theta = np.linspace(0, np.pi, num=100)
        tomo = tomopy.sim.project.project(sim, theta=theta, sinogram_order=True)
        # Call the cpu recon with a mocked astra toolbox and a default rotation center
        astra_mod = mock.MagicMock()
        options = dict(method='FBP')
        astra_rec_cuda(tomo, center=[64], recon=recon, theta=theta,
                       vol_geom=None, niter=1, proj_type='linear',
                       opts=options, astra_mod=astra_mod, gpu_index=0)
