# -*- coding: utf-8 -*-


#///////////////////////////////////////////////////
#---------------------------------------------------
# File: test_findlane.py
# Author: Andreas Ntalakas
#---------------------------------------------------

#///////////////////////////////////////////////////
# Python
#---------------------------------------------------
import time
import string
import random

#///////////////////////////////////////////////////
# test_findlane
#---------------------------------------------------
import numpy as np
from findlane.findlane import *

#---------------------------------------------------
import unittest

#///////////////////////////////////////////////////
class TestFindLane(unittest.TestCase):
    """
    TestFindLane
    """

    #///////////////////////////////////////////////////
    def setUp(self):
        self.f = FindLane()

    def find_chessboard_corners(self):
        successfully_calibrated = self.f.find_chessboard_corners(6, 9)
        print("successfully calibrated: %s images" % str(successfully_calibrated))
        self.assertTrue(successfully_calibrated == 17)
