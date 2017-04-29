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
        self.assertTrue(1 == 1)
