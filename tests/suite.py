# -*- coding: utf-8 -*-


#///////////////////////////////////////////////////
#---------------------------------------------------
# File: suite.py
# Author: Andreas Ntalakas
#---------------------------------------------------

#///////////////////////////////////////////////////
# Python
#---------------------------------------------------
import os
#---------------------------------------------------
import unittest

#///////////////////////////////////////////////////
# findlane Tests
#---------------------------------------------------
from .test_findlane import TestFindLane
#---------------------------------------------------

#///////////////////////////////////////////////////
def test_suite():
    """
        `test_suite()` method creates a test suite
        for the unit-tests of findlane package.
    """

    allTests = unittest.TestSuite()

    allTests.addTest(TestFindLane('find_chessboard_corners'))

    return allTests
