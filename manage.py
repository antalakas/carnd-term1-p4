#!/usr/bin/env python

# -*- coding: utf-8 -*-

#///////////////////////////////////////////////////////////
#-----------------------------------------------------------
# File: manage.py
# Author: Andreas Ntalakas
#-----------------------------------------------------------

#///////////////////////////////////////////////////
# Python packages
#---------------------------------------------------
import unittest
import argparse
#---------------------------------------------------
from subprocess import call

#///////////////////////////////////////////////////
# findlane - Import Test Suite
from tests.suite import test_suite

#///////////////////////////////////////////////////
# Execute
if __name__ == "__main__":

    try:
        #///////////////////////////////////////////////////
        parser = argparse.ArgumentParser(description="Installation Manager.")

        #///////////////////////////////////////////////////
        parser.add_argument('--run-tests', action='store_true', dest='run_tests', help='Run unit-tests for findlane package.')
        parser.add_argument('--generate-docs', action='store_true', dest='generate_docs', help='Generate html documentation for findlane package.')
        parser.add_argument('--install-pkg', action='store_true', dest='install_pkg', help='Install findlane package.')

        #///////////////////////////////////////////////////
        args = parser.parse_args()

        if args.run_tests:
            unittest.TextTestRunner(verbosity=2).run(test_suite())

        if args.generate_docs:
            call(["make", "html", "-C", "./docs"])

        if args.install_pkg:
            call(["python", "setup.py", "install"])

    except KeyboardInterrupt:
        print("\n")
