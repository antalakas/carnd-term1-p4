#!/usr/bin/env python

# -*- coding: utf-8 -*-

#///////////////////////////////////////////////////////////
#-----------------------------------------------------------
# File: setup.py
# Author: Andreas Ntalakas
#-----------------------------------------------------------

#///////////////////////////////////////////////////////////
# Import a setup module
from setuptools import setup, find_packages

#///////////////////////////////////////////////////////////
# Import package
import findlane

#///////////////////////////////////////////////////////////
# Basic setup
setup(

    # Package Details
    name='findlane',
    version=findlane.__version__,
    author=findlane.__author__,
    author_email=findlane.__email__,
    url=findlane.__url__,
    packages=find_packages(exclude=['tests','docs'],),
    include_package_data=True,

    # Disable for allowing copying files
    zip_safe=False,

    # Package Description
    description='Advanced Lane Finding, Project 4, Term 1 CarND Nanodegree Udacity.',
    long_description=findlane.__description__,
    license='',

    # Dependent Packages (if any)
    requires=['Sphinx',],

    install_requires=[
        "Sphinx>=1.3.1",
    ],
)
