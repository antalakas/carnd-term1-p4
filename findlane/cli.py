#!/usr/bin/env python

from findlane import FindLane


#///////////////////////////////////////////////////
# Checks if the required modules have been installed.
def dependencies():
    try:
        return True
    except ImportError:
        return False


#///////////////////////////////////////////////////
#  FindLane command line interface
def findlane_cli():

    fl = FindLane()

    # successfully_calibrated = fl.find_chessboard_corners(6, 9, False)
    # fl.check_undistort(True)
    # print("successfully calibrated: %s images" % str(successfully_calibrated))
    #
    # fl.execute_pipeline()

    fl.warp()

if __name__ == "__main__":

    try:
        if dependencies():
            findlane_cli()
        else:
            raise(Exception, "Packages required: ...")
    except KeyboardInterrupt:
        print("\n")
