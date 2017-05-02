#!/usr/bin/env python

from findlane import FindLane
from calibratecamera import CalibrateCamera


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
    # cc = CalibrateCamera()
    #
    # successfully_calibrated = cc.find_chessboard_corners(6, 9, False)
    # print("successfully calibrated: %s images" % str(successfully_calibrated))
    #
    # cc.check_undistort(False)

    fl = FindLane()
    fl.execute_image_pipeline(True)
    # fl.project_video()

if __name__ == "__main__":

    try:
        if dependencies():
            findlane_cli()
        else:
            raise(Exception, "Packages required: ...")
    except KeyboardInterrupt:
        print("\n")
