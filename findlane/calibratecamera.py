import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import ntpath


class CalibrateCamera(object):
    def __init__(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.

        self.mtx = []
        self.dist = []

        self.camera_calibration_path = '../camera_cal/'
        self.test_images_path = '../test_images/'
        self.output_images_path = '../output_images/'

    def find_chessboard_corners(self, row_corners, column_corners, visualize=False):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((row_corners * column_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[0:column_corners, 0:row_corners].T.reshape(-1, 2)

        # Make a list of calibration images
        images = glob.glob(self.camera_calibration_path + 'calibration*.jpg')
        # images = glob.glob('../camera_cal2/*.jpg')

        successfully_calibrated = 0

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (column_corners, row_corners), None)

            # If found, add object points, image points
            if ret == True:

                successfully_calibrated += 1

                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                if visualize is True:
                    # Draw and display the corners
                    cv2.drawChessboardCorners(img, (column_corners, row_corners), corners, ret)
                    # write_name = 'corners_found'+str(idx)+'.jpg'
                    # cv2.imwrite(write_name, img)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)

        cv2.destroyAllWindows()

        return successfully_calibrated

    def check_undistort(self, visualize=False):
        # Test undistortion on test images
        images = glob.glob(self.test_images_path + '*.jpg')

        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            img_size = (img.shape[1], img.shape[0])

            # Do camera calibration given object points and image points
            ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)

            dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            image_fname = ntpath.basename(fname)
            cv2.imwrite(self.output_images_path + 'undist_' + image_fname, dst)

            # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
            if visualize is True:
                # Visualize undistortion
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
                ax1.imshow(img)
                ax1.set_title('Original Image', fontsize=30)
                ax2.imshow(dst)
                ax2.set_title('Undistorted Image', fontsize=30)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open(self.camera_calibration_path + "wide_dist_pickle.p", "wb"))