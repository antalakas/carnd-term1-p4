import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


class FindLane(object):
    def __init__(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.

        self.mtx = []
        self.dist = []

    def find_chessboard_corners(self, row_corners, column_corners, visualize=False):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((row_corners * column_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[0:column_corners, 0:row_corners].T.reshape(-1, 2)

        # Make a list of calibration images
        images = glob.glob('../camera_cal/calibration*.jpg')

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
        # Test undistortion on an image
        img = cv2.imread('../test_images/straight_lines1.jpg')
        img_size = (img.shape[1], img.shape[0])

        # Do camera calibration given object points and image points
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, img_size, None, None)

        dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        cv2.imwrite('../output_images/straight_lines1_undist.jpg', dst)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open("../camera_cal/wide_dist_pickle.p", "wb"))
        # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        if visualize is True:
            # Visualize undistortion
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=30)
            ax2.imshow(dst)
            ax2.set_title('Undistorted Image', fontsize=30)

    # Edit this function to create your own pipeline.
    def pipeline(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:, :, 1]
        s_channel = hsv[:, :, 2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

        color_binary = np.zeros_like(sxbinary)
        color_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        return color_binary

    def execute_pipeline(self):
        image = mpimg.imread('../output_images/straight_lines1_undist.jpg')

        result = self.pipeline(image)

        mpimg.imsave('../output_images/straight_lines1_thres.jpg', result)

        # cv2.imwrite('../output_images/test1_thres.jpg', result)

        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=40)

        ax2.imshow(result)
        ax2.set_title('Pipeline Result', fontsize=40)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def warp(self):
        undist = cv2.imread('../output_images/straight_lines1_undist.jpg')
        img_size = (undist.shape[1], undist.shape[0])

        src = np.float32([[582, 460], [702, 460], [1102, 719], [206, 719]])
        dst = np.float32([[320, 0], [980, 0], [980, 719], [320, 719]])

        perspective_M = cv2.getPerspectiveTransform(src, dst)
        # warped
        top_down = cv2.warpPerspective(undist, perspective_M, img_size, flags=cv2.INTER_LINEAR)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undist)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(top_down)
        ax2.set_title('Undistorted and Warped Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)