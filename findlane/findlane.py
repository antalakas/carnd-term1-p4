import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
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
        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        # Grayscale image
        # NOTE: we already saw that standard grayscaling lost color information for the lane lines
        # Explore gradients in other colors spaces / color channels to see what might work better
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

        print(sxbinary.shape)
        print(s_binary.shape)
        print(color_binary.shape)

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')

        return combined_binary

        # img = np.copy(img)
        # # Convert to HSV color space and separate the V channel
        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        # l_channel = hsv[:, :, 1]
        # s_channel = hsv[:, :, 2]
        # # Sobel x
        # sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
        # abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        # scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        #
        # # Threshold x gradient
        # sxbinary = np.zeros_like(scaled_sobel)
        # sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
        #
        # # Threshold color channel
        # s_binary = np.zeros_like(s_channel)
        # s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # # Stack each channel
        # # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # # be beneficial to replace this channel with something else.
        # # color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
        #
        # color_binary = np.zeros_like(sxbinary)
        # color_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        #
        # return color_binary

    def execute_pipeline(self):
        image = mpimg.imread('../output_images/straight_lines1_undist.jpg')

        result = self.pipeline(image)

        mpimg.imsave('../output_images/straight_lines1_thres.jpg', result, cmap=cm.gray)

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
        undist = cv2.imread('../output_images/straight_lines1_thres.jpg')
        img_size = (undist.shape[1], undist.shape[0])

        src = np.float32([[582, 460], [702, 460], [1102, 719], [206, 719]])
        dst = np.float32([[320, 0], [980, 0], [980, 719], [320, 719]])

        perspective_M = cv2.getPerspectiveTransform(src, dst)
        # warped
        top_down = cv2.warpPerspective(undist, perspective_M, img_size, flags=cv2.INTER_LINEAR)

        # f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(undist)
        # # ax1.set_title('Original Image', fontsize=50)
        # ax2.imshow(top_down)
        # # ax2.set_title('Undistorted and Warped Image', fontsize=50)
        # # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        #
        # histogram = np.sum(top_down[top_down.shape[0] // 2:, :], axis=0)
        # ax3.plot(histogram)

        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        # f.tight_layout()
        # ax1.imshow(undist)
        # ax1.set_title('Original Image', fontsize=50)
        # ax2.imshow(top_down)
        # ax2.set_title('Undistorted and Warped Image', fontsize=50)
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

        # histogram = np.sum(top_down[top_down.shape[0] // 2:, :], axis=0)
        # plt.plot(histogram)

        return top_down

    def find_lane(self):

        binary_warped = self.warp()

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

        print(binary_warped.shape)
        print(out_img.shape)
        print(out_img[0].shape)

        plt.imshow(out_img[0])

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # # Draw the windows on the visualization image
            # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        # right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)