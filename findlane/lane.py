import numpy as np
import cv2
import matplotlib.pyplot as plt
from line import Line

class Lane(object):
    def __init__(self):

        self.n = 30

        self.left_line = Line()
        self.right_line = Line()

        self.left_line.current_fit = [0, 0, 0]
        self.right_line.current_fit = [0, 0, 0]

        self.init_diff = True

    def find(self, binary_warped, test_image_pipeline=False):
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        out_img = None

        if test_image_pipeline:
            # Create an output image to draw on and visualize the result
            out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

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

            if test_image_pipeline:
                # Draw the windows on the visualization image
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

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
        self.left_line.allx = nonzerox[left_lane_inds]
        self.left_line.ally = nonzeroy[left_lane_inds]
        self.right_line.allx = nonzerox[right_lane_inds]
        self.right_line.ally = nonzeroy[right_lane_inds]

        # save previous fit values for later use
        left_line_previous_fit = self.left_line.current_fit[:]
        right_line_previous_fit = self.right_line.current_fit[:]

        # Fit a second order polynomial to each
        self.left_line.current_fit = np.polyfit(self.left_line.ally, self.left_line.allx, 2)
        self.right_line.current_fit = np.polyfit(self.right_line.ally, self.right_line.allx, 2)

        # find difference in fit coefficients between current and previous image
        self.left_line.diffs = left_line_previous_fit - self.left_line.current_fit
        self.right_line.diffs = right_line_previous_fit - self.right_line.current_fit

        if self.init_diff is True:
            self.init_diff = False
        else:
            if not test_image_pipeline:
                if abs(self.right_line.diffs[2]) > 250:
                    self.right_line.current_fit = right_line_previous_fit[:]

        # self.left_line.recent_fit.append(self.left_line.current_fit)
        # if len(self.left_line.recent_fit) > self.n:
        #     self.left_line.recent_fit.pop(0)
        # # print( self.left_line.recent_fit)
        # # print(np.average(self.left_line.recent_fit, axis=0))
        # self.left_line.best_fit = np.average(self.left_line.recent_fit, axis=0)
        # # print(self.left_line.best_fit)
        #
        # self.right_line.recent_fit.append(self.right_line.current_fit)
        # if len(self.right_line.recent_fit) > self.n:
        #     self.right_line.recent_fit.pop(0)
        # # print( self.right_line.recent_fit)
        # # print(np.average(self.right_line.recent_fit, axis=0))
        # self.right_line.best_fit = np.average(self.right_line.recent_fit, axis=0)
        # # print(self.right_line.best_fit)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        # left_fitx = self.left_line.best_fit[0] * ploty ** 2 + self.left_line.best_fit[1] * ploty + self.left_line.best_fit[2]
        # right_fitx = self.right_line.best_fit[0] * ploty ** 2 + self.right_line.best_fit[1] * ploty + self.right_line.best_fit[2]
        left_fitx = self.left_line.current_fit[0] * ploty ** 2 + self.left_line.current_fit[1] * ploty + self.left_line.current_fit[2]
        right_fitx = self.right_line.current_fit[0] * ploty ** 2 + self.right_line.current_fit[1] * ploty + self.right_line.current_fit[2]

        if test_image_pipeline:
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        if test_image_pipeline:
            return out_img, ploty, left_fitx, right_fitx
        else:
            self.left_line.recent_xfitted.append(left_fitx)
            if len(self.left_line.recent_xfitted) > self.n:
                self.left_line.recent_xfitted.pop(0)
            self.left_line.bestx = np.average(self.left_line.recent_xfitted, axis=0)

            self.right_line.recent_xfitted.append(right_fitx)
            if len(self.right_line.recent_xfitted) > self.n:
                self.right_line.recent_xfitted.pop(0)
            self.right_line.bestx = np.average(self.right_line.recent_xfitted, axis=0)

            self.left_line.detected = True
            self.left_line.detected = True

            return out_img, ploty, self.left_line.bestx, self.right_line.bestx
