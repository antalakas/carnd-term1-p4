import numpy as np
import cv2


class Threshold(object):
    def __init__(self):
        pass

    def sobel_x_threshold(self, img, sx_thresh):
        # Grayscale image
        # img is the undistorted image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sobel x
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
        abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        return sxbinary

    def sobel_thresh(self, img, sx_thresh, sy_thresh, kernel_size=3):
        # Grayscale image
        # img is the undistorted image
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=kernel_size, thresh=sx_thresh)
        grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=kernel_size, thresh=sy_thresh)

        return gray, gradx, grady

    def color_thresh(self, img, s_thresh):

        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # mask_white = cv2.inRange(gray, 200, 255)
        # image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # lower_yellow = np.array([20, 100, 100], dtype="uint8")
        # upper_yellow = np.array([30, 255, 255], dtype="uint8")
        # mask_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
        # # Retrieve the final masked image
        # mask_yellow_white = cv2.bitwise_or(mask_white, mask_yellow)
        # plt.imshow(mask_yellow_white, cmap='gray')

        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]

        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        return s_binary

    # Applies Sobel x or y
    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Apply x or y gradient
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        elif orient == 'y':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute values
        sobel = np.absolute(sobel)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
        # Return the result
        return binary_output

    # Calculate magnitude of the gradient
    def mag_thresh(self, gray, sobel_kernel=3, mag_thresh=(0, 255)):
        # Apply x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        binary_output[(scaled_sobel > mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
        # Return the result
        return binary_output

    # Calculate direction of gradient
    def dir_threshold(self, gray, sobel_kernel=3, thresh=(0, np.pi / 2)):
        # Apply x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        # Error statement to ignore division and invalid errors
        with np.errstate(divide='ignore', invalid='ignore'):
            absgraddir = np.absolute(np.arctan(sobely / sobelx))
            dir_binary = np.zeros_like(absgraddir)
            dir_binary[(absgraddir > thresh[0]) & (absgraddir < thresh[1])] = 1
        # Return the result
        return np.asarray(dir_binary, dtype="uint8")
