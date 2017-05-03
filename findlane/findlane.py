import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import pickle
import ntpath
from moviepy.editor import VideoFileClip
from threshold import Threshold
from lane import Lane


class FindLane(object):
    def __init__(self):

        self.image_shape = [0, 0]

        self.camera_calibration_path = '../camera_cal/'
        self.output_images_path = '../output_images/'
        self.input_video_path = '../input_video/'
        self.output_video_path = '../output_video/'

        self.sobel_kernel_size = 7
        self.sx_thresh = (60, 255)
        self.sy_thresh = (60, 150)
        self.s_thresh = (170, 255)
        self.mag_thresh = (40, 255)
        self.dir_thresh = (.65, 1.05)

        self.wrap_src = np.float32([[595, 450], [686, 450], [1102, 719], [206, 719]])
        # self.wrap_src = np.float32([[582, 460], [702, 460], [1102, 719], [206, 719]])
        self.wrap_dst = np.float32([[320, 0], [980, 0], [980, 719], [320, 719]])

        self.mask_offset = 30
        self.vertices = [np.array([[206-self.mask_offset, 719],
                                   [595-self.mask_offset, 460-self.mask_offset],
                                   [686+self.mask_offset, 460-self.mask_offset],
                                   [1102+self.mask_offset, 719]],
                                  dtype=np.int32)]
        # self.vertices = [np.array([[0, 719],
        #                            [595-self.mask_offset, 460-self.mask_offset],
        #                            [686+self.mask_offset, 460-self.mask_offset],
        #                            [1280, 719]],
        #                           dtype=np.int32)]
        # self.vertices = [np.array([[206-self.mask_offset, 719],
        #                            [582-self.mask_offset, 460-self.mask_offset],
        #                            [702+self.mask_offset, 460-self.mask_offset],
        #                            [1102+self.mask_offset, 719]],
        #                           dtype=np.int32)]

        self.thresh = Threshold()
        self.lane = Lane()

    def warp(self, img, visualize=False):
        img_size = (img.shape[1], img.shape[0])

        perspective_M = cv2.getPerspectiveTransform(self.wrap_src, self.wrap_dst)
        # warped
        top_down = cv2.warpPerspective(img, perspective_M, img_size, flags=cv2.INTER_LINEAR)

        top_down[:, 0:100] = 0
        top_down[:, top_down.shape[1]-100:top_down.shape[1]] = 0

        if visualize:
            f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(24, 9))
            f.tight_layout()
            ax1.imshow(img)
            # ax1.set_title('Original Image', fontsize=50)
            ax2.imshow(top_down)
            # ax2.set_title('Undistorted and Warped Image', fontsize=50)
            # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

            histogram = np.sum(top_down[top_down.shape[0] // 2:, :], axis=0)
            histogram[:180] = 0
            ax3.plot(histogram)

            # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
            # f.tight_layout()
            # ax1.imshow(img)
            # ax1.set_title('Original Image', fontsize=50)
            # ax2.imshow(top_down)
            # ax2.set_title('Undistorted and Warped Image', fontsize=50)
            # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

            # histogram = np.sum(top_down[top_down.shape[0] // 2:, :], axis=0)
            # plt.plot(histogram)

        return top_down


    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    # The pipeline.
    def pipeline(self, img, sobel_kernel_size=3, sx_thresh=(20, 100), sy_thresh=(20, 100), s_thresh=(170, 255),
                 mag_thresh=(10, 255), dir_thresh=(0, 1), visualize=False):

        # # Scenario 1
        # # def pipeline(self, img, s_thresh=(170, 255), sx_thresh=(20, 100), visualize=False):
        # sobel_thresh = self.thresh.sobel_x_threshold(img, sx_thresh)
        # print('sobel_thresh shape', sobel_thresh.shape)
        # color_thresh = self.thresh.color_thresh(img, s_thresh)
        # print('color_thresh shape', color_thresh.shape)
        #
        # # # Stack each channel to view their individual contributions in green and blue respectively
        # # # This returns a stack of the two binary images, whose components you can see as different colors
        # # color_binary = np.dstack((np.zeros_like(sobel_thresh), sobel_thresh, color_thresh))
        # # print('color_binary shape', color_binary.shape)
        #
        # # Combine the two binary thresholds
        # thresholded_binary = np.zeros_like(sobel_thresh)
        # thresholded_binary[(color_thresh == 1) | (sobel_thresh == 1)] = 1
        # print('thresholded_binary shape', thresholded_binary.shape)

        # Scenario 2
        # Perform Gaussian Blur
        kernel_size = 5
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        gray, gradx, grady = self.thresh.sobel_thresh(img, sx_thresh, sy_thresh, sobel_kernel_size)
        #
        # mag_binary = self.thresh.mag_thresh(gray, sobel_kernel=sobel_kernel_size, mag_thresh=mag_thresh)
        # dir_binary = self.thresh.dir_threshold(gray, sobel_kernel=sobel_kernel_size, thresh=dir_thresh)
        #
        # sobel_combined = np.zeros_like(dir_binary)
        # sobel_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        #
        # color_thresh = self.thresh.color_thresh(img, s_thresh)
        #
        # thresholded_binary = np.zeros_like(sobel_combined)
        # thresholded_binary[(color_thresh > 0) | (sobel_combined > 0)] = 1

        sobel_combined = np.zeros_like(gradx)
        sobel_combined[(gradx == 1) & (grady == 1)] = 1

        color_thresh = self.thresh.color_thresh(img, s_thresh)

        thresholded_binary = np.zeros_like(sobel_combined)
        thresholded_binary[(color_thresh > 0) | (sobel_combined > 0)] = 1

        # Masked area
        thresholded_binary = self.region_of_interest(thresholded_binary, self.vertices)
        thresholded_binary[0:450, :] = 0

        if visualize:
            # Plotting thresholded images
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            ax1.set_title('Initial image')
            ax1.imshow(img)

            ax2.set_title('Combined S channel and gradient thresholds')
            ax2.imshow(thresholded_binary, cmap='gray')

        warped = self.warp(thresholded_binary, False)
        out_img, ploty, left_fitx, right_fitx = self.lane.find(warped, False)

        return thresholded_binary, warped, out_img, ploty, left_fitx, right_fitx

    def calculate_position(self, pts):
        # Find the position of the car from the center
        # It will show if the car is 'x' meters from the left or right
        position = self.image_shape[1] / 2
        left = np.min(pts[(pts[:, 1] < position) & (pts[:, 0] > 700)][:, 1])
        right = np.max(pts[(pts[:, 1] > position) & (pts[:, 0] > 700)][:, 1])
        center = (left + right) / 2
        # Define conversions in x and y from pixels space to meters
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        return (position - center), (position - center) * xm_per_pix

    def calculate_curvatures(self, ploty, left_fitx, right_fitx):
        y_eval = np.max(ploty)

        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

        # Calculate the new radius of curvature
        left_curverad = \
            ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / \
                np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / \
                np.absolute(2 * right_fit_cr[0])

        # Now our radius of curvature is in meters
        return left_curverad, right_curverad

    def plot_dashboard(self, image, ploty, left_fitx, right_fitx, newwarp):
        left_curverad, right_curverad = self.calculate_curvatures(ploty, left_fitx, right_fitx)

        # Put text on an image
        font = cv2.FONT_HERSHEY_SIMPLEX

        text = "Radius of Left Line Curvature: {} m".format(int(left_curverad))
        cv2.putText(image, text, (100, 50), font, 1, (255, 255, 255), 2)

        text = "Radius of Right Line Curvature: {} m".format(int(right_curverad))
        cv2.putText(image, text, (100, 100), font, 1, (255, 255, 255), 2)

        # Find the position of the car
        pts = np.argwhere(newwarp[:, :, 1])
        position_pixels, position_meters = self.calculate_position(pts)

        if position_meters < 0:
            text = "Vehicle is {:.2f} m left of center".format(-position_meters)
        else:
            text = "Vehicle is {:.2f} m right of center".format(position_meters)
        cv2.putText(image, text, (100, 150), font, 1, (255, 255, 255), 2)

        return image


    def project_back(self, undist, thresholded_binary, warped, ploty, left_fitx, right_fitx):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        Minv = cv2.getPerspectiveTransform(self.wrap_dst, self.wrap_src)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (thresholded_binary.shape[1], thresholded_binary.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        # print('result shape', result.shape)
        # plt.imshow(result)

        return result, newwarp

    def execute_image_pipeline(self, visualize=False):
        images = glob.glob(self.output_images_path + 'undist_*.jpg')

        for idx, fname in enumerate(images):
            image_fname = ntpath.basename(fname)

            print("Processing", fname)

            image = mpimg.imread(self.output_images_path + image_fname)

            self.image_shape = image.shape

            thresholded_binary, warped, out_img, ploty, left_fitx, right_fitx = \
                self.pipeline(
                    image,
                    sobel_kernel_size=self.sobel_kernel_size,
                    sx_thresh=self.sx_thresh,
                    sy_thresh=self.sy_thresh,
                    s_thresh=self.s_thresh,
                    mag_thresh=self.mag_thresh,
                    dir_thresh=self.dir_thresh,
                    visualize=True)

            result, newwarp = self.project_back(image, thresholded_binary, warped, ploty, left_fitx, right_fitx)

            result = self.plot_dashboard(result, ploty, left_fitx, right_fitx, newwarp)

            mpimg.imsave(self.output_images_path + 'thres_' + image_fname, thresholded_binary, cmap=cm.gray)
            mpimg.imsave(self.output_images_path + 'warp_' + 'thres_' + image_fname, warped, cmap=cm.gray)
            mpimg.imsave(self.output_images_path + 'out_' + 'warp_' + 'thres_' + image_fname, out_img)
            mpimg.imsave(self.output_images_path + 'result_' + 'out_' + 'warp_' + 'thres_' + image_fname, result)

            if visualize:
                # Plot the result
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                f.tight_layout()

                ax1.imshow(image)
                ax1.set_title('Original Image', fontsize=40)

                ax2.imshow(thresholded_binary, cmap='gray')
                ax2.set_title('Thresholded Result', fontsize=40)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

                ax2.imshow(warped, cmap='gray')
                ax2.set_title('Warped Result', fontsize=40)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

                ax2.imshow(out_img, cmap='gray')
                ax2.set_title('Pipeline Result', fontsize=40)
                ax2.plot(left_fitx, ploty, color='yellow')
                ax2.plot(right_fitx, ploty, color='yellow')
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    def execute_video_pipeline(self, image):

        handle = open(self.camera_calibration_path + "wide_dist_pickle.p", 'rb')
        dist_pickle = pickle.load(handle)

        undist_image = cv2.undistort(image, dist_pickle["mtx"], dist_pickle["dist"], None, dist_pickle["mtx"])

        self.image_shape = undist_image.shape

        thresholded_binary, warped, out_img, ploty, left_fitx, right_fitx = self.pipeline(undist_image)
        result, newwarp = self.project_back(image, thresholded_binary, warped, ploty, left_fitx, right_fitx)

        return self.plot_dashboard(result, ploty, left_fitx, right_fitx, newwarp)

    def project_video(self):
        # Execute the pipeline for the video file
        in_project_video = self.input_video_path + 'project_video.mp4'
        project_clip = VideoFileClip(in_project_video)

        out_project_clip = project_clip.fl_image(self.execute_video_pipeline)  # NOTE: this function expects color images!!

        out_project_video = self.output_video_path + 'out_project_video.mp4'
        out_project_clip.write_videofile(out_project_video, audio=False)
