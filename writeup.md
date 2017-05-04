**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./test_images/test5.jpg "Input Image"
[image2]: ./output_images/undist_test5.jpg "Undistorted Image"
[image3]: ./pipeline_images/thres_undist_test5.jpg "Thresholded Binary Image"
[image4]: ./pipeline_images/warp_thres_undist_test5.jpg "Warped Image"
[image5]: ./pipeline_images/histogram.png "Histogram"
[image6]: ./pipeline_images/line_fit.png "Fit Visual"
[image7]: ./pipeline_images/output.png "Output"
[video1]: ./output_video/out_project_video_.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file/class `./findlane/calibratecamera.py`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. 
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  
Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `
imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  
I found one image in the set that retuned an error after running against `cv2.calibrateCamera()` 
I also persist camera parameters to disk in file `./camera_cal/wide_dist_pickle.p`. 
This file is loaded from module `./findlane/findlane.py` to undistort images for testing the pipeline, as well as to produce the project video.

The calibration procedure is executed from the command line by issuing `python ./findlane/cli.py --cal 1` and undistorts all images from `./test_images`,
persisting them in `./output_images`

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]
(Input / Original image)

### Pipeline (single images)

The pipeline procedure (for all undistorted images found in `./output_images`) is executed from the command line by issuing `python ./findlane/cli.py --img 1` and persists 
images of the procedure, for each image in `./output_images`

The function that implements the pipeline is located in line 146 of `./findlane/findlane.py`.

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one (unistorted):
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 140 through 155 in `./findlane/findlane.py`).
Supporting code for this step is contained in `./findlane/threshold.py`
Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines 51 through 73 in the file `./findlane/findlane.py`
The `warp()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  
I chose to hardcode the source and destination points in the following manner (lines 31-32 in `./findlane/findlane.py`):

```python
        self.wrap_src = np.float32([[595, 450], [686, 450], [1102, 719], [206, 719]])
        self.wrap_dst = np.float32([[320, 0], [980, 0], [980, 719], [320, 719]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped 
counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding window method to fit polynomials to the lines lines, after locating them using the histogram method.
The previous picture depicts the lane finding, along with the warping procedure.

Also, in lines 143-154 in file `./findlane/lane.py`, the left and right lines are averaged using data from previous frames 
in order to smooth the result.

The following picture demonstrates the actual polynomial fit:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 177 through 261 in my code in `./findlanes/findlanes.py`. Particularly function `calculate_position()`, calculates the position in meters, 
`calculate_curvatures()` calculates the curvature for both lines, in meters, while `plot_dashboard()` makes the calls to the previous functions, also calculates
the distance of the center of the vehicle to both lines and draws a "dashboard" to provide visual feedback.  

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 322 through 346 in my code in `./findlane/findlane.py` in the function `execute_image_pipeline()`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  
Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  
What could you do to make it more robust?

I thought that it would be fairly easy to perform well at least in the challenge video, but unfortunately this was not the case. I used methodology from P1 to mask out irrelevant pixels 
outside the lane's "bounding box", i averaged over the lines' data, i tried different combinations for color thresholding, also HSV, it seems that given the time i could provide
i reached a fair result only for the project video. The pipeline is likely to fail when the lines are not clearly visible, as well as when the light intensity of the road changes
going from normal to high (there is such situation in the middle of the project video) or normal to dark (tree shadow). I would like to improve current situation, first i need to
develop better intuition for the techniques involved, probably also apply filters (automatically adaptable) depending on the condition of the road. Probably also based on the 
knowledge of the position of the car i should use it to adapt the mask that masks out irrelevant pixels dynamically. I would also like to try the convolutional approach of sliding window
as described in the lesson.
