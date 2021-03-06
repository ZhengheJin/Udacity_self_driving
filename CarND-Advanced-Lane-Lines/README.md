## Writeup 
---

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

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.png "Binary Example"
[image4]: ./output_images/warped_straight_lines.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell (section Camera Calibration) of the IPython notebook located in "p4.ipynb". 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, thresholding steps at in p4.ipynb (Section starts from "Define a function that applies Sobel x or y, then takes an absolute value and applies a threshold" to "Combine the Saturation and Red channel filters").  Here's an example of my output for this step. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `unwarper()`, which appears in the p4.ipynb (Section "Image undistortion on one of the test images").  The `unwarper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32([(581,460), (705,460), (279,675), (1042,675)])
dst = np.float32([(300,0), (w-300,0), (300,h), (w-300,h)])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 581, 460      | 300, 0        | 
| 705, 460      | 980, 0      |
| 279, 675     | 300, 720      |
| 1042, 675      | 980, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in **Define the function for calcuating the curvature** part of the ipython notebook.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I did this in **Define the function for calcuating the distance to lane center** part of the ipython notebook.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult part in this project is to generate a reasonably good binary image using the combination of the pre-precosssing filters. As mentioned earlier in this report, quite a few different pre-processing combinations have been experimented (abs_sobel_thresh + mag_binary + dir_binary or HSL+abs_sobel_thresh or HSL + RGB). The saturation channel and red channel combination seems most promising. However, this combination is not very efficient in isolating the pixels for the white lane. It turns out that some background white pixels in the adjacent lane would "disrupt" the white lane detection if just using the "saturation + red" combination. After close examination of the individual output of the saturation and red channel output, I realize that the "noise" pixels near the white lane is generated by the saturation filter. Therefore, instead of doing a straightforward stacking of the two channel images (combined_binary[(hls_binary == 1) | (red_binary == 1)] = 1), I used a half-image-size mask to mask out the right half of the output from the saturation filter as shown below.
```Python
l_r_midpoint = 660
w = hls_binary.shape[1]
hls_binary[:, l_r_midpoint:w] = 0
```
In addition to the above procedure, a pool of fitted results is also maintained to keep track of the fitted lanes from the last few frames, which is shown in the function `def update_fit(self, fit, inds)` inside `class Line()`.

Beside these techniques, other possible methods can also be tried to improve the robustness of the results, such as enforce the equal distance between two lanes, try different color space to do more combinations of the filters to generate binary images. However, due to the time constraint, these ideas may be experimented during future revisits.
