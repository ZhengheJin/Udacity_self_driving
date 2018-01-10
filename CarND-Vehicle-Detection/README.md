## Project 5 -- Vehicle Detection and Tracking


---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/car_notcar.png
[image2]: ./writeup_images/car_notcar_hog.png
[image3]: ./writeup_images/single_scale_slide_window.png
[image4]: ./writeup_images/subsampling.png
[image5]: ./writeup_images/heatmap.png
[image6]: ./writeup_images/final_pipeline_test.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook. And the function name is `get_hog_features`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=10`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.
One thing worth pointing out before discussion is that this section just documents the different settings I have tried and offers a general direction of how to do the parameter tuning. It does not aim to exhaustively explore all possible combinations of parameters.

Firstly, different color space are experimented and the following results are obtained. The other parameters are fixed as follows:

```Python
orient = 10  
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (16, 16)
hist_bins = 16
spatial_feat = True
hist_feat = True
hog_feat = True
```

| Color Space        | Test accuracy   |
|:-------------:|:-------------:|
| RGB       | 0.9772        |
| HSV      | 0.9893      |
| LUV      | 0.9864      |
| HLS     | 0.984      |
| YUV     | 0.9878      |
| YCrCb     | 0.991      |

The highest accuracy is achieved when using color space 'YCrCb'.

Secondly, various number of orientations are tried
`orient = 8, 9, 10, 11`
The other parameters are fixed (same as the ones used in previous paragraph) with `color_space = 'YCrCb'`.

| orient        | Test accuracy   |
|:-------------:|:-------------:|
| 8       | 0.9893        |
| 9      |  0.9862     |
| 10     | 0.991      |
| 11     | 0.9904   |

10 or 11 orientations give similar accuracy results. Considering one more orientation may help the SVM classifier better distinguish the car and non-car clips (since more features are available)
I choose 11 orientations for following calculations.

Thirdly, different pix_per_cell values are tested
`pix_per_cell = 8, 12, 16`.
The other parameters are fixed (same as the ones used in previous paragraph) with `color_space = 'YCrCb' and orient = 11`

| pix_per_cell        | Test accuracy   |  
|:-------------:|:-------------:|
| 8       | 0.9904        |
| 12      | 0.9879      |  
| 16     | 0.987      |  

8 pixels per cell provides the highest test accuracy, so I will use `pix_per_cell = 8` for the subsequent investigations. Note that here I am trying to pick up parameters that offer highest accuracy. In case that calculation speed is the top priority, trade-offs can be made to use sub-optimal values with slight performance degradation. Also I did not carry out the rigorous cross experimenting, meaning grid search to find the best possible combinations.

After experimenting with these different combinations of parameters. I decided to use the following set of values:

```Python
color_space = 'YCrCb'
orient = 11  
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL"
spatial_size = (16, 16)
hist_bins = 16
spatial_feat = True
hist_feat = True
hog_feat = True
```

Of course, there is still room for parameter tuning, like changing the default value of hist_bins and spatial_feat, playing with the hog_channel, etc. However, due to time constraint, these are not tested and the curent setup also offers reasonably good results.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In the 11th code cell of jupyter notebook `Vehicle_Detection_and_Tracking.ipynb`,
I trained a linear SVM using the car, non-car data downloaded from the Udacity website. There are 8792 cars and 8968 non-car images. The feature vectors are constructed using the parameters specified in previous section with all three features (`spatial_feat`, `hist_feat` and `hog_feat`)  enabled.   The values are normalized using the `StandardScaler()` from Python's sklearn package.

```Python
X = np.vstack((car_features, notcar_features)).astype(np.float64)  
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First I tested the sliding window search by adopting the functions form the lecture: `slide_window()` takes an image, start/step positions, window size and overlap and outputs a list of window positions. `search_windows()` takes in output of `slide_window()` and an image and generates the windows that give positive detection results. And this flow is applied to the six test images with a single size of window, the results of which are shown below (it is also shown in the jupyter notebook output of cell 12). In next section, an improved version of the implementation is demonstrated.

![alt text][image3]

As can be seen from these output images, the single scale window sliding method misses the car in the 3rd image (right image on the first row). Also there are few false positives in image 1 and 5 (left image on the first row and middle image on the second row).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Following the suggestions in the lecture, a new function `find_cars()` is defined, which takes in an image and performs the sub-sampling and making detections using the trained SVM. Code is in the 14th cell.
This function is used with `ystart = 400 ystop = 656 scale = 1.5` and applied to the same set of test images, the results are shown below.
Other parameters are set to the default values, namely 8 x 8 cells per window and `cells_per_step = 2` (75% overlapping).


![alt text][image4]
---

Apparently, this method improves the results. The false positives in image 1 and 5 are rejected and also the small car in 3rd image is also detected. However, there is still one false positive in image 4.
Next section will talk about how to implement a mechanism to further invalidates the false positives, if there is any. Also how to combing different detection windows into one bounding box is also discussed.

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out_save_laste10_frames.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from one of the test images, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image:

### Here are one of the test images ("test5.jpg" in test5.jpg folder) and its corresponding heatmaps:

![alt text][image5]

Finally, a image/video processing pipeline is built using the functions/routines I have so far. 6 different scales (`scale = 1.0, 1.5, 2.0, 2.5, 3.0, 3.5`) are used to generated different window sizes to search in the image. The search area is kept the same with `ystart = 400 ystop = 656`. This is in code cell 20 in the jupyter notebook.


### Here the resulting bounding boxes are drawn onto the six test images after applying the heatmap generation and false positive filtering using thresholding algorithm:
![alt text][image6]

As can be seen from these pictures, no false positives are shown and each box corresponds a car being identified.

Besides defining a heatmap and using different scales for searching. I also implemented a basic list to keep the positive boxes found in previous frames (10 frames) and rejecting the detections that below the threshold, which can be found in code cell 19 and 20.
```Python
def add_boxes(self, boxes):
    self.prev_boxes.append(boxes)
    if len(self.prev_boxes) > 10:
        # remove old boxes when overflow
        self.prev_boxes = self.prev_boxes[len(self.prev_boxes)-10:]
```
```Python
if len(box_list) > 0:
    det.add_boxes(box_list)

heat = np.zeros_like(image[:,:,0]).astype(np.float)

for box_list in det.prev_boxes:
    heat = add_heat(heat, box_list)
heat = apply_threshold(heat, 1+ len(det.prev_boxes)//2)
```


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. How to treat the occlusion.  In 26' second of the video, the black car appears and occludes the white car. The pipeline loses tracking of the white car for one or two seconds. We may need to extrapolate the trajectory of the white to predict its movement.

2. How to reduce the process time and make it real-time detector. The pipeline implemented in this project takes 40-50 mins to process the 50-second video, which is already slow for offline analysis. Different classification and detection methods or frameworks may be needed to improve the calculation speed, such as CNN.

3. Further improving the detection accuracy and reducing the false positives. At 42' second of the video, there are some false boxes being detected. Like briefly discussed in the beginning of the writeup, more parameter tuning can be done to get a better classifier and also the heatmap algorithm can be further enhanced to reject more false detections.
