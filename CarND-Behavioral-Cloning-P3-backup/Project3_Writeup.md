**Behavioral Cloning**
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center_2017_10_27_18_49_38_859.jpg "center"
[image2]: ./writeup_images/clock_wise_center_2017_10_30_22_11_16_085.jpg "clock-wise"
[image3]: ./writeup_images/before_flip_clock_wise_center_2017_10_30_22_09_14_000.jpg "before flipping"
[image4]: ./writeup_images/after_flip_clock_wise_center_2017_10_30_22_09_14_000.png "after flipping"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---


## **Files Submitted & Code Quality**

**1. Submission includes all required files and can be used to run the simulator in autonomous mode**

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode (increased the speed to 15mph)
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

**2. Submission includes functional code**

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

**3. Submission code is usable and readable**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## **Model Architecture**

**1. An appropriate model architecture has been employed**

My model consists of a convolution neural network of three 5x5 convolutional layers followed by two 3x3 convolutional layers followed by four fully-connected layers. The details of the dimensions are list below. (model.py lines 132-150)

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 color image   							|
| Layer 1 Convolution 5x5     	| output shape =(24,5,5), subsample=(2, 2), valid padding, activation = RELU 	|
| Layer 2 Convolution 5x5					|	output shape =(36,5,5), subsample=(2, 2), valid padding, activation = RELU											|
| Layer 3 Convolution 5x5	      	| output shape =(48,5,5), subsample=(2, 2), valid padding, activation = RELU 				|
| Dropout					|					rate = 0.1							|
| Layer 4 Convolution 3x3	    | output shape =(64,3,3), subsample=(1, 1), valid padding, activation = RELU     									|
| Layer 5 Convolution 3x3	    | output shape =(64,3,3), subsample=(1, 1), valid padding, activation = RELU     									|
| Flatten					|										|
| Layer 6 fully-connected			    |								output number = 100				|
| Dropout					|					rate = 0.1							|
| Layer 7 fully-connected					|						    output number = 50				|
| Dropout					|					rate = 0.1							|
| Layer 8 fully-connected					|								output number = 10				|
| Dropout					|					rate = 0.1							|
| Layer 9 fully-connected					|								output number = 1				|


The model includes RELU layers to introduce nonlinearity (model.py lines 136-142), and the data is normalized in the model using a Keras lambda layer (model.py lines 133).

**2. Attempts to reduce overfitting in the model**

The model contains dropout layers in order to reduce overfitting (model.py lines 140, 145, 147, 149). And the dropout rate is set to 0.1.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 23, split the training and validation dataset; line 156, the actual model fitting). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

**3. Model parameter tuning**

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 153).

**4. Appropriate training data**

Training data was chosen to keep the vehicle driving on the road. Two different sets of raw datasets are used. One is the counter-clock wise driving dataset and the other the clock wise driving dataset.

For details about how I created the training data, see the next section.

## **Training Strategy**

**1. Solution Design Approach**

The overall strategy for deriving a model architecture was to start with a basic model that can keep the car moving forward and then gradually increase the model complexity so that it can fit the training data better and generates smooth driving trajectory.
Some of my intermediate trained model are saved in "fitted_models" folder.

* My first step was to repeat what is taught in the video lectures, starting with the simplest regression model without neural network. The trained model is saved as "simple_lr_model.h5". The test drive turns out to be terrible that the car drove off road immediately after start. ("simple_lr_model.h5")

* Secondly, I switched to the google Lenet and tried few preprocessing techniques, normalization and converting the images from BGR to RGB (since OpenCV reads in the images in BGR format). This drastically reduced the training loss and the car stay in the track until hits the first curve. ("lr_prepro_lenet_rgb_model.h5")

* Thirdly, I tried the data augmentation by flipping the images. This enhances the training by teach the car how to steer in the opposite directions (clock-wise instead of the defaul counter-clock wise). The result keeps improving that the car can move after the curve but drives into the lake that near the second curve. ("lr_prepro_lenet_rgb_flip_model.h5")

* Fourthly, I tried more powerful NVIDIA CNN architecture with three 5x5 convolutional layers followed by two 3x3 convolutional layers followed by four fully-connected layers. The details of this model is documented in above section.
The car can stay in the track for the first two curves now, but still drives off road near the curve after the bridge. ("lr_prepro_Nvidia_rgb_flip_model.h5")

* Fifthly, I added dropout layers and tried different dropout rates (0.1, 0.5, 0.9). Also different number of epochs are investigated (3, 4, 5, 6). Also I collected more data by adding a reverse driving in clock-wise. The result is getting even better that the car can make through the curve after the lake now. ("lr_prepro_Nvidia_rgb_flip_generator_epoch6_dropout0.1_reversetrack_model.h5")

* Finally, the model is finalized with more facilities/functionalities such as adding the generator, so that the training data does not need to be read into the memory all at once. Instead they can be read in batches, so that when presented with large amount of data, the model's training can still be fit into memory. ("lr_prepro_Nvidia_rgb_flip_generator_epoch3_dropout0.1_reversetrack_try2_model.h5")

At the end of the above processes/improvements, the vehicle is able to drive autonomously around the track without leaving the road.

**2. Final Model Architecture**

The final model architecture (model.py lines 133-150) consisted of a convolution neural network with three 5x5 convolutional layers followed by two 3x3 convolutional layers followed by four fully-connected layers. The details of the dimensions are list in the table above.

**3. Creation of the Training Set & Training Process**

To capture good driving behavior, I first recorded one and half laps on track one using center lane, counter-clock wise driving.
Here is an example image of center lane, counter-clock-wise driving:

![alt text][image1]

I then recorded the vehicle driving in clock-wise directions along the same track. This way the training data will not be biased towards only driving counter-clock wise. The steer angles will be more diverse, which will helps the model to learn and generalize.
Here is the image of center lane, clock-wise driving:

![alt text][image2]


To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image3]
![alt text][image4]

After the collection process, I had 6702 images from counter-clock-wise driving dataset and 5610 images from clock-wise driving dataset. I then preprocessed this data by doing normalization, BGR2RGB conversion.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by trying out different values (3, 4, 5, 6). I used an adam optimizer so that manually training the learning rate wasn't necessary.
