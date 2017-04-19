#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./images/model_summary.png =650x450 "Model Visualization"
[image2]: ./images/loss.png =650x450 "Loss"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* `model.py` containing the script to create and train the model.
* `drive.py` for driving the car in autonomous mode. The original version is used for testing. 
* `model.h5` containing a trained convolution neural network.
* `video.mp4` containing a video where the car drives on track 1.
* `writeup_report.md` summarizing the results.

####2. Submission includes functional code
Using the Udacity provided simulator and `drive.py` file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```


####3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The model is based on [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with an extra fully connected layer. 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

At the beginning of the model, the data is normalized using a Keras lambda layer and is also cropped to get rid of unnecessary backgroud images. (The model starts at the comment saying 'Model' in `model.py`) 

The model is based on NVIDIA architecture which consists of a convolution neural network with `3x3` filter and `5x5` sizes and depths between `24` and `64`. The model includes RELU layers to introduce nonlinearity after each convolution layer. The only difference is an extra fully connected layer where the number of the neurons is `25`. 


####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (The training and validation start at the comment saying 'Training and validation'). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model does not contain any dropout layers. The reasons of this decision are

1. As explained later, the model based on NVIDIA architecture was able to generate a good fit of the data.
2. After several trial-and-errors, it turned out that the model based on NVIDIA architecture with dropout layers did not generate a better fit of the data than the model based on NVIDIA architecture. 
3. The test results of the model based on NVIDIA architecture with dropout layers did not meet the specification, either. 

####3. Model parameter tuning

The model used an adam optimizer. The default settings of the parameters such as the learning rate are used. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides around the curves, driving the curves. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was iterative. 

1. Define a model
2. Collect training data
3. Test the model on the simulator
4. Modify the parameters and/or the model architecture

At Step 1, it was decided to use [NVIDIA architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) because it has been proven to be appropriate for this problem, then would be a good starting point. 

At Step 2, the training data was collected using the simulator and no training data provided by the lesson were used. It will be explained what training data was collected. 

At Step 3, the simulator was used to test the model as instructed. 

At Step 4, mainly the following parameters were modified: 

* Steering correction: the parameter to adjust the steering angles for the left and right images
* Batch size: the data batch size to be fed to the model
* Number of epochs: the number of iteration of the training

For the modification of the model architecture, the following things were attempted.

* Adding dropout layers
* Adding fully connected layers

After Step 1, the pair of Step 2 and Step 3, or the pair of Step 3 and Step 4 was iterated depending on the test result. 

Here is the process I took to define the final model architecture. 

1. Collected training data of center lane driving x 3, curve driving x the number of curves, constant wandering driving x 1
2. Tested the training data and found that the car got stuck at the side of the road. 
3. Collected training data of center lane driving x 3, curve driving x the number of curves, recovering from the left and right sides around the curves x the number of curves in order to exclude steering data to go to the side of the road.
4. Tested the training data and found that the car went out from the road from curves. 
5. Collected training data center lane driving x 3, curve driving x the number of curves, recovering from the left and right sides around the curves x the number of curves x 2 in order to increase steering data to recover from the side of the road
6. Tested the training data and found that the car still went out from the road at curves.
7. Supposed that enough training data was collected, then attempted to modify the model architecture. Added a dropout layer after each convolution layer or fully connected layer and increased the batch size and the number of epochs because the behavior of the training became probabilistic. 
8. Tested the same training data several times while changing the place to insert a dropout layer and modifying the batch size and the number of epochs and found that the result did not get any better. 
9. Gave up to add the dropout layer, then attempted to get better training result by making the model architecutre more complex. Both of the losses of training data and validation data were low, so it was supposed that overfitting was not occurring. Then, decided to add a fully connected layer to make the model architecture more complex a bit. 
10. Tested the training data and found that the car still went out from the road at curves. Got stuck here for a long time. 
11. Modified the steering correction at the range of 0.1 to 0.3.
12. Tested the same training data with the steering correction 0.15 and found that the test managed to go well finally. 
13. Collected training data center lane driving x 2, curve driving x the number of curves, recovering from the left and right sides around the curves x the number of curves x 2 and clockwise center lane drving x 1 in order to get a better test result. 
14. Tested the training data and found that the test went well a little bit better. Stopped training and testing here. 

####2. Final Model Architecture

The final model architecture is as below. The only difference from NVIDIA architecture is an extra fully connected layer where the number of the neurons is `25`. 

![alt text][image1]

Here is a graph that shows the losses of the training and validation data for each epoch using the final model architecture. 

![alt text][image2]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
