#**Behavioral Cloning** 

##Writeup

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model_image]: ./model.png "Model"
[raw_data]: ./examples/raw_data.png "Raw data"
[augmented_data]: ./examples/augmented_data.png "Augmented data"
[preprocessed_data]: ./examples/preprocessed_data.png "Preprocessed data"
[steering_angles_distribution]: ./examples/steering_angles_distribution.png "Steering angles distribution"
[loss_plot]: ./examples/loss_plot.png "Loss plot"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:  
* model.py containing the script to create and train the model  
* drive.py for driving the car in autonomous mode  
* model.h5 containing a trained convolution neural network  
* writeup_report.md and writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network, as well as code for generating training and validation data. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a 3 blocks of batch normalization, dropout, and 2 succesive 3x3 convolutions with elu nonlinearities. Convolutions use [24, 36, 36, 48, 48, 64] filters, following [NVIDIA's end to end learning paper] (https://arxiv.org/pdf/1604.07316.pdf).

Preprocessing model (image resizing, normalization and cropping) is found on lines 16-29 and prediction model is found on lines 49-83.

####2. Attempts to reduce overfitting in the model

The model contains a generous helping of batch normalization layers followed by dropout layers, lines 49-83.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 367). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer. Since batch normalization was used, initial learning rate is quite large, 0.1.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I use left side frames only when turning angle is at least 0.1 (i.e. car is turning to right at least a bit) and right frames only when turning angle is -0.1 or less (i.e. car is turning to left at least a bit). For left and right frames angles are bumped up slightly.

I used plenty of data augmentation. Images are randomly translated, rotated, brightened and flipped (functions __get\_augmented\_image()__ and __get\_single\_dataset\_generator()__).

In __get\_balanced\_paths\_angles\_tuples()__ I make sure (frame, steering angles) tuples returned by generators are approximately uniformly distributed across whole steering angles spectrum.

Finally I created different datasets concentrating on different parts of the task - smooth driving, curves, recovery - and using a generator that rotates between these datasets (rotating through separate generators for each dataset)

###Model Architecture and Training Strategy

####1. Solution Design Approach

Input frames are cropped to remove most of pixels above road surface, then resized to half its original size. Input is also normalized to (0, 1) range.

My first prediction model used 3x3 convolutions with filter sizes in 64~256 range and occasional maxpoolings until output was small (about 20x5), then a single dense layer with linear activation and mse error. After some playing with model I found that decreasing filters size didn't hurt performance at all, while increasing training speed. This brought me to model very similar to that from NVIDIA's paper.

This model was able to do some parts of the track, but had a large overfit to training data. Adding plenty of dropout and batch normalization help to bring training and validation loss closer to each other.

Important note on validation data - it was obtained from totally different runs that training data. This is important since just splitting single run data randomly would mean we might end up with nearly identical frames in training and validation sets, thus the validation loss there wouldn't really be a true representation of how well model generalizes.

A very important part in getting a working model was to make sure training data is evenly distributed between all steering angles.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.
One can vie the demonstration in the video attached with this submission, or in the following [Youtube video] (https://youtu.be/W_OpquHQedU)

####2. Final Model Architecture

The final model architecture (__model.py::get\_prediction_pipeline()__ lines 49-83) consisted of a convolution neural network with the following layers and layer sizes:

It follows following design:

* Preprocessing:  
	- cropping  
	- scaling  
	- normalization  

* Prediction:
	- 3 blocks of batch normalization, dropout, convolution 3x3, convolution 3x3
	- 3 dense layers

####3. Creation of the Training Set & Training Process

My total amount of data used for training is about ~18k frames for track 1 and ~7k frames from track 2. This is out of a total of about 140k captured frames. As explained earlier, frames are chose so that angles distribution is even and dropped frames are mostly 0 and -1/+1 (or extreme values) inputs frames. Adding good training data would of course help generalization, but it would also push training times longer than I am happy to wait for. 

I recorded three types of actions:  
* smooth driving  
* concentrating on turns  
* concentrating on recoveries - both from entering road boudaries and from escaping them if entered

Here are some examples of captured frames with no data augmentation applied:  
![alt raw data][raw_data]

Here are some examples of captured frames with data augmentation applied:  
![alt raw data][augmented_data]
Augmentations are rather small, thus not necessary clearly visible, perhaps apart for brightness changes.

Here are some examples of preprocessed data that gets fed to prediction model
![alt preprocessed data][preprocessed_data]

Finally here is a plot of distribution of steering angles in training data:
![alt steering angles][steering_angles_distribution].  
It has clear spikes on 0 and extreme steerings, but other than that has a reasonably uniform distribution.

Below is a plot of training and validation loss across epochs:
![alt loss plot][loss_plot].  

