# Build a Traffic Sign Recognition Project - Luca Fiaschi

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plots/training_set_counts.png "Visualization"
[image2]: ./plots/random_examples.png "Random Examples"
[image3]: ./plots/image_transformations.png "Transformations"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points

Here I will consider the [rubric points](https://github.com/lfiaschi/udacity-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)
individually and describe how I addressed each point in my implementation. The implementation and the project writeup 
can be found here to my [project code](https://github.com/lfiaschi/udacity-traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

---

## Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is 
ditributed across the different labels.

![alt text][image1]

The average number of training examples per class is 809, the minimum is 180 and the maximum 2010, 
hence some labels are one order of magnitude more abundant than others.

Most common signs:
* `Speed limit (50km/h)`  train samples: 2010
* `Speed limit (30km/h)`  train samples: 1980
* `Yield`  train samples: 1920
* `Priority road`  train samples: 1890
* `Keep right`  train samples: 1860


Most rare signs:

* `Speed limit (20km/h)`  train samples: 180
* `Dangerous curve to the left`  train samples: 180
* `Go straight or left`  train samples: 180
* `Pedestrians`  train samples: 210
* `End of all speed and passing limits`  train samples: 210

Here is an visualization of some 10 randomly picked training examples for each class. 
As we can see, within each class there is a high variability in appearance due to different 
weather conditions, time of the day and image angle.

![alt text][image2]

---

## Design and Test a Model Architecture

#### Question 1:
*Describe how you preprocessed the data. Why did you choose that technique?*
##### Answer
Following a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) 
I applied similar normalization and image enhancements . Images were transformed in the YUV space and adjusted by 
histogram sketching and by increasing sharpness.  Finally only the Y channel was selected as in some preliminary 
experiments full color images seem to confuse the classifier (as also reported in the published baseline), 
the latter effect however may depend on the network architecture, as in the long term we would intuitively expect 
to have networks trained with full color images to perform better.

Here is an example of an original image and the transformed image.

![alt text][image3]

The difference between the original data set and the augmented data set is the following is the reduced level of  noise and number of channels.


#### Question 2:
*Describe how you set up the training, validation and testing data for your model. Optional: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?*
##### Answer:
All images were processed by transform_img function as discribed in the question 1. Training test and validation set were provided in the exercise. Training set was also augmented by generating 5 additional images from every given image. Images were augmented by augment_img function. The process consists of random rotation around image center (random value between -15 and 15 deg) and random vertical stretching (as the simplest way to simulate different viewing angle) by random value up to 40 %.

#### Question 3:

*Describe what your final model architecture looks like.*

##### Answer
My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Y channe image   						|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|

#### Question 4:
*How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)*

##### Answer:

I trained the model using an Adam optimizer , a learning rate of 1e-4 , dropout rate of 0.3 and batch size of 128.


#### Question 5
*What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem.*

##### Answer:

To train the model, I started from a a well known architecture (LeNet) because of simplicity of implementation and because it performs well on recognition task with tens of classes (such as carachter recognition). After a few runs with this architecture I noted that the model tended to overfit to the original training set, in fact the learning curve showed that the training error converged to 99% while the validation error was above a truly satisfactory performance. Hence I tested two regularization techniques to improve the results:

* Data augmentation
* Dropout

I started trying with an high droput rate 50% and this seemed to slow down overfitting as the model was slower to learn but also achieved a slightly higher accuracy. When added the augmented dataset however I started seing increased performance as the model was now able to learn within a few epocs but at the same time to generalize well on the validation set.

A dropout rate of .30% was selected after a few trial and errors. Training the model overall takes around an hour

Training curves can be seen below, at the end of the curves both training and validation error converge.



My final model results were:
* training set accuracy of ?
* validation set accuracy of ?
* test set accuracy of ?

### Test a Model on New Images

#### Question 1:

*Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.*

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...