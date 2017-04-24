# Traffic Sign Recognition

Here is a link to my [project code](https://github.com/kei-sato/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[//]: # (Image References)

[image1]: ./images/visualization1.png "Visualization 1"
[image2]: ./images/visualization2.png "Visualization 2"
[image3]: ./images/grayscale1.png "Grayscaling"
[image4]: ./images/normalization1.png "Normalization"
[image5]: ./images/augmentation1.png "Augmentation"
[image6]: ./traffic-signs-from-web/1.png "Traffic Sign 1"
[image7]: ./traffic-signs-from-web/2.png "Traffic Sign 2"
[image8]: ./traffic-signs-from-web/3.png "Traffic Sign 3"
[image9]: ./traffic-signs-from-web/4.png "Traffic Sign 4"
[image10]: ./traffic-signs-from-web/5.png "Traffic Sign 5"
[image11]: ./traffic-signs-from-web/6.png "Traffic Sign 6"
[image12]: ./traffic-signs-from-web/7.png "Traffic Sign 7"
[image13]: ./traffic-signs-from-web/8.png "Traffic Sign 8"
[image14]: ./images/prediction1.png "Prediction 1"
[image15]: ./images/prediction2.png "Prediction 2"
[image16]: ./images/prediction3.png "Prediction 3"

###Data Set Summary & Exploration

I used the python and numpy to calculate summary statistics of the traffic signs data set as described in my project code:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set.

These are 16 images randomly selected from the training set.

![alt text][image1]

It is a bar chart showing how the data are distributed by classes. X axis shows classes, and Y axis shows an amout of the data belonging to the corresponding class.

![alt text][image2]

###Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale bacause it reduces the size of the data on memory, and performs well as described in [the report](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by the competitor of the competition using this data set.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image3]

As a second step, I normalized the image data to emphasize the contrast of the images.

Here is an example of a traffic sign image before and after normalization.

![alt text][image4]

As a last step, I generated additional data to add more robustness to the training.

I created additional data randomly scaled (+/- 0.1), rotated (+/- 15 degrees), and transformed (+/- 2 pixels) to the training set.

Here is an example of an original image and an augmented image:

![alt text][image5]

My final model consisted of the following layers:

| Layer							| Description														| 
|:-----------------------------:|:-----------------------------------------------------------------:| 
| 1. Input						| 32x32x1 Grayscale image											| 
| 2. Convolution 5x5			| 1x1 stride, outputs 28x28x6										|
| 3. ReLU						|																	|
| 4. Max pooling				| 2x2 stride, outputs 14x14x6										|
| 5. Convolution 5x5			| 1x1 stride, outputs 10x10x6										|
| 6. ReLU						|																	|
| 7. Max pooling				| 2x2 stride, outputs 5x5x6											|
| 8. Max pooling				| pooling outputs from layer 4, 2x2 stride, outputs 7x7x6			|
| 9. Flatten and Concatenate	| flatten and concatinate outputs from layer 7 and 8, outputs 694	|
| 10. Dropout					| ratio 0.5															|
| 11. Fully connected			| 694 to 120														|
| 12. ReLU						|																	|
| 13. Fully connected			| 120 to 43															|
| 14. Softmax					|																	|


To train the model, I used an optimizer, and hyperparamters as follows:

- optimizer: Adam
- batch size: 128
- epochs: 15
- learning rate: 0.001
- mu: 0
- sigma: 0.1

My final model results were:

* validation set accuracy of 0.954
* test set accuracy of 0.93

I chosen LeNet for the initial architecture because that was described in the video and well performed. But it was not as good as described in the video. When I tried on [the data set](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip) given by the introduction, the accuracy was about 0.88.

So I read [the report by Sermanet and LeCun](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), and decided to take same aproach as much as possible.

First, I converted images to grayscale and normalize them, and achieved the accuracy of 0.91.

Next, I augmented images with random scale, rotation, and transformation. Then the accuracy slightly increased to 0.919.

Next, I inserted a dropout layer between Convolution layer and Fully connected layer so that the model can be more redundant. Then the accuracy increased to 0.928.

And finally, I adopted the idea described in the report so that the Fully connected layer can take the output from the first Convolution layer. Then the accuracy reached to 0.954.

###Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12] ![alt text][image13]

These signs might be misclassified because:

- 1st one contains bold virtical yellow line which is same color of the content of the sign
- 2nd one is dark and contains weird thing on the background of the sign
- 3rd one is tiny and difficult to identify even for human
- 4th, 5th ones contain unnecessary white board
- 6th one is small and slightly deformed
- 7th, 8th ones are dark

And here are the results of the prediction:

![alt text][image14]

There was one mistake on the third one from the left. It's predicted as "*Beware of ice/snow*", but the correct answer is "*Right-of-way at the next intersection*".

The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This is less than the accuracy on the test set, but it could be ignored because it's just one mistake.

The top 5 softmax probabilities for each image are the followings:

![alt text][image15]
![alt text][image16]

At most images, the certainty of predictions are quite high. But for the third one, we can see the less confidence of 51% to "*Beware of ice/snow*" and 45% to "*Right-of-way at the next intersection*". I found similar images on the training data of both of them, so it makes some sense.
