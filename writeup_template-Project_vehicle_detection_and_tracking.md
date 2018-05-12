## Writeup



**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/features.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/bad_detection.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th and 5th code cells of the jupyter notebook.

I started by reading in all the `vehicle` and `non-vehicle` images. The images are stored in several sub folders, and I used `glob` function to read all image with `**` for multiple folders. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. 

Here is an example using the `YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

Besides extracting HOG feature from all training images, I also used spatial features and color histogram features from each training images. But since HOG feature is the most important feature for detection, I set the spatial feature and color histogram feature to be a smaller size features, using `size=(16, 16)` for spatial features and `nbins=16` for color histogram feature. The feature extraction function is defined in the 3rd cell of the jupyter notebook.

#### 2. Explain how you settled on your final choice of HOG parameters.

In some experiments, I tried different combination of parameter orientations (8, 11, 12), pixels_per_cell (8, 16) and cells_per_block (2). By considering factors below:

* Calculation time
* Accuracy
* Implementation of window search

I decided to use `orientations=11`, `pixels_per_cell=8`, and `cells_per_block=2` as the final choice of HOG parameters.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After I extracted features from all training images, I created label arrays for Car (as 1) and NonCar (as 0). Then I used `train_test_split` to split data for training and testing. Before fitting the data for training, in order to standardize the feature data, I used `StandardScaler()` to fit the training data and get a `Scaler`, and used the `Scaler` to trasform both training and tesing data.

I then trained a linear SVM using training data and label data, and use testing data to calculate the accuracy. I got testing accuracy of 98.8% from the trained SVM.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

At first, I used `slide_window` function with a scale list as input to create a list of windows to search from, extracted features from each window and used trained SVM to detect cars. This method worked fine for single image, but it became very unefficient for video processing, because of repeating the HOG feature extraction. 

After that, I changed to use the 'find_cars' function to search cars, which is defined in the 8th cell of the jupyter notebook. This function extracts the HOG features of the whole seaching area once, and uses the info in each search window, which makes it work much faster than the `slide_window` method.

Regarding scales of the searching windows, based on the fact that farther away cars should have smaller images and should appear in the upper part of the searching area, I decided to use parameters below to scale the search windows and define the searching area of each scale, and always using 75% overlap in searching.

`scale = [1.0, 1.5, 2.0, 3.5]`: scales used to define the search windows

`ystart = 400`: starting row of searching area of every scales

`ystop = [480, 528, 560, 660]` stopping rows of searching area of each scale

Regarding the windows scale, at first I was trying to scale the search window itself (use 64X64 as baseline). After I figured out I still need to resize it to be 64X64 no matter what scale is in use, I changed to scale the whole searching area of the image, like the way suggested in the lesson.

All search windows in the whole searching area are as showed in the image below. 

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YUV 3-channel HOG features plus spatially binned color (16*16) and histograms of color (nbin = 16) in the feature vector, which provided a good result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's the [link to my video result](./project_video_out.mp4).

I also combined the lane detection part from last project and here is the [link to my video result with lane detection](./project_video_out_combined1.mp4).


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

This part of code is in the 2nd cell of the jupyter notebook.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One interesting thing I found is that, I got pretty satisfactory results working on all test images and the video, but it performs not as good in image "bbox-example-image.jpg", as showed in the image below. I am thinking the reasom might be there are more cars in the image, and the sizes are in more difference, also maybe the tree color is affecting the performance. I am looking for some suggestions to improve performance on such images.

![alt text][image7]

Another interesting thing is the performance of neural network solution. I tried to carry over the LeNet model used in "traffic sign detection project" here and modified the input and output of each layer, used the raw training images (but not their certain feature) to train the network. After 20 epoches training I can get accuracy between 98-99%. However, the performance on the same test images are not that good. I am still trying to figure out the cause, and some help/advice is appreciated.

