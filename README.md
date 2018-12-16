# Image detection 
This programme uses Python OpenCV library to detect and image given a dataset of images. 
To do this: 
1. we divided our dataset in to two ( training and test)
2. we created a model made of SIFT descriptors and KeyPoint of images in the training set. 

To dectect an image, we :
1. extract the image SIFT decriptor 
2. we use KnnMatcher to look for matching descriptors
3. we then count the number of descriptors of a certain distance ( ratio of 0,75 for example)
4. we normalize the count by dividing them with the total number of image of the selected subset 