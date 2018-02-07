Step1 : Change CIFAR-10 images into TFRecord 

Step2 : Feeding TFRecord into Neural Network
 
Step3 : Get image recognizing result



If you want to change the amount of labels, you should change the "onehot" in the code.


test.v2：train_v2 contains 0~1 two classes, and each has about 100 images

test.v3：train_v3 contains 0~9 ten classes, and each has 100 images

test.v4：train_v4 contains 0~1 two classes, and each has 1000 images

test.v5：try to use another NN architecture
Now want to shuffle images in advance

-->   try to use high resolution images to solve the problem of reshaping

In conclusion, the result is that the image dataset's rewolution is too low, leading to reshape error
Solution: replace with image for higher res	olution


train.tfrecords_v6：try to collect 2 classes images with size biger than 70 KB


========================================================
======               final version                 =====        
========================================================
test.v6 + train_v6：try to classify 10 classes with AlexNet
Use higher resolution images instead of all images to resolve reshape problem
