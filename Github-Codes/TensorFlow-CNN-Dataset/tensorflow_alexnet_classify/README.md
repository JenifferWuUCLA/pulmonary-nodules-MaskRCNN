# tensorflow_alexnet_classify
> This repository aims to implement a alexnet with tensorflow . it gives a pretrain weight (bvlc_alexnet.npy), you can download from 
[here](http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/).the train file contains 25000 images (cat and dog). 
> We built this AlexNet in Windows ,  it's very convenient for most of you to train the net.

## Requirements
* Python 3.5
* TensorFlow 1.0
* Numpy
* cat and dog images [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## Usage 
* image_generator: it can  generate imageurl  from your image file.  

    **example:**
    
    /path/to/train/image1.png 0
    
    /path/to/train/image2.png 1
    
    /path/to/train/image3.png 2
    
    /path/to/train/image4.png 0

## Notes:
* The alexnet.py and datagenerator.py files have been builded, you don't have to modify it. But if you have more concise or effective codes, please do share them with us.
* finetune.py is aimed to tune the weights and bias in the full connected layer, you must define some varibles,functions,and class numbers according to your own classification projects.  

## Example output:
We choosed ten pictures from the internet to validate the AlexNet, there were three being misidentified, the accuracy is about 70%, which is similar to the accuracy we tested before. But, On the whole, the AlexNet is not as good as we expected, the reason may have something to do with the datesets. If you have more than One hundred thousand dataset, the accuracy must be better than we trained.
See the results below:

![2017-10-18-10-16-50](http://qiniu.xdpie.com/2017-10-18-10-16-50.png)

![2017-10-18-10-18-37](http://qiniu.xdpie.com/2017-10-18-10-18-37.png)

![2017-10-18-10-19-57](http://qiniu.xdpie.com/2017-10-18-10-19-57.png)

![2017-10-18-10-21-22](http://qiniu.xdpie.com/2017-10-18-10-21-22.png)

![2017-10-18-10-23-09](http://qiniu.xdpie.com/2017-10-18-10-23-09.png)

![2017-10-18-10-27-53](http://qiniu.xdpie.com/2017-10-18-10-27-53.png)

![2017-10-18-10-26-36](http://qiniu.xdpie.com/2017-10-18-10-26-36.png)

![2017-10-18-10-29-58](http://qiniu.xdpie.com/2017-10-18-10-29-58.png)

![2017-10-18-10-33-15](http://qiniu.xdpie.com/2017-10-18-10-33-15.png)
![2017-10-18-10-38-02](http://qiniu.xdpie.com/2017-10-18-10-38-02.png)
    
 
