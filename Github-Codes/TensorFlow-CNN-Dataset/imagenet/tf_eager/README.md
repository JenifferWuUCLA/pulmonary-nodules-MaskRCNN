## TensorFlow Eager code

This folder contains the script to train and test Alexnet on ImageNet. This code was developed using the new TensorFlow imperative style, TensorFlow Eager. So instructions are now executed in the order they appear. No more pre-defining a computational graph and then executing it in a ```tf.Session```.


### Training
To train AlexNet just run the command:
```shell
python train.py [option]
``` 
with option ```--resume``` to resume the training from a previous checkpoint. If no option is specified the training begins from scratch.


### Testing
To evaluate the accuracy of the trained model on the ILSVRC validation set (no test set is available). Run simply:
```shell
python test.py --test
```
This evaluates *Top-1* and *Top-k* (you can set *k* inside the ```config.py``` script) accuracy and error-rate.
Inside that script you can also play with the ```K_CROPS``` parameter to see how the accuracy change when the predictions are averaged through different random crops of the images.


### Classify an image
To predict the classes of an input image run:
```shell
python test.py --classify image
```
where ```image``` is the path of the image you want to classify.
Again, you can change the number of random crops produced and the *Top-k* prediction retrieved (here are both `5`).



### Notes
```train.py``` and ```test.py``` scripts assume that ImageNet dataset folder is structured in this way:
```
ILSVRC2012
    ILSVRC2012_img_train
        n01440764
        n01443537
        n01484850
        ...
    ILSVRC2012_img_val
        ILSVRC2012_val_00000001.JPEG
        ILSVRC2012_val_00000002.JPEG
        ...
    data
        meta.mat
        ILSVRC2012_validation_ground_truth.txt
```



### Important! Help wanted here!
I did not have the chance to train this implementation, so I don't know how the training time nor the testing result on ILSVRC validation set could be. The net's architecture and the training and testing pipeline are pretty much identical to the ones contained in ```tf``` folder of this repository, so I expect outcomes not too different from that ones.
