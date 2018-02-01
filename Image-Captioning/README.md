# Image Captioning

We reimplemented the complicated [Google' Image Captioning](https://github.com/tensorflow/models/tree/master/im2txt) model by simple [TensorLayer](https://github.com/zsdonghao/tensorlayer) APIs.

This script run well under Python2 or 3 and TensorFlow 10 or 11.

### 1. Prepare MSCOCO data and Inception model
Before you run the scripts, you need to follow Google's [setup guide]((https://github.com/tensorflow/models/tree/master/im2txt)), and setup the model, ckpt and data directories in *.py.

- Creat a ``data`` folder.
- Download and Preprocessing MSCOCO Data [click here](https://github.com/tensorflow/models/tree/master/research/im2txt)
- Download the Inception_V3 CKPT [click here](https://github.com/tensorflow/models/tree/master/research/slim)


### 2. Train the model
- via ``train.py``

### 3. Evaluate the model
- via ``evaluate.py``

### 4. Generate captions by given image and model
- via ``run_inference.py``

### 5. Evaluation
- [tylin/coco-caption](https://github.com/tylin/coco-caption/blob/master/cocoEvalCapDemo.ipynb)
