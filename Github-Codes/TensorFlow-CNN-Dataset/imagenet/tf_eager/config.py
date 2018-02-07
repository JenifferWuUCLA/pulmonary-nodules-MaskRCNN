"""
Written by Matteo Dunnhofer - 2017

Configuration class
"""


class Configuration(object):
    """
    Class defining all the hyper parameters of:
        - net architecture
        - training
        - test
        - dataset path
        ...
    """
    DATA_PATH = 'ILSVRC2012'
    IMAGENET_MEAN = [123.68, 116.779, 103.939]
    IMG_SHAPE = [224, 224, 3]  # height, width, channels
    NUM_CLASSES = 1000

    # training hyperparameters
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    BATCH_SIZE = 128
    EPOCHS = 90
    DISPLAY_STEP = 10
    VALIDATION_STEP = 1000
    SAVE_STEP = 1000
    CKPT_PATH = 'ckpt'
    SUMMARY_PATH = 'summary'

    # net architecture hyperparamaters
    LAMBDA = 5e-4  # for weight decay
    DROPOUT = 0.5

    # test hyper parameters
    K_PATCHES = 5
    TOP_K = 5
