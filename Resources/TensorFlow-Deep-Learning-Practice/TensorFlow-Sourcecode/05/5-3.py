import cv2
import numpy as np
import os

randomByteArray = bytearray(os.urandom(120000))
flatNumpyArray = np.array(randomByteArray).reshape(300,400)
cv2.imshow("cool",flatNumpyArray)
cv2.waitKey(0)
