import numpy as np
import cv2
from scipy import ndimage

kernel33 = np.array([[-1,-1,-1],
                  [-1,8,-1],
                  [-1,-1,-1]])

kernel33_D = np.array([[1,1,1],
                  [1,-8,1],
                  [1,1,1]])

img = cv2.imread("lena.jpg",0)
linghtImg = ndimage.convolve(img,kernel33_D)
cv2.imshow("img",linghtImg)
cv2.waitKey()
