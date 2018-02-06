import numpy as np
import cv2
from scipy import ndimage

img = cv2.imread("lena.jpg",0)
blurred = cv2.GaussianBlur(img,(11,11),0)
gaussImg = img - blurred
cv2.imshow("img",gaussImg)
cv2.waitKey()
