import cv2
import numpy as np
img = np.zeros((300,300))
img[0,0] = 255
cv2.imshow("img",img)
cv2.waitKey(0)
