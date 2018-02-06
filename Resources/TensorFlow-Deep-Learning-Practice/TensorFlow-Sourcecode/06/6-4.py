import cv2
import numpy as np
img = cv2.imread("lena.jpg")
M_copy_img = np.array([
    [0, 0.8, -100],
    [0.8, 0, -12]
], dtype=np.float32)
img_change = cv2.warpAffine(img, M_copy_img,(300,300))
cv2.imshow("test",img_change)
cv2.waitKey(0)
