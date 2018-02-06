import numpy as np
import cv2
image = np.mat(np.zeros((300,300)))
imageByteArray = bytearray(image)
print(imageByteArray)
imageBGR = np.array(imageByteArray).reshape(300,300)
cv2.imshow("cool",imageBGR)
cv2.waitKey(0)
