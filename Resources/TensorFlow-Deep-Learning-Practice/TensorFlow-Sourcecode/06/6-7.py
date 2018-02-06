import cv2
import  numpy as np
img = cv2.imread("lena.jpg")
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
turn_green_hsv = img_hsv.copy()
turn_green_hsv[:,:,0] = (turn_green_hsv[:,:,0] + np.random.random() ) % 180
turn_green_hsv[:,:,1] = (turn_green_hsv[:,:,1] + np.random.random() ) % 180
turn_green_hsv[:,:,2] = (turn_green_hsv[:,:,2] + np.random.random() ) % 180
turn_green_img = cv2.cvtColor(turn_green_hsv,cv2.COLOR_HSV2BGR)
cv2.imshow("test",turn_green_img)
cv2.waitKey(0)
