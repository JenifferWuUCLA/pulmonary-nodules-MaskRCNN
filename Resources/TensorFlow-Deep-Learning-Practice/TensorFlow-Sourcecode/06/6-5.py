import cv2
import random

img = cv2.imread("lena.jpg")
width,height,depth = img.shape
img_width_box = width * 0.2
img_height_box = height * 0.2
for _ in range(9):
    start_pointX = random.uniform(0, img_width_box)
    start_pointY = random.uniform(0, img_height_box)
    copyImg = img[start_pointX:200, start_pointY:200]
    cv2.imshow("test", copyImg)
    cv2.waitKey(0)
