import cv2
img = cv2.imread("lena.jpg")
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
less_color_hsv = img_hsv.copy()
less_color_hsv[:, :, 0] = less_color_hsv[:, :, 0] * 0.6
turn_green_img = cv2.cvtColor(less_color_hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("test",turn_green_img)
cv2.waitKey(0)
