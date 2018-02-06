import cv2

img = cv2.imread("lena.jpg")
rows,cols,depth = img.shape
img_change = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)
res = cv2.warpAffine(img,img_change,(rows,cols))
cv2.imshow("test",res)
cv2.waitKey(0)
