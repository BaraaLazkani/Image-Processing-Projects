import cv2
import functions


img = cv2.imread("I0.jpg",0)
img_resize = cv2.resize(img,(100,100))
cv2.imshow("image", img_resize)

img_1 = functions.LBP_feature_extractor(img)
