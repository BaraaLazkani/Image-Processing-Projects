import cv2 as cv
a = cv.imread('picka.png',0)
a = cv.imread('picka.png',1)
cv.imshow('P',a)
cv.waitKey()
print("Hi")
