import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

a = cv.imread('picka.png',1)
gray = cv.cvtColor(a,cv.COLOR_BGR2GRAY)
cv.imshow('imgae',gray)
cv.waitKey()


_,binary = cv.threshold(gray,127,255,cv.THRESH_BINARY)
cv.imshow('Binary Imge',binary)
cv.waitKey()


double_gray = gray.astype(np.float64)
double_gray = double_gray/255
double_gray = double_gray**0.5
print(np.max(double_gray))
print(np.min(double_gray))

fig, axs = plt.subplots(2,1)
axs[0].imshow(gray,cmap ='gray')
axs[1].imshow(double_gray,cmap ='gray')
plt.show()

