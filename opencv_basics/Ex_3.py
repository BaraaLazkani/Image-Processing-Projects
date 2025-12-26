import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('rose.jpg',0)
hist = cv2.calcHist([img],[0],None,[256],[0,255])

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(img,cmap='gray')
axs[0,0].set_title('Original Image')

axs[0,1].plot(hist)
axs[0,1].set_title('Histogram of Original Image')
plt.show()
