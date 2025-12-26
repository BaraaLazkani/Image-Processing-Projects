import cv2
import numpy as np
import matplotlib.pyplot as plt


def bin2dec(binary):
    decimal = 0
    n = len(binary)
    for i in range(n-1,-1,-1):
        decimal += binary[n-1-i]*(2**i)
    return decimal


def LBP(img):
    r,c = np.shape(img)
    res = np.zeros((r,c))
    for i in range(r):
        for j in range(c):
            if(i == 0 or i == r-1): res[i,j] = 0; continue;
            if(j == 0 or j == c-1): res[i,j] = 0; continue;
            Vec = np.zeros((8))
            Vec[0] = img[i-1,j-1] > img [i,j]
            Vec[1] = img[i-1,j]   > img [i,j]
            Vec[2] = img[i-1,j+1] > img [i,j]
            Vec[3] = img[i,j+1]   > img [i,j]
            Vec[4] = img[i+1,j+1]   > img [i,j]
            Vec[5] = img[i+1,j] > img [i,j]
            Vec[6] = img[i+1,j-1]   > img [i,j]
            Vec[7] = img[i,j-1] > img [i,j]
            res[i,j] += bin2dec(Vec)
    #res[i,j]/=255;
    return res

def Hist(img):
    r,c = np.shape(img)
    hist = np.zeros((256))
    for i in range(r):
        for j in range(c):
            hist[int(img[i,j])] += 1
    hist = hist/max(hist)
    return hist

def Hamming(hist1,hist2,threshold):
    res = abs(hist1 - hist2)
    res = sum(res < threshold)
    return res*100/256

img1  = cv2.imread('I0.jpg')
img1  = cv2.resize(img1,(100,100))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
res1 = LBP(gray1)
hist1 = Hist(res1)

img2  = cv2.imread('I1.jpeg')
img2  = cv2.resize(img2,(100,100))
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
res2 = LBP(gray2)
hist2 = Hist(res2)

print(Hamming(hist1,hist2,0.01))

fig, axs = plt.subplots(2)
axs[0].plot(hist1)
axs[1].plot(hist2)
plt.show()

