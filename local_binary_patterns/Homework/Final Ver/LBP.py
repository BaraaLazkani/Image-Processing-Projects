import cv2
import numpy as np
import matplotlib.pyplot as plt

def binary_to_decimal(binary):
    decimal = 0
    n = len(binary)
    for i in range(n-1,-1,-1):
        decimal = decimal + binary[n-1-i]*(2**i)
    return decimal

def LBP_feat(img):
    rows = img.shape[0]
    cols = img.shape[1]
    result = np.zeros([rows,cols],int)
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            Vec = np.zeros((8))
            Vec[0] = img[i-1,j-1] > img [i,j]
            Vec[1] = img[i-1,j]   > img [i,j]
            Vec[2] = img[i-1,j+1] > img [i,j]
            Vec[3] = img[i,j+1]   > img [i,j]
            Vec[4] = img[i+1,j+1]   > img [i,j]
            Vec[5] = img[i+1,j] > img [i,j]
            Vec[6] = img[i+1,j-1]   > img [i,j]
            Vec[7] = img[i,j-1] > img [i,j]
            result[i-1,j-1] += binary_to_decimal(Vec)
    return result

def Hamming_dis(hist1,hist2,threshold):
    result = abs(hist1 - hist2)
    result = sum(result < threshold)
    return result*100/256

def Histogram(img):
    rows = img.shape[0]
    cols = img.shape[1]
    hist = np.zeros((256))
    for i in range(rows):
        for j in range(cols):
            temp = int(img[i,j])
            hist[temp] += 1
    hist = hist/max(hist)
    return hist


im_0  = cv2.imread('I0.jpg')
im_0  = cv2.resize(im_0,(100,100))
gray_0 = cv2.cvtColor(im_0, cv2.COLOR_BGR2GRAY) 
LBP_0 = LBP_feat(gray_0)
Hist_0 = Histogram(LBP_0)

im_1  = cv2.imread('I1.jpeg')
im_1  = cv2.resize(im_1,(100,100))
gray_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY) 
LBP_1 = LBP_feat(gray_1)
Hist_1 = Histogram(LBP_1)

im_2  = cv2.imread('I2.jpg')
im_2  = cv2.resize(im_2,(100,100))
gray_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY) 
LBP_2 = LBP_feat(gray_2)
Hist_2 = Histogram(LBP_2)

im_3  = cv2.imread('I3.jpg')
im_3  = cv2.resize(im_3,(100,100))
gray_3 = cv2.cvtColor(im_3, cv2.COLOR_BGR2GRAY) 
LBP_3 = LBP_feat(gray_3)
Hist_3 = Histogram(LBP_3)

im_4  = cv2.imread('I4.jpg')
im_4  = cv2.resize(im_4,(100,100))
gray_4 = cv2.cvtColor(im_4, cv2.COLOR_BGR2GRAY) 
LBP_4 = LBP_feat(gray_4)
Hist_4 = Histogram(LBP_4)

print("Hamming distance:",25/256)
print("Picture_1:")
print(Hamming_dis(Hist_0,Hist_0,0.01),",",Hamming_dis(Hist_0,Hist_1,0.01),",",Hamming_dis(Hist_0,Hist_2,0.01),",",Hamming_dis(Hist_0,Hist_3,0.01),",",Hamming_dis(Hist_0,Hist_4,0.01))
print("Picture_2:")
print(Hamming_dis(Hist_1,Hist_0,0.01),",",Hamming_dis(Hist_1,Hist_1,0.01),",",Hamming_dis(Hist_1,Hist_2,0.01),",",Hamming_dis(Hist_1,Hist_3,0.01),",",Hamming_dis(Hist_1,Hist_4,0.01))
print("Picture_3:")
print(Hamming_dis(Hist_2,Hist_0,0.01),",",Hamming_dis(Hist_2,Hist_1,0.01),",",Hamming_dis(Hist_2,Hist_2,0.01),",",Hamming_dis(Hist_2,Hist_3,0.01),",",Hamming_dis(Hist_2,Hist_4,0.01))
print("Picture_4:")
print(Hamming_dis(Hist_3,Hist_0,0.01),",",Hamming_dis(Hist_3,Hist_1,0.01),",",Hamming_dis(Hist_3,Hist_2,0.01),",",Hamming_dis(Hist_3,Hist_3,0.01),",",Hamming_dis(Hist_3,Hist_4,0.01))
print("Picture_5:")
print(Hamming_dis(Hist_4,Hist_0,0.01),",",Hamming_dis(Hist_4,Hist_1,0.01),",",Hamming_dis(Hist_4,Hist_2,0.01),",",Hamming_dis(Hist_4,Hist_3,0.01),",",Hamming_dis(Hist_4,Hist_4,0.01))

fig, axs = plt.subplots(5)
axs[0].plot(Hist_0)
axs[1].plot(Hist_1)
axs[2].plot(Hist_2)
axs[3].plot(Hist_3)
axs[4].plot(Hist_4)
plt.show()


