import cv2
import numpy as np

def binary_to_Decimal(binary_vector):
    decimal = 0
    power = len(binary_vector) - 1
    for bit in binary_vector:
        decimal += bit * (2 ** power)
        power -= 1
    return decimal

def LBP_binary_calc(img,row,col):
    res = np.zeros(8)
    indx = 0
    for i in range(-1,1):
        for j in range(-1,1):
            if i!=0 and j!=0:
                if img[row+i,col+j]>img[row,col]:
                    res[indx] = 1
                    indx = indx + 1
                if img[row+i,col+j]<=img[row,col]:
                    res[indx] = 0
                    indx = indx + 1
    return res
                
def LBP_feature_extractor(img):
    height,width = img.shape
    for row in range(height):
        for col in range(width):
            if row==0 or col==0:
                img[row,col] = 0
                continue
            LBP_binary = LBP_binary_calc(img,row,col)
            img[row,col] = binary_to_Decimal(LBP_binary)
    return img

def calcHistogram(img):
    height,width = img.shape
    histogram = np.zeros([256], np.int32) 
    for i in range(height):
        for j in range(width):
            histogram[img[i, j]] +=1
    for k in range(255):
        histogram[k] = histogram[k]/max(histogram)
    return histogram
            
