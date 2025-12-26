import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


def conv(img,kernel):
    r = img.shape[0]
    c = img.shape[1]
    ker_r,ker_c = np.shape(kernel)
    ans = np.zeros((r,c),dtype = np.float32)
    for i in range(r):
        for j in range(c):
            ans[i][j] = img[i][j]
    for i in range(ker_r//2,r-ker_r//2):
        for j in range(ker_c//2,c-ker_c//2):
            ans[i][j] = sum(sum((img[i-ker_r//2:i+ker_r//2+1,j-ker_c//2:j+ker_c//2+1])*kernel))
    return ans

def mat_div(A,B):
    res = np.zeros((A.shape[0],A.shape[1]))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if (B[i][j]==0):
                continue
            else:
                res[i][j] = A[i][j]/B[i][j]
    return res

def Non_Max_Sup (A,dim):
    res = np.zeros((A.shape[0],A.shape[1]),dtype = np.float32)
    for i in range(dim//2,A.shape[0]-dim//2):
        for j in range(dim//2,A.shape[1]-dim//2):
            res_max = np.max(A[i-dim//2:i+dim//2+1,j-dim//2:j+dim//2+1])
            if (A[i][j]==res_max):
                res[i][j] = A[i][j]
            else:
                res[i][j] = 0
    return res

def threshold(matrix,thresh):
    res = np.zeros((matrix.shape[0],matrix.shape[1]),dtype = np.float32)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix[i][j]>=thresh):
                res[i][j]=1
            else:
                res[i][j]=0
    return res



def merge(list1,list2):
    merged_list = [(list1[i], list2[i]) for i in range(len(list1))]
    return merged_list

gaussian_filter1 =   [[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                      [0.0133, 0.0596, 0.0983, 0.0596, 0.0133], 
                      [0.0219, 0.0983, 0.1621, 0.0983, 0.0219], 
                      [0.0133, 0.0596, 0.0983, 0.0596, 0.0133], 
                      [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]]

w = [
    [0.0000, 0.0000, 0.0001, 0.0001, 0.0003, 0.0004, 0.0004, 0.0004, 0.0003, 0.0001, 0.0001, 0.0000, 0.0000],
    [0.0000, 0.0001, 0.0002, 0.0006, 0.0011, 0.0015, 0.0018, 0.0015, 0.0011, 0.0006, 0.0002, 0.0001, 0.0000],
    [0.0001, 0.0002, 0.0007, 0.0018, 0.0033, 0.0048, 0.0054, 0.0048, 0.0033, 0.0018, 0.0007, 0.0002, 0.0001],
    [0.0001, 0.0006, 0.0018, 0.0042, 0.0079, 0.0114, 0.0129, 0.0114, 0.0079, 0.0042, 0.0018, 0.0006, 0.0001],
    [0.0003, 0.0011, 0.0033, 0.0079, 0.0147, 0.0213, 0.0242, 0.0213, 0.0147, 0.0079, 0.0033, 0.0011, 0.0003],
    [0.0004, 0.0015, 0.0048, 0.0114, 0.0213, 0.0311, 0.0352, 0.0311, 0.0213, 0.0114, 0.0048, 0.0015, 0.0004],
    [0.0004, 0.0018, 0.0054, 0.0129, 0.0242, 0.0352, 0.0399, 0.0352, 0.0242, 0.0129, 0.0054, 0.0018, 0.0004],
    [0.0004, 0.0015, 0.0048, 0.0114, 0.0213, 0.0311, 0.0352, 0.0311, 0.0213, 0.0114, 0.0048, 0.0015, 0.0004],
    [0.0003, 0.0011, 0.0033, 0.0079, 0.0147, 0.0213, 0.0242, 0.0213, 0.0147, 0.0079, 0.0033, 0.0011, 0.0003],
    [0.0001, 0.0006, 0.0018, 0.0042, 0.0079, 0.0114, 0.0129, 0.0114, 0.0079, 0.0042, 0.0018, 0.0006, 0.0001],
    [0.0001, 0.0002, 0.0007, 0.0018, 0.0033, 0.0048, 0.0054 , 0.0048, 0.0033, 0.0018, 0.0007, 0.0002, 0.0001],
    [0.0000, 0.0001, 0.0002, 0.0006, 0.0011, 0.0015, 0.0018, 0.0015, 0.0011, 0.0006, 0.0002, 0.0001, 0.0000],
    [0.0000, 0.0000, 0.0001, 0.0001, 0.0003, 0.0004, 0.0004, 0.0004, 0.0003, 0.0001, 0.0001, 0.0000, 0.0000]]


Hy = [[-1, -1, -1], 
      [0, 0, 0], 
      [1, 1, 1]]

Hx = [[-1, 0, 1], 
      [-1, 0, 1], 
      [-1, 0, 1]]

img = cv2.imread('board_2.jpg')
img = cv2.resize(img,(200,200))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

smooth = conv(gray,gaussian_filter1)
Ix = conv(smooth,Hx)
Iy = conv(smooth,Hy)

Ixx = Ix * Ix
Iyy = Iy * Iy
Ixy = Ix * Iy

M11 = conv(Ixx,w)
M12 = conv(Ixy,w)
M21 = M12
M22 = conv(Iyy,w)

det_M = M11*M22-M12*M21
trace_M = M11+M22

res_1 = mat_div(det_M,trace_M)

res_2 = Non_Max_Sup(res_1,15)

res_2 /= np.max(res_2)
res_2 = threshold(res_2,0.25)

# Radius of circle 
radius = 5
# Blue color in BGR 
color = (0,0,255) 
# Line thickness of 2 px 
thickness = 2
rows,cols = np.where(res_2 == 1)
points = merge(cols,rows)
for point in points:
    image = cv2.circle(img,point,radius,color,thickness)
cv2.imshow('res', image)  
cv2.waitKey(0)
cv2.destroyAllWindows()

##cv2.imshow('1',im_g) 
##cv2.imshow('2',im_1)
##cv2.imshow('3',im_2) 
