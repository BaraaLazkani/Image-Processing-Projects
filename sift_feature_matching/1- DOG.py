#Ali Hasan
import cv2
import numpy as np
import matplotlib.pyplot as plt

Hx = [[-1, 0, 1], 
      [-2, 0, 2], 
      [-1, 0, 1]]

Hy = [[-1,-2,-1], 
      [ 0, 0, 0], 
      [ 1, 2 ,1]]

def conv(img,kernel):
    ans = cv2.filter2D(img,-1,np.array(kernel))
    return ans

def octave(img,sigma,kernel_size):
    res = []
    res.append(cv2.GaussianBlur(img,(kernel_size[0],kernel_size[0]),sigma[0]))
    res.append(cv2.GaussianBlur(img,(kernel_size[1],kernel_size[1]),sigma[1]))
    res.append(cv2.GaussianBlur(img,(kernel_size[2],kernel_size[2]),sigma[2]))
    res.append(cv2.GaussianBlur(img,(kernel_size[3],kernel_size[3]),sigma[3]))
    return res

def DOG(octave):
    temp = []
    for i in range(3):
        temp.append(octave[i+1]-octave[i])
    res = np.array(temp)
    return res

def extreme_finder(DOG,dim):
    r,c = np.shape(DOG[0])
    final_matrix = np.zeros((r,c))
    for i in range(dim//2,r-dim//2):
        for j in range(dim//2,c-dim//2):
            if (DOG[1,i,j] == np.max(DOG[:,i-dim//2:i+dim//2+1,j-dim//2:j+dim//2+1])):
                final_matrix[i][j] = 1
            if (DOG[1,i,j] == np.min(DOG[:,i-dim//2:i+dim//2+1,j-dim//2:j+dim//2+1])):
                final_matrix[i][j] = 1
    return final_matrix

def threshold(matrix,thresh):
    matrix /= np.max(matrix)
    res = np.zeros((matrix.shape[0],matrix.shape[1]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if(matrix[i][j]>=thresh):
                res[i][j]=1
            else:
                res[i][j]=0
    return res

def merge(list1,list2):
    merged_list = [(int(list1[i]), int(list2[i])) for i in range(len(list1))]
    return merged_list

img = cv2.imread('I8.jpg')
I_1 = cv2.resize(img,(400,400))
I_2 = cv2.resize(img,(200,200))
I_3 = cv2.resize(img,(100,100))
I_4 = cv2.resize(img,(50,50))

IG_1 = cv2.cvtColor(I_1,cv2.COLOR_BGR2GRAY)
IG_2 = cv2.cvtColor(I_2,cv2.COLOR_BGR2GRAY)
IG_3 = cv2.cvtColor(I_3,cv2.COLOR_BGR2GRAY)
IG_4 = cv2.cvtColor(I_4,cv2.COLOR_BGR2GRAY)

IG_1 =  IG_1.astype(np.float32)
IG_2 =  IG_2.astype(np.float32)
IG_3 =  IG_3.astype(np.float32)
IG_4 =  IG_4.astype(np.float32)

sigma = [2,np.sqrt(2)*2,4,np.sqrt(2)*4]
kernel_size = [int(6*sigma[0]),int(6*sigma[1]),int(6*sigma[2]),int(6*sigma[3])]

for i in range(4):
    if(kernel_size[i]%2==0):
        kernel_size[i] += 1

octave_1 = octave(IG_1,sigma,kernel_size)
octave_2 = octave(IG_2,sigma,kernel_size)
octave_3 = octave(IG_3,sigma,kernel_size)
octave_4 = octave(IG_4,sigma,kernel_size)

DOG_1 = DOG(octave_1)
DOG_2 = DOG(octave_2)
DOG_3 = DOG(octave_3)
DOG_4 = DOG(octave_4)

loc_extrems_1 = extreme_finder(DOG_1,3)
loc_extrems_2 = extreme_finder(DOG_2,3)
loc_extrems_3 = extreme_finder(DOG_3,3)
loc_extrems_4 = extreme_finder(DOG_4,3)

res_1 = np.abs(DOG_1[1]*loc_extrems_1)
res_2 = np.abs(DOG_2[1]*loc_extrems_2)
res_3 = np.abs(DOG_3[1]*loc_extrems_3)
res_4 = np.abs(DOG_4[1]*loc_extrems_4)

thresh_part_1 = threshold(res_1,0.7)
thresh_part_2 = threshold(res_2,0.9)
thresh_part_3 = threshold(res_3,0.5)
thresh_part_4 = threshold(res_4,0.5)


rows_1,cols_1 = np.where(thresh_part_1 == 1)
points_1 = merge(cols_1,rows_1)

rows_2,cols_2 = np.where(thresh_part_2 == 1)
rows_2_mod = rows_2 * (400/200)
cols_2_mod = cols_2 * (400/200)
points_2 = merge(cols_2_mod,rows_2_mod)

rows_3,cols_3 = np.where(thresh_part_3 == 1)
rows_3_mod = rows_3 * (400/100)
cols_3_mod = cols_3 * (400/100)
points_3 = merge(cols_3_mod,rows_3_mod)

rows_4,cols_4 = np.where(thresh_part_4 == 1)
rows_4_mod = rows_4 * (400/50)
cols_4_mod = cols_4 * (400/50)
points_4 = merge(cols_4_mod,rows_4_mod)

radius_1 = 5
color_1 = (255,0,255) 
thickness_1 = 2
for point in points_1:
    image = cv2.circle(I_1,point,radius_1,color_1,thickness_1)

radius_2 = 5
color_2 = (0,0,255) 
thickness_2 = 2
for point in points_2:
    image = cv2.circle(image,point,radius_2,color_2,thickness_2)
 
radius_3 = 5
color_3 = (0,255,0) 
thickness_3 = 2
for point in points_3:
    image = cv2.circle(image,point,radius_3,color_3,thickness_3)

radius_4 = 5
color_4 = (255,0,0) 
thickness_4 = 2
for point in points_4:
    image = cv2.circle(image,point,radius_4,color_4,thickness_4)

cv2.imshow('res', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
