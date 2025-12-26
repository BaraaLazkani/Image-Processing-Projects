#Ali Hasan
import cv2
import numpy as np
import matplotlib.pyplot as plt

Hx = [[-1, 0, 1], 
      [-2, 0, 2], 
      [-1, 0, 1]]

Hy = [[-1,-2,-1], 
      [ 0, 0, 0], 
      [ 1, 2, 1]]

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
    res[np.where(matrix>=thresh)] = 1
    return res

def Quantize(angles):
    x = angles.shape[0]
    y = angles.shape[1]
    for i in range(x):
        for j in range(y):
            if (angles[i][j]<0 and angles[i][j]>=-180):
                angles[i][j]+=180
            if ((angles[i][j]>=0)and(angles[i][j]<22.5)):
                angles[i][j] = 0
            elif ((angles[i][j]>=22.5)and(angles[i][j]<67.5)):
                angles[i][j] = 45
            elif ((angles[i][j]>=67.5)and(angles[i][j]<112.5)):
                angles[i][j] = 90
            elif ((angles[i][j]>=112.5)and(angles[i][j]<157.5)):
                angles[i][j] = 135
            elif ((angles[i][j]>=157.5)and(angles[i][j]<202.5)):
                angles[i][j] = 180
            elif ((angles[i][j]>=202.5)and(angles[i][j]<247.5)):
                angles[i][j] = 225
            elif ((angles[i][j]>=247.5)and(angles[i][j]<292.5)):
                angles[i][j] = 270
            elif ((angles[i][j]>=292.5)and(angles[i][j]<337.5)):
                angles[i][j] = 315
            else:
                angles[i][j] = 0
    return angles

def Histogram(ker,flag):
    rows = ker.shape[0]
    cols = ker.shape[1]
    hist = np.zeros((8))
    for i in range(rows):
        for j in range(cols):
            temp = int(ker[i,j]/45)
            hist[temp] += 1
    if (flag == 1):
        hist = hist/max(hist)# Normalized Histogram " if flag == 0 => Histogram without Normalizing "
    return hist

def image_rotate(img,angle):
     res = np.zeros((17,17))
     for i in range(img.shape[0]):
         for j in range(img.shape[1]):
             x_old = i-16//2
             y_old = j-16//2
             x_new = round(np.cos(np.radians(angle))*x_old-np.sin(np.radians(angle))*y_old)
             y_new = round(np.sin(np.radians(angle))*x_old+np.cos(np.radians(angle))*y_old)
             if (x_new>=-8 and x_new<=8 and y_new>=-8 and y_new<=8):
                 res[x_new+8][y_new+8] = img[x_old][y_old]
     Ix = conv(res[0:17,0:17],Hx)
     Iy = conv(res[0:17,0:17],Hy)
     Dir = np.arctan2(Iy,Ix) * 180 / np.pi
     Quan_Dir = Quantize(Dir)
     return Quan_Dir

def hist_extra(img):
    res = np.zeros(16*8)
    for i in range(4):
        for j in range(4):
            h1 = Histogram(img[i*4:i*4+4,j*4:j*4+4],0)
            res[i*32+j*8:i*32+j*8+8] = h1
    return res

def descriptor_finder(key_points,directions,img):
    res = []
    for i in range(17//2,key_points.shape[0]-17//2):
        for j in range(17//2,key_points.shape[1]-17//2):
            if(key_points[i][j]==1):
                w_1 = directions[i-17//2:i+17//2+1,j-17//2:j+17//2+1]
                dominant_orientation = np.min(45*np.where(Histogram(w_1,1)==1))
                sub_img = img[i-17//2:i+17//2+1,j-17//2:j+17//2+1]
                rotate_img = image_rotate(sub_img,-dominant_orientation)
                res.append([i,j,hist_extra(rotate_img)])
    return res            

def euclidian_distance(list_1,list_2,thresh):
    res = []
    for i in range(len(list_1)):
        for j in range(len(list_2)):
            error=0
            for x in range(128):
                error += (list_1[i][2][x]-list_2[j][2][x])**2
            error = np.sqrt(error)
            if (error<=thresh):
                res.append([[list_1[i][0],list_1[i][1]],[list_2[j][0],list_2[j][1]]])
    return res

def show_points(point_list,num):
    merged_list = [(int(point_list[i][num][1]),int(point_list[i][num][0])) for i in range(len(point_list))]
    return merged_list


def match_img(img_1,img_2,points_1,points_2):
    image = np.concatenate((img_1,img_2),1)
    radius = 5
    color = (0,0,255) 
    thickness = 2
    for i in range(len(points_1)):
        image = cv2.circle(image,(points_1[i][0],points_1[i][1]),radius,color,thickness)
    for i in range(len(points_2)):
        image = cv2.circle(image,(points_2[i][0]+400,points_2[i][1]),radius,color,thickness)
    for i in range(len(points_1)):
        image = cv2.line(image,(points_1[i][0],points_1[i][1]),(points_2[i][0]+400,points_2[i][1]),color,thickness)
    return image

def features_extraction(img,sigma,kernel_size,thresh_1,thresh_2,thresh_3,thresh_4):
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

    thresh_part_1 = threshold(res_1,thresh_1)
    thresh_part_2 = threshold(res_2,thresh_2)
    thresh_part_3 = threshold(res_3,thresh_3)
    thresh_part_4 = threshold(res_4,thresh_4)

    res = [thresh_part_1,thresh_part_2,thresh_part_3,thresh_part_4]
    return res

def HOG(img,key_points):
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

    Ix_1 = conv(IG_1,Hx)
    Iy_1 = conv(IG_1,Hy)
    Ix_2 = conv(IG_2,Hx)
    Iy_2 = conv(IG_2,Hy)
    Ix_3 = conv(IG_3,Hx)
    Iy_3 = conv(IG_3,Hy)
    Ix_4 = conv(IG_4,Hx)
    Iy_4 = conv(IG_4,Hy)

    Dir_1 = np.arctan2(Iy_1,Ix_1) * 180 / np.pi
    Dir_2 = np.arctan2(Iy_2,Ix_2) * 180 / np.pi
    Dir_3 = np.arctan2(Iy_3,Ix_3) * 180 / np.pi
    Dir_4 = np.arctan2(Iy_4,Ix_4) * 180 / np.pi

    Quan_Dir_1 = Quantize(Dir_1)
    Quan_Dir_2 = Quantize(Dir_2)
    Quan_Dir_3 = Quantize(Dir_3)
    Quan_Dir_4 = Quantize(Dir_4)

    X_1 = descriptor_finder(key_points[0],Quan_Dir_1,IG_1)
    X_2 = descriptor_finder(key_points[1],Quan_Dir_2,IG_2)
    for i in range(len(X_2)):
        X_2[i][0] *= (400/200)
        X_2[i][1] *= (400/200)
    X_3 = descriptor_finder(key_points[2],Quan_Dir_3,IG_3)
    for i in range(len(X_3)):
        X_3[i][0] *= (400/100)
        X_3[i][1] *= (400/100)
    X_4 = descriptor_finder(key_points[3],Quan_Dir_4,IG_4)
    for i in range(len(X_4)):
        X_4[i][0] *= (400/50)
        X_4[i][1] *= (400/50)

    res = X_1+X_2+X_3+X_4
    return res



img = cv2.imread('I8.jpg')
img_1 = cv2.imread('I9.jpg')

I_1 = cv2.resize(img,(400,400))
IG_1 = cv2.cvtColor(I_1,cv2.COLOR_BGR2GRAY)
I_2 = cv2.resize(img_1,(400,400))
IG_2 = cv2.cvtColor(I_2,cv2.COLOR_BGR2GRAY)

sigma = [2,np.sqrt(2)*2,4,np.sqrt(2)*4]
kernel_size = [int(6*sigma[0]),int(6*sigma[1]),int(6*sigma[2]),int(6*sigma[3])]

for i in range(4):
    if(kernel_size[i]%2==0):
        kernel_size[i] += 1

key_points_1 = features_extraction(img,sigma,kernel_size,0.7,0.9,0.5,0.5)
key_points_2 = features_extraction(img_1,sigma,kernel_size,0.5,0.5,0.5,0.5)
descriptor_vector_1 = HOG(img,key_points_1)
descriptor_vector_2 = HOG(img_1,key_points_2)

macth_point = euclidian_distance(descriptor_vector_1,descriptor_vector_2,21)

P_1 = show_points(macth_point,0)
P_2 = show_points(macth_point,1)

final_img = match_img(I_1,I_2,P_1,P_2)

cv2.imshow('res', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
