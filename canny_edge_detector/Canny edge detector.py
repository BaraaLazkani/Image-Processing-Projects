#Ali Hasan
import cv2
import numpy as np
import matplotlib.pyplot as plt


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
    #ans = ans[ker_r//2:r-ker_r//2,ker_c//2:c-ker_c//2]
    return ans

def Positive_Quantize(angles):
    x = angles.shape[0]
    y = angles.shape[1]
    for i in range(x):
        for j in range(y):
            angles[i][j] += 180
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

def Non_Max_Sup(edges,angles):
    res = np.zeros((edges.shape[0],edges.shape[1]),dtype = np.float32)
    for i in range(1,angles.shape[0]-1):
        for j in range(1,angles.shape[1]-1):
            if (angles[i][j]==0 or angles[i][j]==180):
                if (edges[i][j]<=edges[i][j-1] or edges[i][j]<=edges[i][j+1]):
                    res[i][j]=0
                else:
                    res[i][j]=edges[i][j]
            elif (angles[i][j]==45 or angles[i][j]==225):
                if (edges[i][j]<=edges[i-1][j+1] or edges[i][j]<=edges[i+1][j-1]):
                    res[i][j]=0
                else:
                    res[i][j]=edges[i][j]   
            elif (angles[i][j]==90 or angles[i][j]==270):
                if (edges[i][j]<=edges[i-1][j] or edges[i][j]<=edges[i+1][j]):
                    res[i][j]=0
                else:
                    res[i][j]=edges[i][j]
            else :
                if (edges[i][j]<=edges[i-1][j-1] or edges[i][j]<=edges[i+1][j+1]):
                    res[i][j]=0
                else:
                    res[i][j]=edges[i][j]
    return res[1:edges.shape[0]-1,1:edges.shape[1]-1]


def Edges_Classify(edges,min_thres,max_thres):
    res = np.zeros((edges.shape[0],edges.shape[1]),dtype = np.float32)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if(edges[i][j]<=max_thres and edges[i][j]>=min_thres):
                res[i][j]=1
            if(edges[i][j]>max_thres):
                res[i][j]=2
    return res


def Edges_Eliminate_Weak(edges,min_thres):
    res = np.zeros((edges.shape[0],edges.shape[1]),dtype = np.float32)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if (edges[i][j]<min_thres):
                res[i][j]=0
            else:
                res[i][j] = edges[i][j]
    return res

def Edges_Eliminate_Candidate(edges,min_thres,max_thres):
    res = np.zeros((edges.shape[0],edges.shape[1]),dtype = np.float32)
    flag = 0
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            res[i][j] = edges[i][j]
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            if (edges[i][j]<=max_thres and edges[i][j]>=min_thres):
                try:
                    for x in range(-1,2):
                        for y in range(-1,2):
                            if (edges[i+x][j+y]>max_thres):
                                flag = 1
                                break
                        if(flag == 1):
                            break
                    if(flag==0):
                        res[i][j]=0
                        flag = 0
                    else:
                        flag = 0
                except:
                    continue
    return res


gaussian_kernel = np.array([[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                            [0.0133, 0.0596, 0.0983, 0.0596, 0.0133], 
                            [0.0219, 0.0983, 0.1621, 0.0983, 0.0219], 
                            [0.0133, 0.0596, 0.0983, 0.0596, 0.0133], 
                            [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]])

Hx = [[-1, 0, 1], 
      [-2, 0, 2], 
      [-1, 0, 1]]

Hy = [[ 1, 2, 1], 
      [ 0, 0, 0], 
      [-1,-2,-1]]


img_1 = cv2.imread('city.jpg')
img_2 = cv2.imread('I2.jpg')
img_3 = cv2.imread('I3.jpg')

img_1 = cv2.resize(img_1,(200,200))
img_2 = cv2.resize(img_2,(200,200))
img_3 = cv2.resize(img_3,(200,200))

gray_1 = cv2.cvtColor(img_1,cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(img_2,cv2.COLOR_BGR2GRAY)
gray_3 = cv2.cvtColor(img_3,cv2.COLOR_BGR2GRAY)

im_1 = conv(gray_1,gaussian_kernel)
im_2 = conv(gray_2,gaussian_kernel)
im_3 = conv(gray_3,gaussian_kernel)


Ix_1 = conv(im_1,Hx)
Iy_1 = conv(im_1,Hy)
Ix_2 = conv(im_2,Hx)
Iy_2 = conv(im_2,Hy)
Ix_3 = conv(im_3,Hx)
Iy_3 = conv(im_3,Hy)

E_1 = np.sqrt(Ix_1**2+Iy_1**2)
E_1 /= np.max(E_1)
E_2 = np.sqrt(Ix_2**2+Iy_2**2)
E_2 /= np.max(E_2)
E_3 = np.sqrt(Ix_3**2+Iy_3**2)
E_3 /= np.max(E_3)

angles_1 = np.arctan2(Iy_1,Ix_1) * 180 / np.pi
Quan_angles_1 = Positive_Quantize(angles_1)
angles_2 = np.arctan2(Iy_2,Ix_2) * 180 / np.pi
Quan_angles_2 = Positive_Quantize(angles_2)
angles_3 = np.arctan2(Iy_3,Ix_3) * 180 / np.pi
Quan_angles_3 = Positive_Quantize(angles_3)

E_nms_1 = Non_Max_Sup(E_1,Quan_angles_1)
E_class_1 = Edges_Classify(E_nms_1,0.03,0.3)
E_eliminate_1 = Edges_Eliminate_Weak(E_nms_1,0.03)
final_ans_1 = Edges_Eliminate_Candidate(E_eliminate_1,0.03,0.3)

E_nms_2 = Non_Max_Sup(E_2,Quan_angles_2)
E_class_2 = Edges_Classify(E_nms_2,0.02,0.1)
E_eliminate_2 = Edges_Eliminate_Weak(E_nms_2,0.02)
final_ans_2 = Edges_Eliminate_Candidate(E_eliminate_2,0.02,0.1)

E_nms_3 = Non_Max_Sup(E_3,Quan_angles_3)
E_class_3 = Edges_Classify(E_nms_3,0.03,0.23)
E_eliminate_3 = Edges_Eliminate_Weak(E_nms_3,0.03)
final_ans_3 = Edges_Eliminate_Candidate(E_eliminate_3,0.03,0.23)


fig,axs = plt.subplots(3,2)
axs[0,0].imshow(gray_1,cmap = 'gray')
axs[0,1].imshow(final_ans_1,cmap = 'gray')
axs[1,0].imshow(gray_2,cmap = 'gray')
axs[1,1].imshow(final_ans_2,cmap = 'gray')
axs[2,0].imshow(gray_3,cmap = 'gray')
axs[2,1].imshow(final_ans_3,cmap = 'gray')
plt.show()


