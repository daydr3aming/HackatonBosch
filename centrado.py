import cv2
import numpy as np

img = cv2.imread('Images/REF_23.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,5,3,0.04)
ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

img2 = cv2.imread('Images/1.png')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray2 = np.float32(gray2)
dst2 = cv2.cornerHarris(gray2,5,3,0.04)
ret2, dst2 = cv2.threshold(dst2,0.1*dst2.max(),255,0)
dst2 = np.uint8(dst2)
ret2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(dst2)
criteria2 = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners2 = cv2.cornerSubPix(gray2,np.float32(centroids2),(5,5),(-1,-1),criteria2)


for i in range(1, len(corners)):
    x, y = corners[i]
    x2, y2 = corners2[i]
    print(x,y)
    print(x2,y2)
    
    distance = np.sqrt((x- x2)**2 + (y - y2)**2)
    if distance < 0:
        distance = distance * -1
    print("distance: " + str(distance))        
    if(distance > 10):
        print("Mayor de 10")
        #break
    else:
        print("Menor de 10")

img[dst>0.1*dst.max()]=[0,0,255]
cv2.imshow('image', img)

img2[dst2>0.1*dst2.max()]=[0,0,255]
cv2.imshow('image2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows