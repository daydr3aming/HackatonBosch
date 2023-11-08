import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def distance(kp_original):
    keypoints_unicos = []
    for kp in kp_original:
        esta_cerca = False
        for kp_unico in keypoints_unicos:
            distancia = kp.pt[1] - kp_unico.pt[1]
            dist = kp.pt[0] - kp_unico.pt[0] 
            if distancia < 0:
                distancia = distancia * - 1
            if dist < 0:
                dist = dist * - 1
            
            distancia = np.sqrt((dist)*2 + (distancia)*2)

            if distancia < 5:
                esta_cerca = True
                break
        if not esta_cerca:
            keypoints_unicos.append(kp)
    
    return keypoints_unicos


img1 = cv.imread('Images/REF_23.png', cv.IMREAD_GRAYSCALE)          # queryImage
#img2 = cv.imread('Images/' + image, cv.IMREAD_GRAYSCALE) # trainImage
img2 = cv.imread('Images/8.png', cv.IMREAD_GRAYSCALE) # trainImage

x1, y1, x2, y2 = 200, 120, 250, 250  

roi_original = img1[y1:y1+y2, x1:x1+x2]
roi_comparacion = img2[y1:y1+y2, x1:x1+x2]

img1 = cv.threshold(img1, 128, 255, cv.THRESH_BINARY)
img2 = cv.threshold(img2, 128, 255, cv.THRESH_BINARY)

# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp_original = orb.detect(roi_original,None)
kp_comparation = orb.detect(roi_comparacion,None)

# compute the descriptors with ORB
kp_original, des_original = orb.compute(roi_original, kp_original)
kp_comparation, des_comparation = orb.compute(roi_comparacion, kp_comparation)

keypoints_unicosOriginal = distance(kp_original)
keypoints_unicosComparation = distance(kp_comparation)

# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(roi_original, keypoints_unicosOriginal, None, color=(0,255,0), flags=0)
img3 = cv.drawKeypoints(roi_comparacion, keypoints_unicosComparation, None, color=(0,255,0), flags=0)

plt.imshow(img2), plt.show()
plt.imshow(img3), plt.show()

if len(keypoints_unicosOriginal) == len(keypoints_unicosComparation):
    # Calcula la diferencia en coordenadas (desfase) entre los keypoints
    desfase = [(kp1.pt[0] - kp2.pt[0], kp1.pt[1] - kp2.pt[1]) for kp1, kp2 in zip(keypoints_unicosOriginal, keypoints_unicosComparation)]
    
    for i in range (0, len(desfase)):
        x, y = desfase[i]
        if x < 0:
            x = x * - 1
        if  y < 0:
            y = y * - 1

        if x < 10 or y < 10:
            print("Ya jalo")
        else:
            print("Nel")

else:
    print("La cantidad de keypoints en ambas imágenes no coincide.")

