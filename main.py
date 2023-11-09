import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

images = [1, 2, 4, 8, 9, 11, 12, 14, 18, 19, 20, 21, 22, 24, 26, 27, 28, 29, 32, 36]

def orientacion(image):

    img1 = cv.imread('Images/REF_23.png',cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread('Images/' + image ,cv.IMREAD_GRAYSCALE) # trainImage

    x1, y1, x2, y2 = 0, 0, 640, 120 

    roi_original = img1[y1:y1+y2, x1:x1+x2]
    roi_comparacion = img2[y1:y1+y2, x1:x1+x2]

    #cv.imwrite('roi_original.jpg', roi_original)
    #cv.imwrite('roi_comparacion.jpg', roi_comparacion)
    #plt.imshow(roi_comparacion),plt.show()
    #plt.imshow(roi_original),plt.show()

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(roi_original,None)
    kp2, des2 = sift.detectAndCompute(roi_comparacion,None)
    # BFMatcher with default params
    bf = cv.BFMatcher()

    try:
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 1.00 * n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv.drawMatchesKnn(roi_original,kp1,roi_comparacion,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if len(good) > 1:
            return "Pasa la prueba"
        else:
            return "Falla la prueba"
        
        #plt.imshow(img3),plt.show()

    except:
        return "Falla la prueba"

def intensidad(image):

    # Carga la imagen en color
    imagen = cv.imread('Images/' + image)
    x1, y1, x2, y2 = 199, 161, 45, 137 
    x3, y3, x4, y4 = 263, 114, 139, 39
    x5 , y5, x6, y6 = 414, 178, 26, 130
    x7, y7, x8, y8 = 245, 321, 141 , 29

    roi = imagen[y1:y1+y2, x1:x1+x2]
    roi2 = imagen[y3:y3+y4, x3:x3+x4]
    roi3 = imagen[y5:y5+y6, x5:x5+x6]
    roi4 = imagen[y7:y7+y8, x7:x7+x8]

    # Separa los canales de color
    b, g, r = cv.split(roi)
    b2, g2, r2 = cv.split(roi2)
    b3, g3, r3 = cv.split(roi3)
    b4, g4, r4 = cv.split(roi4)

    #cv.imshow("lol",roi)
    #cv.imshow("lol2",roi2)
    #cv.imshow("lol3",roi3)
    #cv.imshow("lol4",roi4)
    #cv.waitKey(0)

    # Calcula el promedio de intensidad para cada canal
    promedio_r = np.mean(r)
    promedio_g = np.mean(g)
    promedio_b = np.mean(b)

    promedio_r2= np.mean(r2)
    promedio_g2 = np.mean(g2)
    promedio_b2= np.mean(b2)

    promedio_r3 = np.mean(r3)
    promedio_g3 = np.mean(g3)
    promedio_b3 = np.mean(b3)

    promedio_r4 = np.mean(r4)
    promedio_g4= np.mean(g4)
    promedio_b4 = np.mean(b4)

    valoresR = [promedio_r, promedio_r2, promedio_r3, promedio_r4]
    valoresG = [promedio_g, promedio_g2, promedio_g3, promedio_g4]
    valoresB = [promedio_b, promedio_b2, promedio_b3, promedio_b4]

    promedioFinalR = np.mean(valoresR)
    promedioFinalG = np.mean(valoresG)
    promedioFinalB = np.mean(valoresB)

    return promedioFinalR, promedioFinalG, promedioFinalB
    #print(f"Promedio del canal Rojo (R): {promedioFinalR}")
    #print(f"Promedio del canal Verde (G): {promedioFinalG}")
    #print(f"Promedio del canal Azul (B): {promedioFinalB}")

def centradoFinal(image):

    img = cv.imread('Images/REF_23.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray,5,3,0.04)
    ret, dst = cv.threshold(dst,0.1*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv.connectedComponentsWithStats(dst)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

    img2 = cv.imread('Images/' + image)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray2 = np.float32(gray2)
    dst2 = cv.cornerHarris(gray2,5,3,0.04)
    ret2, dst2 = cv.threshold(dst2,0.1*dst2.max(),255,0)
    dst2 = np.uint8(dst2)
    ret2, labels2, stats2, centroids2 = cv.connectedComponentsWithStats(dst2)
    criteria2 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners2 = cv.cornerSubPix(gray2,np.float32(centroids2),(5,5),(-1,-1),criteria2)


    for i in range(1, len(corners)):
        x, y = corners[i]
        x2, y2 = corners2[i]
        #print(x,y)
        #print(x2,y2)
        
        distance = np.sqrt((x- x2)**2 + (y - y2)**2)
        if distance < 0:
            distance = distance * -1
        #print("distance: " + str(distance))        
        if(distance > 10):
            #print("Mayor de 10")
            return "Falla la prueba"
            #break
        else:
            #print("Menor de 10")
            return "Pasa la prueba"

    #img[dst>0.1*dst.max()]=[0,0,255]
    #cv2.imshow('image', img)

    #img2[dst2>0.1*dst2.max()]=[0,0,255]
    #cv2.imshow('image2', img2)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows


out = ['#', 'Nitidez', 'Red Intensity', 'Green Intensity', 'Blue Intensity', 'Centrado', 'Orientacion']

#''' #
num = ['#', 1, 2, 4, 8, 9, 11, 12, 14, 18, 19, 20, 21, 22, 24, 26, 27, 28, 29, 32, 36]
out[0] = num
#'''

#''' Orientacion
orient = ['Orientacion']
for i in range (0, len(images)):
    orient.append(orientacion(str(images[i]) + ".png"))

out[6] = orient
#'''

#''' Intensidad
red = ['Red']
green = ['Green']
blue = ['Blue']

for i in range(0, len(images)):
    results = intensidad(str(images[i]) + ".png")
    red.append(results[0])
    green.append(results[1])
    blue.append(results[2])

out[2] = red
out[3] = green
out[4] = blue

#'''

#''' Centrado final
cent = ['Centrado']
for i in range (0, len(images)):
    cent.append(centradoFinal(str(images[i]) + ".png"))

out[5] = cent
#'''

for i in range(0, 7):
    print(out[i])
