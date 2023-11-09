import numpy as np
import cv2 as cv


import os
import matplotlib.pyplot as plt


def intensity(image):

    img = cv.imread('Images/' + image)
    x1, y1, x2, y2 = 200, 50, 100, 100 
    roi = img[y1:y1+y2, x1:x1+x2]
    intensidad_promedio = np.mean(roi)
    #plt.imshow(roi),plt.show()
    mean_red = np.mean(roi[:, :, 0])  # Red channel
    mean_green = np.mean(roi[:, :, 1])  # Green channel
    mean_blue = np.mean(roi[:, :, 2])  # Blue channel
    
    return mean_red, mean_green, mean_blue, intensidad_promedio


def orientation(image):

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
            return "Go"
        else:
            return "No Go"
        
        #plt.imshow(img3),plt.show()

    except:
        return "No Go"
        
    

images = [1, 2, 4, 8, 9, 11, 12, 14, 18, 19, 20, 21, 22, 24, 26, 27, 28, 29, 32, 36]

for i in range (0, len(images)):
    print(orientation(str(images[i]) + ".png"))

intesityresults = (intensity("REF_23" + ".png"))

print("REF_23_IMAGE: ")
print("Red: ")
print(intesityresults[0])
print("Green: ")
print(intesityresults[1])
print("Blue: ")
print(intesityresults[2])
print("Mean intensity: ")
print(intesityresults[3])
print(" ")

for i in range(0, len(images)):
    results = intensity(str(images[i]) + ".png")
    print("IMAGE: " + str(images[i]) + ".png")
    print("Red: ")
    print(results[0])
    print("Green: ")
    print(results[1])
    print("Blue: ")
    print(results[2])
    print("Mean intensity: ")
    print(results[3])
    print(" ")