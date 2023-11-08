import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

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
        

print(orientation("1.png"))