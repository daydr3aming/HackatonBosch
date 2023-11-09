import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt

img = cv.imread("Images/REF_23.png", cv.COLOR_BGR2GRAY)

x1, y1, x2, y2 =  350, 170, 100, 135
roi = img[y1:y1+y2, x1:x1+x2]

gauss = cv.GaussianBlur(roi, (5,5), 0)

canny = cv.Canny(roi, 50, 150)

plt.imshow(canny),plt.show()


