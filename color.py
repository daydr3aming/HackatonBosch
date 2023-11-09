import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt


img = cv.imread('Images/REF_23.png')

x1, y1, x2, y2 = 70, 0, 170, 480 
roi = img[y1:y1+y2, x1:x1+x2]

intensidad_promedio = np.mean(roi[:, :, :])
#plt.imshow(roi),plt.show()
mean_red = np.mean(roi[:, :, 0])  # Red channel
mean_green = np.mean(roi[:, :, 1])  # Green channel
mean_blue = np.mean(roi[:, :, 2])  # Blue channel

plt.imshow(roi),plt.show()


print(mean_red, mean_green, mean_blue, intensidad_promedio)
