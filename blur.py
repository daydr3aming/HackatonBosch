import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt


# Cargar la imagen original
img = cv.imread("Images/REF_23.png", cv.IMREAD_COLOR)  # Asegúrate de cargar la imagen en color


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Definir las coordenadas del ROI
x1, y1, x2, y2 = 350, 170, 100, 135

# Obtener la región de interés (ROI)
roi = img_gray[y1:y1+y2, x1:x1+x2]

intensity = roi[20, 40]
print(intensity)

# Mostrar la imagen original con los bordes superpuestos en la ROI
plt.imshow(cv.cvtColor(roi, cv.COLOR_BGR2RGB))
plt.show()

