import cv2
import numpy as np
    

# Carga la imagen en color
imagen = cv2.imread('Images/11.png')
x1, y1, x2, y2 = 199, 161, 45, 137 

x3, y3, x4, y4 = 263, 114, 139, 39

x5 , y5, x6, y6 = 414, 178, 26, 130

x7, y7, x8, y8 = 245, 321, 141 , 29
roi = imagen[y1:y1+y2, x1:x1+x2]

roi2 = imagen[y3:y3+y4, x3:x3+x4]

roi3 = imagen[y5:y5+y6, x5:x5+x6]

roi4 = imagen[y7:y7+y8, x7:x7+x8]

# Separa los canales de color
b, g, r = cv2.split(roi)
b2, g2, r2 = cv2.split(roi2)
b3, g3, r3 = cv2.split(roi3)
b4, g4, r4 = cv2.split(roi4)

cv2.imshow("lol",roi)
cv2.imshow("lol2",roi2)
cv2.imshow("lol3",roi3)
cv2.imshow("lol4",roi4)
cv2.waitKey(0)

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

print(f"Promedio del canal Rojo (R): {promedioFinalR}")
print(f"Promedio del canal Verde (G): {promedioFinalG}")
print(f"Promedio del canal Azul (B): {promedioFinalB}")