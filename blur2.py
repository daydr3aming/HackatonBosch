import numpy as np
import cv2 as cv

import matplotlib.pyplot as plt
import random
import sympy as sp
def valor_aleatorio():
    return random.randint(1, 4)


# Cargar la imagen original
img = cv.imread("Images/REF_23.png", cv.IMREAD_COLOR)  # Asegúrate de cargar la imagen en color


img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# Definir las coordenadas del ROI
x1, y1, x2, y2 = 350, 170, 100, 135

# Obtener la región de interés (ROI)
roi = img_gray[y1:y1+y2, x1:x1+x2]

valores = []
for i in range(1,97):
    intensity = roi[valor_aleatorio(),i]
    valores.append(intensity)



valores = [valor / max(valores) for valor in valores]







lsf_fourier = np.fft.fft(valores)

# Calcular la magnitud de la Transformada de Fourier
magnitud_lsf_fourier = np.abs(lsf_fourier)

# Crear un rango de frecuencias espaciales
num_muestras = len(valores)
frecuencias_espaciales = np.fft.fftfreq(num_muestras)



# Graficar la magnitud de la Transformada de Fourier
plt.plot(frecuencias_espaciales,magnitud_lsf_fourier)
plt.xlabel("Frecuencia Espacial")
plt.ylabel("Magnitud")
plt.title("Magnitud de la Transformada de Fourier de LSF")
plt.grid()
plt.show()





# ESF
# Crear una figura y un eje para el gráfico
plt.figure(figsize=(10, 4))
plt.plot(valores, label="ESF")

# Agregar etiquetas y título
plt.xlabel("Píxeles")
plt.ylabel("Intensidad")
plt.title("ESF")

# Mostrar la leyenda
plt.legend()

# Mostrar el gráfico
plt.show()

# Mostrar la imagen original con los bordes superpuestos en la ROI
plt.imshow(cv.cvtColor(roi, cv.COLOR_BGR2RGB))
plt.show()

