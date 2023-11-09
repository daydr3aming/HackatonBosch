import numpy as np
import cv2 as cv
from math import e
import matplotlib.pyplot as plt
import random
import sympy as sp
from sklearn import preprocessing

from scipy.fft import fft, fftfreq
def valor_aleatorio():
    return random.randint(1, 4)


# Cargar la imagen original
img = cv.imread("Images/REF_23.png", cv.IMREAD_COLOR)  # Asegúrate de cargar la imagen en color



coordenadas = "34.3 53.9, 35.8 54.9, 38.4 55, 41.0 55.6 , 43.41 56.2, 45.1 56.5 , 46.3 56.2, 47.7 57.1 , 48.6 57.4 , 49.5 58.2 , 51.5 58.5, 53.0 59.1 , 54.5 59.1"

# Dividir las coordenadas basadas en la coma
pares_coordenadas = coordenadas.split(", ")

# Inicializar una lista para almacenar los resultados
resultados = []

# Iterar sobre cada par de coordenadas
for par in pares_coordenadas:
    x, y = map(float, par.split())
    resultados.append((x, y))



img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
# Definir las coordenadas del ROI
x1, y1, x2, y2 = 350, 170, 100, 135

# Obtener la región de interés (ROI)
roi = img_gray[y1:y1+y2, x1:x1+x2]

valores = []
'''
for i in range(1,97):
    intensity = roi[i,i+1]
    valores.append(intensity)
'''

# Imprimir la lista de resultados
for i, (x, y) in enumerate(resultados, 1):
    intensity = roi[int(x),int(y)]
    valores.append(intensity)

#print(valores)
#valores = [valor / max(valores) for valor in valores]

x = sp.Symbol("x")
#y= 3*e-10*(x**6) - 7*e-8*(x**5) + 6*e-6*(x**4) - 0.0003*(x**3) + 0.0045*(x**2) - 0.0275*x + 0.0719
#y = 3*(10**(-10))*(x**6) - 7*(10**(-8))*(x**5) + 6*(10**(-6))*(x**4) - 0.0003*(x**3) + 0.0045*(x**2) - 0.0275*x + 0.0719
y = 2*(e**-6)*(x**5) - 0.0005*(x**4) + 0.0379*(x**3) - 1.2084*(x**2) + 13.602*x - 29.229


LSF = sp.diff(y,x)
valores_x = range(1,97 )


# Crea una lista para almacenar los resultados de la derivada
resultados_derivada = []

for valor_x in valores_x:
    resultado = LSF.subs(x, valor_x)  # Evalúa la derivada en el valor_x
    resultados_derivada.append(resultado)
print(resultados_derivada)
#resultados_derivada = [valor / max(resultados_derivada) for valor in resultados_derivada]


# lsf_fourier = np.fft.fft(resultados_derivada)
lsf_fourier = np.fft.fft(resultados_derivada)

#mtf = np.abs(lsf_fourier) / np.max(np.abs(lsf_fourier))

mtf = np.abs(lsf_fourier)

pixel_size = 0.01  # Tamaño del píxel en mm
nyquist_frequency = 1 / (2 * pixel_size)

frequencies = np.fft.fftfreq(len(resultados_derivada))

n = len(lsf_fourier)

frequencies2 = np.fft.fftfreq(n)

'''
xmin = min(frequencies)
xmax = max(frequencies)
#print(xmax, xmin)
for i, x in enumerate(frequencies):
    frequencies[i] = (x - xmin) / (xmax - xmin)

#normalized_frequency = frequencies / nyquist_frequency

xmin2 = min(mtf)
xmax2 = max(mtf)
#print(xmax2, xmin2)
for i, x in enumerate(mtf):
    mtf[i] = (x - xmin2) / (xmax2 - xmin2)
    
'''


plt.plot(frequencies, mtf)
plt.xlabel('Frecuencia Espacial (ciclos/mm)')
plt.ylabel('MTF')
plt.title('Modulation Transfer Function')
plt.grid(True)
plt.show()
#'''
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
#'''

