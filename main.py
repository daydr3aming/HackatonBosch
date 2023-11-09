import numpy as np
import cv2 as cv

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk

import matplotlib.pyplot as plt

images = [1, 2, 4, 8, 9, 11, 12, 14, 18, 19, 20, 21, 22, 24, 26, 27, 28, 29, 32, 36]

def orientacion(image):

    img1 = cv.imread('Images/REF_23.png',cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread('Images/' + image ,cv.IMREAD_GRAYSCALE) # trainImage

    x1, y1, x2, y2 = 0, 0, 640, 120 

    roi_original = img1[y1:y1+y2, x1:x1+x2]
    roi_comparacion = img2[y1:y1+y2, x1:x1+x2]


    # Creamos la instancia de SIFT con sus Keypoints y descriptors 
    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(roi_original,None)
    kp2, des2 = sift.detectAndCompute(roi_comparacion,None)

    bf = cv.BFMatcher()

    try:
        matches = bf.knnMatch(des1,des2,k=2)

        good = []
        for m,n in matches:
            if m.distance < 1.00 * n.distance:
                good.append([m])
        # Checamos si hay algun match, si no, es por que está rotada
        img3 = cv.drawMatchesKnn(roi_original,kp1,roi_comparacion,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        if len(good) > 1:
            return "Go"
        else:
            return "No Go"
        
        plt.imshow(img3),plt.show()

    except:
        return "No go"

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
    # Utilizamos el Harris Corner Detection implementado en OpenCv para poder detectar puntos
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
        return distance


out = [None, None, None, None, None, None]
fields = ['#', 'Nitidez', 'Red Intensity', 'Green Intensity', 'Blue Intensity', 'Centrado', 'Orientacion']

#''' Orientacion
orient = ['Orientacion']
for i in range (0, len(images)):
    orient.append(orientacion(str(images[i]) + ".png"))

out[5] = orient
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

out[1] = red
out[2] = green
out[3] = blue

#'''
#''' Centrado final
cent = ['Centrado']
for i in range (0, len(images)):
    cent.append(centradoFinal(str(images[i]) + ".png"))

out[4] = cent
#'''

for i in range(0, 6):
    print(out[i])

app = tk.Tk()
app.title("Table Example")

# Create a Treeview widget for the table
table = ttk.Treeview(app, columns=fields, show="headings")

# Define column headings
for col in fields:
    table.heading(col, text=col)
    table.column(col, width=100)  # Adjust the width as needed

# Insert data from the 'num' list into the table
for i, data in enumerate(images, start=1):  # Start enumeration from 1
    table.insert("", "end", values=[data, "", out[1][i], out[2][i], out[3][i], out[4][i], out[5][i]])
print(out[1][5])
# Add a scrollbar
scrollbar = ttk.Scrollbar(app, orient="vertical", command=table.yview)
table.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")

# Pack the table
table.pack()

app.mainloop()