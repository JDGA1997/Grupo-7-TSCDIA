#Como crear un histograma desde el dataset de solo imagenes

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Ruta a la carpeta con imágenes
carpeta_imagenes = 'Database/Glaucoma_Training'

# Inicializar un array para almacenar el histograma global
histograma_global = np.zeros(256)

# Leer y procesar cada imagen en la carpeta
for nombre_archivo in os.listdir(carpeta_imagenes):
    if nombre_archivo.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        # Leer la imagen
        ruta_imagen = os.path.join(carpeta_imagenes, nombre_archivo)
        imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)

        # Calcular el histograma de la imagen
        histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])

        # Sumar el histograma al histograma global
        histograma_global += histograma.flatten()

# Normalizar el histograma global
histograma_global /= len(os.listdir(carpeta_imagenes))

# Graficar el histograma global
plt.figure(figsize=(10, 5))
plt.title('Histograma Global de Intensidades de Píxeles')
plt.xlabel('Intensidad')
plt.ylabel('Frecuencia')
plt.plot(histograma_global)
plt.xlim([0, 256])
plt.show()
