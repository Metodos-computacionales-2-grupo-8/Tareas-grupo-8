import numpy as np
from PIL import Image
import os
#print("Directorio actual:", os.getcwd())

# Importar las librerías necesarias

# Abrir la imagen y convertirla en un arreglo numpy
img = np.array(Image.open("Taller 2 - Fourier\Imagenes\miette.jpg"))
#img = np.array(Image.open(r"C:\Users\USUARIO\OneDrive - Universidad de los Andes\Uniandes\Univerisity stuff\SÉPTIMO SEMESTRE\Metodos computacionales\Tareas-grupo-8\Taller 2 - Fourier\Imagenes\miette.jpg"))

# Obtener las dimensiones de la imagen
rows, cols, channels = img.shape

# Crear una malla de coordenadas para la gaussiana
x = np.linspace(-0.5, 0.5, cols)
y = np.linspace(-0.5, 0.5, rows)
X, Y = np.meshgrid(x, y)

# Definir el ancho de la gaussiana (sigma)
sigma = 0.05

# Crear la gaussiana en el espacio de la frecuencia
gaussian = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# Inicializar una imagen para almacenar el resultado borrosoTest-Path .\Imagenes\miette.jpg
blurred_img = np.zeros_like(img)

# Procesar cada canal de color por separado
for c in range(channels):
    # Obtener el canal
    channel = img[:, :, c]
    # Calcular la transformada de Fourier 2D y centrarla con fftshift
    f_channel = np.fft.fftshift(np.fft.fft2(channel))
    # Multiplicar la transformada por la gaussiana (filtrado en frecuencia)
    f_blurred = f_channel * gaussian
    # Transformar de vuelta al espacio de la imagen y tomar la parte real
    blurred_channel = np.real(np.fft.ifft2(np.fft.ifftshift(f_blurred)))
    # Guardar el canal borroso en la imagen de salida
    blurred_img[:, :, c] = np.clip(blurred_channel, 0, 255)

# Convertir la imagen borrosa a tipo uint8
blurred_img_uint8 = blurred_img.astype(np.uint8)

# Guardar la imagen borrosa como "3.a.jpg"
Image.fromarray(blurred_img_uint8).save("Taller 2 - Fourier/3.a.jpg")