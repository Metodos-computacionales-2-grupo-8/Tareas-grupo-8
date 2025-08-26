from scipy.fft import fft, ifft, fftfreq
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

#leer archivo
proyecciones = np.load("Taller 2 - Fourier/Datos/tomography_data/8.npy")
numero_angulos, numero_pixeles = proyecciones.shape

#Es cuadrada
filas = numero_pixeles
columnas = numero_pixeles
#angulos equiespaciados
angulos = np.linspace(0., 180., numero_angulos, endpoint=False)
#molde de la imagen para reemplazar y ya
imagen_total = np.zeros((filas, columnas))

#recorrer cada proyeccion y sumarla a la imagen
for pixel, angulo in zip(proyecciones, angulos):
    imagen_rotada = ndi.rotate(
    np.tile(pixel[:,None],filas).T,
    angulo,
    reshape=False,
    mode="reflect"
                               )
    imagen_total += imagen_rotada


    #ahora necesitamos filtrar las frecuencias bajas
#filtro ram-lak
# --- filtro Ram-Lak ---
def filtro_ram_lak(N):
    freqs = np.fft.fftfreq(N)   # frecuencias normalizadas
    return np.abs(freqs)        # rampa |f|

# aplicar filtro a cada proyección
filtro = filtro_ram_lak(numero_pixeles)
proyecciones_filtradas = np.zeros_like(proyecciones)

for i in range(numero_angulos):
    proyeccion_fft = fft(proyecciones[i])
    proyeccion_filtrada_fft = proyeccion_fft * filtro
    proyecciones_filtradas[i] = np.real(ifft(proyeccion_filtrada_fft))

#funcion para hacer la retroproyeccion
def retroproyeccion(proyecciones, titulo):
    num_angulos, num_pixeles = proyecciones.shape
    rows = num_pixeles
    reconstruccion = np.zeros((rows, rows))

    angulos = np.linspace(0, 180, num_angulos, endpoint=False)

    for p, ang in zip(proyecciones, angulos):
            slab = np.tile(p[:, None], rows).T
            rotado = ndi.rotate(slab, ang, reshape=False, mode="reflect")
            reconstruccion += rotado
    
    plt.imshow(reconstruccion, cmap="gray")
    plt.title(titulo)
    plt.colorbar()
    plt.savefig("Taller 2 - Fourier/4.png")
    
    # Llama la función para mostrar la reconstrucción filtrada
retroproyeccion(proyecciones_filtradas, "Reconstrucción con filtro")