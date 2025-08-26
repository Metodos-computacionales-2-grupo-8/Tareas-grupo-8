import numpy as np
from PIL import Image
import os
#print("Directorio actual:", os.getcwd())

# ---------------------------
# PARTE A
# ---------------------------
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




# ---------------------------
# PARTE B
# ---------------------------
# -----------------------------------------------------------------------------
# Utilidades de E/S
# -----------------------------------------------------------------------------
def load_grayscale_image(image_path: str) -> np.ndarray:
    """Carga imagen en gris como float64 con validación de existencia."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"No se encontró la imagen en la ruta indicada: {image_path}")
    img = Image.open(image_path).convert("L")
    return np.array(img, dtype=np.float64)


def save_grayscale_image(image_array: np.ndarray, output_path: str) -> None:
    """Guarda un array 2D como imagen en escala de grises (uint8)."""
    # Normaliza a [0, 255]
    arr = image_array
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
    arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(arr_u8).save(output_path)
    print(f"Imagen guardada en: {output_path}")


def save_spectrum(Fs: np.ndarray, output_path: str) -> None:
    """Guarda el espectro (magnitud log) como PNG en escala de grises."""
    mag = np.log1p(np.abs(Fs))
    mn, mx = float(mag.min()), float(mag.max())
    if mx > mn:
        mag = (mag - mn) / (mx - mn) * 255.0
    spec_u8 = np.clip(mag, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(spec_u8).save(output_path)
    print(f"Espectro guardado en: {output_path}")


# -----------------------------------------------------------------------------
# FFT helpers
# -----------------------------------------------------------------------------
def fft2_centered(image: np.ndarray) -> np.ndarray:
    F = np.fft.fft2(image)
    return np.fft.fftshift(F)


def ifft2_centered(Fs: np.ndarray) -> np.ndarray:
    F = np.fft.ifftshift(Fs)
    img = np.fft.ifft2(F)
    return np.real(img)


# -----------------------------------------------------------------------------
# Enmascarado de picos (robusto contra bordes)
# -----------------------------------------------------------------------------
def zero_peak_region(Fs: np.ndarray, y: int, x: int, radius: int,
                     soft: bool = False, sigma: float | None = None, strength: float = 1.0) -> None:
    """
    Anula (o atenúa suavemente) un disco centrado en (y, x) con radio 'radius' sobre Fs.
    Trabaja en un recorte local para evitar desajustes de forma cerca de los bordes.
    """
    rows, cols = Fs.shape
    y0 = max(0, y - radius)
    y1 = min(rows, y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(cols, x + radius + 1)

    if y0 >= y1 or x0 >= x1:
        return  # completamente fuera

    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask_local = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2

    patch = Fs[y0:y1, x0:x1]
    if not soft:
        patch[mask_local] = 0
    else:
        # Atenuación tipo notch gaussiano para reducir ringing
        if sigma is None:
            sigma = max(1.0, radius / 2.0)
        g = np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma * sigma))
        patch[mask_local] = patch[mask_local] * (1.0 - strength * g[mask_local])

    Fs[y0:y1, x0:x1] = patch


def remove_peaks(Fs: np.ndarray, offsets: list[tuple[int | float, int | float]],
                 base_radius: int = 14, soft: bool = False) -> np.ndarray:
    """
    Aplica enmascaramiento en pares (dy, dx) y (-dy, -dx) relativos al centro del espectro.
    Preserva el componente DC.
    """
    rows, cols = Fs.shape
    cy, cx = rows // 2, cols // 2
    out = Fs.copy()

    for (dy, dx) in offsets:
        y = cy + int(round(dy))
        x = cx + int(round(dx))
        ys = cy - int(round(dy))
        xs = cx - int(round(dx))

        # Radio (constante o levemente adaptativo si deseas)
        radius = int(base_radius)

        zero_peak_region(out, y, x, radius, soft=soft)
        zero_peak_region(out, ys, xs, radius, soft=soft)

    # Preservar DC (brillo global)
    out[cy, cx] = Fs[cy, cx]
    return out


# -----------------------------------------------------------------------------
# Pipeline de procesamiento para una imagen
# -----------------------------------------------------------------------------
def process_image(image_path: str, output_image_path: str,
                  manual_offsets: list[tuple[int | float, int | float]],
                  base_radius: int = 14, soft: bool = False) -> None:
    """
    1) Carga imagen en gris
    2) FFT2 centrada
    3) Anulación MANUAL de picos en 'manual_offsets' (pares conjugados)
    4) IFFT y guardado de imagen filtrada
    5) Guardado de espectro filtrado (mismo nombre + _espectro.png)
    """
    # Cargar
    img = load_grayscale_image(image_path)

    # FFT centrada
    Fs = fft2_centered(img)

    # Anular picos manualmente
    Fs_filt = remove_peaks(Fs, manual_offsets, base_radius=base_radius, soft=soft)

    # Reconstrucción
    img_rec = ifft2_centered(Fs_filt)

    # Guardado de imagen
    save_grayscale_image(img_rec, output_image_path)

    # Guardado de espectro
    base, ext = os.path.splitext(output_image_path)
    spectrum_path = f"{base}_espectro.png"
    save_spectrum(Fs_filt, spectrum_path)


# -----------------------------------------------------------------------------
# Entradas/salidas del ejercicio 3.b – EJECUCIÓN DIRECTA
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Directorio de salida
    out_dir = r"Taller 2 - Fourier"
    os.makedirs(out_dir, exist_ok=True)

    # 3.b.a – P_a_t_o: picos diagonales (ajustados a patrón típico)
    pato_path = r"Taller 2 - Fourier/Imagenes/p_a_t_o.jpg"
    pato_out = os.path.join(out_dir, "3.b.a.jpg")
    # Offsets manuales relativos al centro (dy, dx)
    # Pares principales + segundo armónico para robustez
    pato_offsets = [
        (40, 40), (-40, 40), (40, -40), (-40, -40),
        (80, 80), (-80, 80), (80, -80), (-80, -80),
    ]
    process_image(pato_path, pato_out, manual_offsets=pato_offsets, base_radius=14, soft=False)

    # 3.b.b – G_a_t_o: patrón tipo “persianas” (vertical y horizontal)
    gato_path = r"Taller 2 - Fourier/Imagenes/g_a_t_o.png"
    gato_out = os.path.join(out_dir, "3.b.b.png")
    # Pares principales (fundamental) en ejes horizontal y vertical + segundo armónico
    gato_offsets = [
        (0, 48), (0, -48),   # vertical
        (48, 0), (-48, 0),   # horizontal
        (0, 96), (0, -96),   # armónico vertical
        (96, 0), (-96, 0),   # armónico horizontal
    ]
    process_image(gato_path, gato_out, manual_offsets=gato_offsets, base_radius=12, soft=False)

    print("Procesamiento 3.b completado. Revisa las imágenes en:", out_dir)