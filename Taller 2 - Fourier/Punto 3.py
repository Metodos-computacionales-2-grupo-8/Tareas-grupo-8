import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from PIL import ImageDraw
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



# Análisis y evidencias del espectro (para 3.b.a y 3.b.b)
# Detecta picos fuertes en el espectro centrado y guarda espectros anotados
# ---------------------------
# PARTE B
# ---------------------------
def _compute_centered_spectrum(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.float64)
    F = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.log1p(np.abs(F))
    return F, mag

def _to_u8(imgf: np.ndarray) -> np.ndarray:
    mn, mx = float(imgf.min()), float(imgf.max())
    if mx > mn:
        imgn = (imgf - mn) / (mx - mn) * 255.0
    else:
        imgn = np.zeros_like(imgf)
    return np.clip(imgn, 0, 255).astype(np.uint8)

def detect_top_peaks(F: np.ndarray, top_k: int = 8, exclude_radius: int = 8, min_dist: int = 6):
    """
    Selecciona los top_k picos en magnitud (excluye centro con exclude_radius).
    Devuelve lista de offsets (dy, dx) relativos al centro (solo un lado).
    """
    rows, cols = F.shape
    cy, cx = rows // 2, cols // 2
    mag = np.abs(F)

    # Excluir región baja frecuencia alrededor del DC
    yy, xx = np.ogrid[:rows, :cols]
    mask_dc = (yy - cy) ** 2 + (xx - cx) ** 2 <= exclude_radius ** 2
    mag_masked = mag.copy()
    mag_masked[mask_dc] = 0.0

    # Obtener índices ordenados por magnitud descendente
    flat_idx = np.argsort(mag_masked.ravel())[::-1]
    chosen = []
    chosen_coords = []

    for idx in flat_idx:
        if len(chosen) >= top_k:
            break
        y, x = divmod(int(idx), cols)
        # evitar cercanía a picos ya seleccionados
        too_close = False
        for (py, px) in chosen_coords:
            if (y - py) ** 2 + (x - px) ** 2 < min_dist ** 2:
                too_close = True
                break
        if too_close:
            continue
        # evitar el DC y bordes triviales
        if mask_dc[y, x]:
            continue
        chosen_coords.append((y, x))
        chosen.append((y - cy, x - cx))

    return chosen

def annotate_and_save_spectrum(F: np.ndarray, mag: np.ndarray, offsets: list[tuple[int, int]], out_path: str):
    """
    Guarda espectro log normalizado y una versión anotada (marcando picos y sus conjugados).
    """
    mag_u8 = _to_u8(mag)
    # imagen espectro en RGB para anotación
    spec_rgb = np.stack([mag_u8, mag_u8, mag_u8], axis=2)
    img_spec = Image.fromarray(spec_rgb)
    draw = ImageDraw.Draw(img_spec)
    rows, cols = mag.shape
    cy, cx = rows // 2, cols // 2

    # marcar cada pico y su conjugado
    r = 6
    for (dy, dx) in offsets:
        y1, x1 = cy + int(dy), cx + int(dx)
        y2, x2 = cy - int(dy), cx - int(dx)
        # círculos rojos
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=(255, 0, 0), width=2)
        draw.ellipse((x2 - r, y2 - r, x2 + r, y2 + r), outline=(255, 0, 0), width=2)
        # línea al centro (opcional, visual ayuda)
        draw.line((cx, cy, x1, y1), fill=(255, 0, 0), width=1)
        draw.line((cx, cy, x2, y2), fill=(255, 0, 0), width=1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img_spec.save(out_path)
    print(f"Espectro anotado guardado en: {out_path}")

# Ejecución inmediata (cuando se corre el script) para generar evidencia:
if __name__ == "__main__":
    pruebas = [
        (r"Taller 2 - Fourier/Imagenes/p_a_t_o.jpg", r"Taller 2 - Fourier/esp_p_a_t_o_espectro_anotado.png", 8),
        (r"Taller 2 - Fourier/Imagenes/g_a_t_o.png", r"Taller 2 - Fourier/esp_g_a_t_o_espectro_anotado.png", 8),
    ]
    for img_path, out_annot, topk in pruebas:
        if not os.path.exists(img_path):
            print(f"Advertencia: no se encontró {img_path}, omitiendo.")
            continue
        F, mag = _compute_centered_spectrum(img_path)
        peaks = detect_top_peaks(F, top_k=topk, exclude_radius=10, min_dist=10)
        print(f"Imagen: {img_path}")
        print("Picos detectados (dy, dx) relativos al centro) — seleccionar estos pares y sus conjugados para enmascarar:")
        for p in peaks:
            print("  ", p)
        annotate_and_save_spectrum(F, mag, peaks, out_annot)
    print("Análisis de espectros (evidencias) generado. Revisa los archivos *_espectro_anotado.png")

# -----------------------------------------------------------------------------
# Utilidades de E/S
# -----------------------------------------------------------------------------
def remove_peaks_and_save(img_path: str, out_path: str, top_k: int = 8, exclude_radius: int = 10, min_dist: int = 10):
    # Cargar imagen y calcular espectro
    img = Image.open(img_path).convert("L")
    arr = np.array(img, dtype=np.float64)
    F = np.fft.fftshift(np.fft.fft2(arr))
    peaks = detect_top_peaks(F, top_k=top_k, exclude_radius=exclude_radius, min_dist=min_dist)
    rows, cols = F.shape
    cy, cx = rows // 2, cols // 2

    # Eliminar picos y sus conjugados
    for dy, dx in peaks:
        y1, x1 = cy + int(dy), cx + int(dx)
        y2, x2 = cy - int(dy), cx - int(dx)
        F[y1, x1] = 0
        F[y2, x2] = 0

    # Reconstruir imagen
    arr_mod = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    arr_mod_u8 = np.clip(arr_mod, 0, 255).astype(np.uint8)
    Image.fromarray(arr_mod_u8).save(out_path)
    print(f"Imagen sin interferencia guardada en: {out_path}")

# Ejecutar para las dos imágenes de ejemplo
if __name__ == "__main__":
    pruebas_b = [
        (r"Taller 2 - Fourier/Imagenes/p_a_t_o.jpg", r"Taller 2 - Fourier/3.b.a.jpg", 8),
        (r"Taller 2 - Fourier/Imagenes/g_a_t_o.png", r"Taller 2 - Fourier/3.b.b.png", 8),
    ]
    for img_path, out_path, topk in pruebas_b:
        if not os.path.exists(img_path):
            print(f"Advertencia: no se encontró {img_path}, omitiendo.")
            continue
        remove_peaks_and_save(img_path, out_path, top_k=topk, exclude_radius=10, min_dist=10)