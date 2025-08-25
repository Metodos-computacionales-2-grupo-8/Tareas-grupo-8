import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------------------
# PARTE A.a
# ---------------------------
def FourierTransform(t, y, freqs):
    N = len(t)
    T = t[1] - t[0]  # Assuming uniform spacing
    transformed_signal = np.array([np.sum(y * np.exp(-2j * np.pi * f * t)) for f in freqs]) * T / N
    return transformed_signal

# ---------------------------
# PARTE A.b
# ---------------------------
def generate_data(tmax,dt,A,freq,noise):
    '''Generates a sin wave of the given amplitude (A) and frequency (freq),
    sampled at times going from t=0 to t=tmax, taking data each dt units of time.
    A random number with the given standard deviation (noise) is added to each data point.
    ----------------
    Returns an array with the times and the measurements of the signal. '''
    ts = np.arange(0,tmax+dt,dt)
    return ts, np.random.normal(loc=A*np.sin(2*np.pi*ts*freq),scale=noise)

# Parámetros de la señal
tmax = 1      # duración total en segundos
dt = 0.001    # intervalo de muestreo en segundos
A = 1         # amplitud
freq = 5      # frecuencia de la señal en Hz
noise = 0.2   # desviación estándar del ruido

# Generar datos
t, y = generate_data(tmax, dt, A, freq, noise)

# Calcular frecuencia de Nyquist
fs = 1 / dt  # frecuencia de muestreo
f_nyquist = fs / 2

# Definir el rango de frecuencias para el espectro
freqs = np.linspace(0, 2.7 * f_nyquist, 1000)

# Calcular la Transformada de Fourier
spectrum = FourierTransform(t, y, freqs)

# Graficar el espectro
plt.figure(figsize=(8,4))
plt.plot(freqs, np.abs(spectrum))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud')
plt.title('Espectro hasta 2.7 veces la frecuencia de Nyquist')
plt.grid(True)
plt.savefig("Taller 2 - Fourier/1.a.pdf", bbox_inches="tight", pad_inches=0.1)
#plt.show()

# Explicación:
# 1. Se generan los datos de una señal senoidal con ruido usando generate_data.
# 2. Se calcula la frecuencia de Nyquist, que es la mitad de la frecuencia de muestreo.
# 3. Se define un rango de frecuencias hasta 2.7 veces la frecuencia de Nyquist.
# 4. Se calcula la transformada de Fourier en ese rango de frecuencias.
# 5. Se grafica el espectro de magnitud en función de la frecuencia.



# ---------------------------
# PARTE B
# ---------------------------
# Definir los valores de SNtime logarítmicamente distribuidos entre 0.01 y 1.0
SNtime_values = np.logspace(-2, 0, 20)  # 20 valores entre 0.01 y 1.0

SNfreq_values = []  # Lista para guardar los SNfreq calculados

for SNtime in SNtime_values:
    # Para cada SNtime, calcular el valor de noise necesario para obtener ese SNtime
    # SNtime = A / noise => noise = A / SNtime
    noise_actual = A / SNtime

    # Generar nueva señal con ese nivel de ruido
    t, y = generate_data(tmax, dt, A, freq, noise_actual)

    # Calcular el espectro de Fourier para la señal generada
    spectrum = FourierTransform(t, y, freqs)

    # Encontrar la altura del pico principal (máximo valor absoluto en el espectro)
    peak_height = np.max(np.abs(spectrum))

    # Para estimar el ruido en el espectro, excluir una ventana alrededor del pico
    peak_index = np.argmax(np.abs(spectrum))
    window_size = 20  # Número de puntos a excluir alrededor del pico

    # Crear máscara para excluir la ventana del pico
    mask = np.ones_like(spectrum, dtype=bool)
    mask[max(0, peak_index - window_size):min(len(spectrum), peak_index + window_size + 1)] = False

    # Calcular la desviación estándar del espectro fuera del pico (ruido de fondo)
    noise_std = np.std(np.abs(spectrum)[mask])

    # Calcular SNfreq como la razón entre el pico y el ruido de fondo
    SNfreq = peak_height / noise_std
    SNfreq_values.append(SNfreq)

# Graficar SNfreq vs SNtime en escala log-log
plt.figure(figsize=(7,4))
plt.loglog(SNtime_values, SNfreq_values, marker='o')
plt.xlabel('SN en el dominio temporal (A/noise)')
plt.ylabel('SN en el dominio de frecuencias (pico/std fondo)')
plt.title('Relación entre SNtime y SNfreq')
plt.grid(True, which="both", ls="--")
plt.savefig("Taller 2 - Fourier/1.b.pdf", bbox_inches="tight", pad_inches=0.1)
#plt.show()

# Modelo: En general, SNfreq crece con SNtime, pero la relación puede no ser exactamente lineal
# debido a cómo el ruido se distribuye en el espectro. En log-log, la relación suele ser aproximadamente lineal.
# Variables que afectan el comportamiento: A (amplitud), tmax (duración de la señal), freq (frecuencia de la señal), dt (intervalo de muestreo).
# Por ejemplo, aumentar tmax mejora la resolución espectral y puede aumentar SNfreq.


# ---------------------------
# PARTE C
# ---------------------------
# Definir los valores de tmax para analizar cómo afecta el ancho de los picos
#tmax_values = [0.5, 1, 2, 4]  # Distintas duraciones de la señal
tmax_values = np.linspace(0.1, 30, 120)  # 10 valores entre 0.1 y 5 segundos

# Lista para guardar los anchos de pico para cada tmax
peak_widths = []

for tmax_c in tmax_values:
    # Generar datos con el tmax actual, manteniendo los otros parámetros fijos
    t, y = generate_data(tmax_c, dt, A, freq, noise)
    
    # Calcular el espectro de Fourier para la señal generada
    spectrum = FourierTransform(t, y, freqs)
    
    # Encontrar el índice del pico principal (máximo valor absoluto en el espectro)
    peak_index = np.argmax(np.abs(spectrum))
    
    # Calcular el ancho del pico a la mitad de la altura máxima (FWHM)
    peak_height = np.abs(spectrum[peak_index])
    half_max = peak_height / 2  # Valor de la mitad de la altura máxima
    
    # Buscar los índices donde el espectro cruza la mitad de la altura máxima a la izquierda y derecha del pico
    left = peak_index
    while left > 0 and np.abs(spectrum[left]) > half_max:
        left -= 1
    right = peak_index
    while right < len(spectrum)-1 and np.abs(spectrum[right]) > half_max:
        right += 1
    
    # Calcular el ancho del pico en Hz (diferencia de frecuencia entre los cruces)
    width = freqs[right] - freqs[left]
    peak_widths.append(width)  # Guardar el ancho calculado

# Graficar el ancho de pico vs tmax
plt.figure(figsize=(7,4))
plt.plot(tmax_values, peak_widths, marker='o')
plt.xlabel('Duración de la señal $t_{max}$ (s)')  # Etiqueta del eje x
plt.ylabel('Ancho de pico (Hz)')                  # Etiqueta del eje y
plt.title('Ancho de pico vs duración de la señal')# Título del gráfico
plt.grid(True)
plt.savefig("Taller 2 - Fourier/1.c.pdf", bbox_inches="tight", pad_inches=0.1)
#plt.show()

# Comentario:
# El ancho del pico disminuye al aumentar tmax, porque una señal más larga permite una mejor resolución en frecuencia.
# Si se cambia dt (intervalo de muestreo), se modifica la frecuencia de Nyquist y el rango de frecuencias representables.
# Cambiar freq (frecuencia de la señal) mueve la posición del pico, pero no su ancho.
# Cambiar noise afecta la visibilidad del pico, pero no su ancho.


# ---------------------------
# PARTE D: Bono - Ruido en el muestreo
# ---------------------------
def generate_data(tmax, dt, A, freq, noise, sampling_noise=0):
    # Genera los tiempos de muestreo uniformemente espaciados
    ts = np.arange(0, tmax + dt, dt)
    # Si sampling_noise > 0, agrega ruido gaussiano a los tiempos de muestreo
    if sampling_noise > 0:
        # Perturba cada tiempo de muestreo con ruido normal de desviación estándar sampling_noise
        ts_perturbed = ts + np.random.normal(0, sampling_noise, size=ts.shape)
    else:
        # Si no hay ruido de muestreo, usa los tiempos originales
        ts_perturbed = ts
    # Calcula la señal senoidal en los tiempos perturbados
    signal = A * np.sin(2 * np.pi * freq * ts_perturbed)
    # Agrega ruido gaussiano a la señal medida
    measured_signal = np.random.normal(loc=signal, scale=noise)
    # Retorna los tiempos perturbados y la señal medida
    return ts_perturbed, measured_signal

# Definir varios valores de sampling_noise para analizar su efecto
sampling_noise_values = [0, 0.0005, 0.001, 0.002, 0.005, 0.01]  # Diferentes niveles de ruido en el muestreo

plt.figure(figsize=(10, 6))  # Crea una figura para graficar los espectros

for sn in sampling_noise_values:
    # Genera datos con el nivel de ruido de muestreo actual
    t_perturbed, y_perturbed = generate_data(tmax, dt, A, freq, noise, sampling_noise=sn)
    # Calcula la transformada de Fourier usando los tiempos perturbados
    spectrum_perturbed = FourierTransform(t_perturbed, y_perturbed, freqs)
    # Grafica el espectro de magnitud para este nivel de ruido de muestreo
    plt.plot(freqs, np.abs(spectrum_perturbed), label=f'sampling_noise={sn}')
    # Explicación:
    # - Para cada valor de sampling_noise, se genera una señal con tiempos de muestreo perturbados.
    # - Se calcula la transformada de Fourier de esa señal.
    # - Se grafica el espectro para comparar cómo cambia con el ruido de muestreo.

plt.xlabel('Frecuencia (Hz)')  # Etiqueta del eje x
plt.ylabel('Magnitud')         # Etiqueta del eje y
plt.title('Efecto del ruido en el muestreo sobre el espectro de Fourier')  # Título del gráfico
plt.legend()                   # Muestra la leyenda para identificar cada curva
plt.grid(True)                 # Agrega una cuadrícula al gráfico
plt.savefig("Taller 2 - Fourier/1.d.pdf", bbox_inches="tight", pad_inches=0.1)  # Guarda el gráfico en un archivo PDF
#plt.show()                    # Muestra el gráfico en pantalla (comentado para guardar solo el PDF)