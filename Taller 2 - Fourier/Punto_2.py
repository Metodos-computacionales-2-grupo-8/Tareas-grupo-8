import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
#lectura de datos
data = pd.read_csv('Taller 2 - Fourier/Datos/SN_d_tot_V2.0.csv')
dias_sin_manchas = (data['spots']==-1).sum()


#datos interpolados para el punto 2.a
fecha_decimal = np.array(data['decimal_date'])
manchas = np.array(data['spots'])

mask_1 = manchas != -1 

fecha_decimal_sin_falla = fecha_decimal[mask_1] # el xp para el np.interp
manchas_sin_falla = manchas[mask_1]             # el fp para el np.interp

manchas_interpoladas = np.interp(fecha_decimal, fecha_decimal_sin_falla ,manchas_sin_falla )

datos_fft = np.fft.fft(manchas_interpoladas)
frecuencias = np.fft.fftfreq(len(manchas_interpoladas), d=1)
promedio_manchas = datos_fft[0]/len(manchas_interpoladas)
maximo = np.argmax(np.abs(datos_fft[1:len(datos_fft)//2])) + 1 #+1 porque se quito el primer elemento
frecuencia_maximo = frecuencias[maximo]
periodo = 1/frecuencia_maximo
print(f'La frecuencia del maximo es: {frecuencia_maximo}, y el periodo es: {periodo} días, lo que es equivalente a {periodo/365} años')

with open('Taller 2 - Fourier/2.b.txt', 'w') as f:
    f.write(f'{round(periodo/365, 3)}')

#la frecuencia del maximo es: 0.00025060012134321664
fc = frecuencia_maximo * 1.5
frecuencias_filtradas = np.where(np.abs(frecuencias) > fc, 0, datos_fft)
manchas_filtradas = np.fft.ifft(frecuencias_filtradas)
plt.plot(fecha_decimal, manchas_interpoladas, label='Datos interpolados', color='gray', alpha=0.5)
plt.plot(fecha_decimal, manchas_filtradas.real, label='Datos filtrados', color='purple')
plt.title(f'Resultado del filtro pasabajas con fc={fc} Hz')
plt.legend()
plt.savefig('Taller 2 - Fourier/2.b.data.pdf', bbox_inches='tight')


picos = []
fechas = []

for i in range(len(manchas_filtradas)):
    if manchas_filtradas[i-1] < manchas_filtradas[i] and manchas_filtradas[i+1] < manchas_filtradas[i]:
        picos.append(manchas_filtradas[i].real)
        fechas.append(fecha_decimal[i])

picos = np.array(picos)
fechas = np.array(fechas)
#plt.plot(fecha_decimal, manchas_filtradas.real, label='Datos filtrados', color='purple')
plt.figure()
plt.scatter(fechas, picos, color='red', label='Picos')
plt.legend()
plt.savefig('Taller 2 - Fourier/2.b.maxima.pdf', bbox_inches='tight')