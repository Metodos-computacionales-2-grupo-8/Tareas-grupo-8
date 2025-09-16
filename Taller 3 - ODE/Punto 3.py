import numpy as np                                      
import matplotlib.pyplot as plt                         
from scipy.signal import argrelextrema                  # importa herramienta para detectar mínimos locales  
from scipy.optimize import minimize_scalar              # importa optimizador para refinar energías (bracketing)  
from scipy.integrate import simpson as simps                        # importa integrador numérico para normalización  

# Parámetros físicos y del potencial -----------------------------------------
hbar = 0.1                                             # constante reducida (unidades dadas en el enunciado)  
a = 0.8                                                # parámetro a del potencial de Morse  
x0 = 10.0                                              # posición del mínimo del pozo de Morse  

# Potencial de Morse corregido (exponente positivo) --------------------------------
def V(x):                                              # función potencial de Morse en x (acepta arrays)  
    # Uso del exponente con signo POSITIVO aquí intencionalmente:
    # exp(a*(x-x0)) -> 0 para x << x0 (izquierda) y -> +inf para x >> x0 (derecha).
    # Esto sitúa el mínimo en x0 y hace que el pozo tenga asintota 0 hacia la izquierda,
    # reproduciendo la forma mostrada en la imagen de referencia.
    exp_term = np.exp(a * (x - x0))                    # término exponencial con signo positivo  
    return (1.0 - exp_term)**2 - 1.0                    # V(x) = (1 - e^{a(x-x0)})^2 - 1 (orientación corregida)  

# Sistema de ecuaciones de primer orden para Schrödinger -----------------------
# y = [psi, phi] con psi' = phi, phi' = (V(x)-eps)/hbar^2 * psi
def derivs(x, y, eps):                                 # devuelve derivadas en x para estado y y energía eps  
    psi, phi = y                                       # desempaqueta valores actuales de psi y phi  
    dpsi = phi                                         # psi' = phi  
    dphi = (V(x) - eps) * psi / (hbar**2)             # phi' = (V - eps)/hbar^2 * psi  
    return np.array([dpsi, dphi], dtype=float)        # retorna vector derivadas  

# Integrador RK4 para el sistema de primer orden --------------------------------
def integrate_rk4(eps, x_start, x_end, dx):            # integra desde x_start hasta x_end con paso dx  
    n_steps = int(np.ceil(abs(x_end - x_start) / dx))  # número de pasos redondeado hacia arriba  
    xs = x_start + np.arange(n_steps + 1) * np.sign(x_end - x_start) * dx  # vector de nodos  
    ys = np.zeros((n_steps + 1, 2), dtype=float)       # arreglo para psi y phi en cada nodo  
    ys[0, 0] = 0.0                                     # condición inicial psi(x_start)=0 (sugerido)  
    ys[0, 1] = 1e-6                                    # condición inicial psi'(x_start) pequeña no nula  
    for i in range(n_steps):                           # bucle de integración RK4  
        x_i = xs[i]                                    # posición actual  
        y_i = ys[i].copy()                             # estado actual [psi,phi]  
        k1 = derivs(x_i, y_i, eps)                     # k1 para RK4  
        k2 = derivs(x_i + 0.5 * dx * np.sign(x_end - x_start), y_i + 0.5 * dx * k1 * np.sign(x_end - x_start), eps)  # k2  
        k3 = derivs(x_i + 0.5 * dx * np.sign(x_end - x_start), y_i + 0.5 * dx * k2 * np.sign(x_end - x_start), eps)  # k3  
        k4 = derivs(x_i + dx * np.sign(x_end - x_start), y_i + dx * k3 * np.sign(x_end - x_start), eps)  # k4  
        ys[i + 1] = y_i + (dx * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0) * np.sign(x_end - x_start)  # actualización RK4  
    return xs, ys[:, 0], ys[:, 1]                     # retorna xs, psi(xs) y phi(xs)  

# Encontrar puntos de giro V(x)=eps por búsqueda de cambios de signo ----------------
def find_turning_points(eps, x_min=0.0, x_max=20.0, dx_search=0.001):  # busca x1,x2 tales que V(x)=eps  
    xs = np.arange(x_min, x_max + dx_search, dx_search)                # malla fina para detectar raíces  
    vals = V(xs) - eps                                                 # valores de V-eps en la malla  
    signs = np.sign(vals)                                              # signo de cada punto  
    changes = np.where(np.diff(signs) != 0)[0]                         # índices donde cambia el signo => raíz entre nodos  
    if len(changes) >= 2:                                               # si hay al menos dos cruces  
        # localizar de forma más precisa con interpolación lineal simple  
        x_left_idx = changes[0]                                        # índice del primer cambio de signo  
        x_right_idx = changes[-1]                                      # índice del último cambio de signo  
        # interpolación lineal para estimar posiciones de giro  
        x1 = xs[x_left_idx] - vals[x_left_idx] * (xs[x_left_idx+1]-xs[x_left_idx])/(vals[x_left_idx+1]-vals[x_left_idx])   
        x2 = xs[x_right_idx] - vals[x_right_idx] * (xs[x_right_idx+1]-xs[x_right_idx])/(vals[x_right_idx+1]-vals[x_right_idx])   
        return max(x_min, x1), min(x_max, x2)                         # retorna x1,x2 acotados al dominio  
    else:
        return x_min, x_max                                             # si no se detectan, usar extremos sugeridos  

# Función objetivo usada en shooting: norma del vector estado al final ----------------
def endpoint_norm(eps, x_start, x_end, dx):                  # calcula norma final para una energía eps  
    xs, psi, phi = integrate_rk4(eps, x_start, x_end, dx)    # integra ecuación de Schrödinger  
    return np.sqrt(psi[-1]**2 + phi[-1]**2)                 # retorna norma en el extremo derecho (residual)  

# Búsqueda de energías mediante barrido y detección de mínimos locales -------------------
# parámetros numéricos del barrido
dx = 0.005                                                  # paso máximo sugerido por el enunciado (<=0.01)  
eps_min, eps_max = -1.0, 0.0                                # intervalo de energías a explorar (pozo entre -1 y 0)  
N_eps = 400                                                 # número de energías para barrido inicial (malla fina)  
eps_grid = np.linspace(eps_min, eps_max, N_eps)             # rejilla de energías para el barrido inicial  

norms = np.zeros_like(eps_grid)                             # vector para almacenar la norma al final para cada eps  
turns_cache = [None] * len(eps_grid)                        # cache de puntos de giro usados por cada eps  

for i, eps in enumerate(eps_grid):                          # bucle sobre energías de prueba  
    x1, x2 = find_turning_points(eps)                       # encuentra puntos de giro V(x)=eps  
    x_start = x1 - 2.0                                      # simular desde x1-2 según el enunciado  
    x_end = x2 + 1.0                                        # hasta x2+1 según el enunciado  
    turns_cache[i] = (x_start, x_end)                       # guarda para reutilizar soluciones después  
    norms[i] = endpoint_norm(eps, x_start, x_end, dx)       # calcula y almacena la norma final  

# detectar mínimos locales en el vector norms ----------------------------------------
min_idx = argrelextrema(norms, np.less, order=2)[0]         # índices de mínimos locales en el barrido inicial  

# refinar cada mínimo con un optimizador unidimensional (bracket alrededor del mínimo) -
refined_energies = []                                       # lista para energías refinadas  
refined_solutions = []                                      # lista para soluciones (xs,psi,phi)  

for idx in min_idx:                                         # para cada índice de mínimo local  
    eps0 = eps_grid[idx]                                    # energía inicial para refinamiento  
    # definir extremo izquierdo y derecho para el bracketing (usando vecinos del grid)  
    left = eps_grid[max(idx - 3, 0)]                        # energía izquierda para bracketing  
    right = eps_grid[min(idx + 3, len(eps_grid) - 1)]       # energía derecha para bracketing  
    x_start, x_end = turns_cache[idx]                       # recupera los límites de integración usados en el barrido  

    res = minimize_scalar(lambda e: endpoint_norm(e, x_start, x_end, dx), bounds=(left, right), method='bounded', options={'xatol':1e-7})  # refina energía por minimización de la norma final  
    eps_ref = res.x                                         # energía refinada (mínimo local)  
    refined_energies.append(eps_ref)                        # guarda energía refinada  

    # integra nuevamente con la energía refinada para obtener la función de onda completa  
    xs, psi, phi = integrate_rk4(eps_ref, x_start, x_end, dx)  # integra con eps refinada  

    # normalizar la función de onda en norma L2 (integral psi^2 dx = 1)  
    # se usa simpson/trapecio numérico para calcular norma y luego se normaliza  
    norm_psi = np.sqrt(simps(psi**2, xs))                   # calcula norma L2 de psi antes de escalar  
    if norm_psi > 0:                                        # evita división por cero  
        psi = psi / norm_psi                                # normaliza psi  

    refined_solutions.append((xs, psi, phi))                # guarda la solución normalizada  


# ordenar niveles numéricos por energía (ascendente) para emparejarlos correctamente
sorted_idx = np.argsort(refined_energies)                              # índices que ordenan refined_energies
refined_energies = [refined_energies[i] for i in sorted_idx]           # energies ordenadas
refined_solutions = [refined_solutions[i] for i in sorted_idx]         # soluciones reordenadas acorde

# calcular energías teóricas corregidas (la fórmula original da eps_raw; el pozo está centrado en -1)
lam = 1.0 / (a * hbar)                                                 # lambda = 1/(a*hbar)
eps_teo_list = []                                                      # lista de energías teóricas corregidas
n = 0
while True:
    eps_raw = (2.0 * lam - n - 0.5) * (n + 0.5) / (lam**2)              # expresión usada originalmente
    eps_t = eps_raw - 1.0                                              # CORRECCIÓN: desplazar por -1 para el pozo usado
    # conservar sólo estados ligados dentro del pozo (-1 < eps < 0)
    if (eps_t < 0.0) and (eps_t > -1.0):
        eps_teo_list.append(eps_t)                                     # guardar energía teórica válida
        n += 1
    else:
        break

# escribir archivo 3.txt emparejando niveles numéricos y teóricos por índice
out_fname = "Taller 3 - ODE/3.txt"
with open(out_fname, "w") as f:
    f.write("n\tE_numerico\tE_teorico\tDiferencia(%)\n")
    N_pairs = min(len(refined_energies), len(eps_teo_list))            # cuantos pares podemos emparejar
    for i in range(N_pairs):
        e_num = refined_energies[i]
        e_teo = eps_teo_list[i]
        diffpct = 100.0 * abs(e_num - e_teo) / (abs(e_teo) if e_teo != 0 else 1.0)
        f.write(f"{i}\t{e_num:.8f}\t{e_teo:.8f}\t{diffpct:.4f}\n")
    # si hay niveles numéricos extra sin teórico, listarlos indicando N/A
    for j in range(N_pairs, len(refined_energies)):
        e_num = refined_energies[j]
        f.write(f"{j}\t{e_num:.8f}\tN/A\tN/A\n")




# Graficar potencial y funciones de onda normalizadas (interpoladas en todo x_plot)
plt.figure(figsize=(8, 8))                                  # crea figura cuadrada para la gráfica final
x_plot = np.linspace(0.0, 20.0, 2000)                       # dominio para dibujar potencial y ondas
plt.plot(x_plot, V(x_plot), 'k', label='Morse potential')   # dibuja el potencial (aparecerá en la leyenda)

amplitude_scale = 0.05                                      # escala de amplitud visual para las ondas
mask_inside = V(x_plot) < 0.0                               # máscara de la región "dentro" del pozo (V<0)
threshold_inside = 0.01                                     # umbral mínimo de amplitud dentro del pozo para aceptar un nivel

n_levels = len(refined_solutions)                           # número de soluciones detectadas
colors = plt.cm.viridis(np.linspace(0, 1, max(1, n_levels)))# paleta de colores

kept_count = 0                                              # contador de niveles mantenidos (para elección de color si se desea)
for i, (xs, psi, phi) in enumerate(refined_solutions):
    # interpolo la psi sobre toda la malla de trazado; fuera de xs se fija en 0
    psi_interp = np.interp(x_plot, xs, psi, left=0.0, right=0.0)

    # mido la amplitud de la onda dentro del pozo (donde importa físicamente)
    if np.any(mask_inside):
        max_abs_inside = np.max(np.abs(psi_interp[mask_inside]))
    else:
        max_abs_inside = 0.0

    # descartar niveles cuya amplitud dentro del pozo es insignificante (parecen "constantes")
    if max_abs_inside <= threshold_inside:
        continue  # omito graficar este nivel

    # si pasa el filtro, escalo la onda para visualización y la dibujo en toda la gráfica
    global_max = np.max(np.abs(psi_interp))
    if global_max == 0:
        psi_plot = psi_interp * 0.0
    else:
        psi_plot = (psi_interp / global_max) * amplitude_scale

    # uso color consistente según el índice original (se puede cambiar a kept_count si prefieres colores consecutivos)
    plt.plot(x_plot, psi_plot + refined_energies[i], color=colors[i % len(colors)], lw=1.2)
    kept_count += 1

# etiquetas y formato final
plt.xlabel("x")
plt.ylabel("Energy")
plt.xlim(0, 11.5)
plt.ylim(-1.1, 0.1)
plt.legend(loc='upper right')   # mostrará solo la entrada 'Morse potential'
plt.tight_layout()
plt.savefig("Taller 3 - ODE/3.pdf", dpi=300)
#plt.show()
