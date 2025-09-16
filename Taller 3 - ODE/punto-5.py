import numpy as np
import matplotlib.pyplot as plt                     # para graficar 5.pdf
from scipy.integrate import solve_ivp               # integrador de ODE eficiente y robusto

# -------------------------------------------------------------------------
# Punto 5: Sistema de 3 genes con regulación negativa tipo "repressilator"
# d m_i/dt = alpha / (1 + p_{i-1}^n) + alpha0 - m_i
# d p_i/dt = - beta (p_i - m_i)
# con i en {1,2,3} y p_0 ≡ p_3 (índices cíclicos). Usar n=2, alpha0 = alpha/1000.
# Queremos la amplitud final (~después de 10 oscilaciones) de p3 para cada (alpha, beta).
# Se simula hasta t ≈ 400, y se grafica log10(amplitud) en un mapa de calor:
#   - eje x: alpha (log-scale)
#   - eje y: beta  (log-scale)
#   - color: log10(amplitud de p3)
# -------------------------------------------------------------------------

# ----------------------------
# 1) Parámetros de la simulación
# ----------------------------
N_ALPHA = 100                          # cantidad de muestras de alpha (log-espaciadas); aumentar si quieres más resolución
N_BETA  = 100                            # cantidad de muestras de beta  (log-espaciadas); aumentar si quieres más resolución

ALPHA_MIN, ALPHA_MAX = 1.0, 1e5           # rango de alpha
BETA_MIN,  BETA_MAX  = 1.0, 1e2           # rango de beta

T_END = 400.0                              # tiempo final de simulación
N_T   = 2001                               # cantidad de puntos de muestreo para evaluar soluciones (≈ dt 0.2)
t_eval = np.linspace(0.0, T_END, N_T)      # vector de tiempos para muestrear

n_hill = 2                                 # n = 2 según enunciado
amp_floor = 1e-12                          # piso para la amplitud (evitar log(0))
final_fraction = 0.25                      # fracción final de la señal para medir amplitud pico-a-pico

# ----------------------------
# 2) Construcción de grillas (alpha, beta) log-espaciadas
# ----------------------------
alphas = np.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), N_ALPHA)
betas  = np.logspace(np.log10(BETA_MIN),  np.log10(BETA_MAX),  N_BETA)

# matriz para guardar amplitudes de p3 (beta en filas, alpha en columnas)
amplitude = np.zeros((N_BETA, N_ALPHA), dtype=float)

# ----------------------------
# 3) Definición del RHS del sistema (6 EDOs: m1,m2,m3,p1,p2,p3)
# ----------------------------
def rhs_factory(alpha, beta):
    """
    Crea y devuelve una función f(t, y) que calcula las derivadas
    para un alpha y beta dados. Se fija n=2 y alpha0=alpha/1000.
    """
    alpha0 = alpha / 1000.0

    def f(t, y):
        # y = [m1, m2, m3, p1, p2, p3]
        m1, m2, m3, p1, p2, p3 = y

        # Índices cíclicos: p0 ≡ p3, p1, p2, p3 ya están
        p0 = p3

        # dm_i/dt = alpha/(1 + p_{i-1}^n) + alpha0 - m_i
        den1 = 1.0 + p0**n_hill
        den2 = 1.0 + p1**n_hill
        den3 = 1.0 + p2**n_hill

        dm1 = alpha / den1 + alpha0 - m1
        dm2 = alpha / den2 + alpha0 - m2
        dm3 = alpha / den3 + alpha0 - m3

        # dp_i/dt = - beta (p_i - m_i)
        dp1 = -beta * (p1 - m1)
        dp2 = -beta * (p2 - m2)
        dp3 = -beta * (p3 - m3)

        return np.array([dm1, dm2, dm3, dp1, dp2, dp3], dtype=float)

    return f

# ----------------------------
# 4) Función auxiliar para simular un par (alpha, beta) y medir amplitud
# ----------------------------
def simulate_and_amplitude(alpha, beta):
    """
    Integra el sistema hasta T_END y calcula la amplitud final de p3
    como el pico-a-pico en el último 'final_fraction' del tiempo.
    """
    f = rhs_factory(alpha, beta)

    # Condiciones iniciales: rompo simetría ligeramente para favorecer oscilaciones
    # y evitar estados puramente simétricos.
    y0 = np.array([1.0, 0.5, 0.1,   # m1, m2, m3
                   0.0, 0.0, 0.0],  # p1, p2, p3
                 dtype=float)

    # Integro con LSODA (maneja bien rigidez), tolerancias moderadas para ser eficiente
    sol = solve_ivp(
        f, (0.0, T_END), y0,
        method='LSODA',
        t_eval=t_eval,
        rtol=1e-4,
        atol=1e-6
    )

    if not sol.success:
        # Si falla, retorno amplitud mínima para que aparezca muy atenuado en el mapa
        return amp_floor

    # Tomo la serie temporal de p3
    p3 = sol.y[5, :]

    # Para estimar la "amplitud final", me enfoco en la parte final del registro
    n_last = max(10, int(final_fraction * p3.size))  # al menos 10 puntos
    p3_tail = p3[-n_last:]

    # Amplitud pico-a-pico simple (max - min) en el tramo final
    amp = float(np.max(p3_tail) - np.min(p3_tail))

    # Evito ceros exactos al tomar log más adelante
    if not np.isfinite(amp) or amp <= 0.0:
        amp = amp_floor

    return max(amp, amp_floor)

# ----------------------------
# 5) Bucle principal sobre (beta, alpha) y cálculo de amplitudes
# ----------------------------
for j, beta in enumerate(betas):
    # (opcional) se podría imprimir progreso por fila:
    # print(f"Fila beta {j+1}/{N_BETA} (beta={beta:.3g})")
    for i, alpha in enumerate(alphas):
        amplitude[j, i] = simulate_and_amplitude(alpha, beta)

# ----------------------------
# 6) Graficar mapa de calor de log10(amplitud) con pcolormesh
# ----------------------------
A_grid, B_grid = np.meshgrid(alphas, betas, indexing='xy')    # A: x, B: y
Z_log = np.log10(np.maximum(amplitude, amp_floor))            # color = log10(amplitud)

plt.figure(figsize=(8, 6))
pc = plt.pcolormesh(A_grid, B_grid, Z_log, shading='auto', cmap='viridis')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\beta$')
plt.title(r'log$_{10}$ amplitud final de $p_3$ (Punto 5)')
cbar = plt.colorbar(pc)
cbar.set_label(r'log$_{10}$(amplitud de $p_3$)')

plt.tight_layout()
plt.savefig("Taller 3 - ODE/5.pdf", dpi=300)
print("punto-5.pdf")
