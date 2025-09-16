# # Script que resuelve los incisos 1.a (Lotka-Volterra), 1.b (Problema de Landau) y 1.c (Sistema binario)
# # Cada sección está separada por comentarios con hashtags para facilitar su uso.
# # Se generan y guardan 1.a.pdf, 1.b.pdf y 1.c.pdf con los subplots pedidos.
# # Importo numpy para cálculos numéricos.
import numpy as np
# # Importo matplotlib para graficar y guardar las figuras.
import matplotlib.pyplot as plt
# # Importo solve_ivp de scipy para integrar sistemas ODE (usado en Landau).
from scipy.integrate import solve_ivp

# ############################
# # 1.a Sistema depredador-presa (Lotka-Volterra)
# ############################
# # Defino una función que implementa las ecuaciones de Lotka-Volterra.
def lotka_volterra(t, y, alpha, beta, gamma, delta):
    # # y[0] = x (presa), y[1] = y (depredador).
    # # dx/dt = alpha*x - beta*x*y
    dxdt = alpha * y[0] - beta * y[0] * y[1]
    # # dy/dt = -gamma*y + delta*x*y
    dydt = -gamma * y[1] + delta * y[0] * y[1]
    # # Devuelvo derivadas como vector.
    return np.array([dxdt, dydt])

# # Parámetros y condiciones iniciales tal como en el enunciado.
alpha = 2.0
beta = 1.5
gamma = 0.3
delta = 0.4
# # Condiciones iniciales: x0 = 3 (presas), y0 = 2 (depredadores).
y0 = np.array([3.0, 2.0])
# # Tiempo de simulación y paso (usamos RK4 explícito con paso fijo pequeño para conservar la cantidad V).
t_max_a = 50.0
dt_a = 0.01
# # Array de tiempos.
t_arr_a = np.arange(0.0, t_max_a + dt_a, dt_a)
# # Arrays para almacenar solución.
sol_a = np.zeros((len(t_arr_a), 2))
# # Inicializo la solución.
sol_a[0, :] = y0

# # Método RK4 explícito (paso fijo) para mayor conservación numérica de la cantidad V.
for i in range(len(t_arr_a) - 1):
    # # Estado actual.
    yi = sol_a[i, :]
    # # Evaluación k1.
    k1 = lotka_volterra(t_arr_a[i], yi, alpha, beta, gamma, delta)
    # # Evaluación k2.
    k2 = lotka_volterra(t_arr_a[i] + 0.5 * dt_a, yi + 0.5 * dt_a * k1, alpha, beta, gamma, delta)
    # # Evaluación k3.
    k3 = lotka_volterra(t_arr_a[i] + 0.5 * dt_a, yi + 0.5 * dt_a * k2, alpha, beta, gamma, delta)
    # # Evaluación k4.
    k4 = lotka_volterra(t_arr_a[i] + dt_a, yi + dt_a * k3, alpha, beta, gamma, delta)
    # # Actualizo la solución con la combinación RK4.
    sol_a[i + 1, :] = yi + (dt_a / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# # Calculo la cantidad conservada V(t) = delta*x - gamma*ln x + beta*y - alpha*ln y
# # Evito logs de cero usando un pequeño epsilon.
eps = 1e-12
x_arr = sol_a[:, 0]
y_arr = sol_a[:, 1]
V_arr = delta * x_arr - gamma * np.log(x_arr) + beta * y_arr - alpha * np.log(y_arr)

# # Grafico y guardo 1.a.pdf con subplots: x(t), y(t) y V(t).
fig_a, axs_a = plt.subplots(3, 1, figsize=(6, 10))
# # Grafico x(t).
axs_a[0].plot(t_arr_a, x_arr, '-b')
# # Etiquetas y título.
axs_a[0].set_ylabel('x (presa)')
axs_a[0].set_title('x(t), y(t) y cantidad conservada V(t)')
# # Grafico y(t).
axs_a[1].plot(t_arr_a, y_arr, '-r')
# # Etiquetas.
axs_a[1].set_ylabel('y (depredador)')
# # Grafico V(t).
axs_a[2].plot(t_arr_a, V_arr, '-k')
# # Etiquetas.
axs_a[2].set_ylabel('V(t)')
axs_a[2].set_xlabel('t')
# # Ajusto layout y guardo PDF.
fig_a.tight_layout()
fig_a.savefig('Taller 3 - ODE/1.a.pdf', dpi=200)


# ############################
# # 1.b Problema de Landau (bloque reemplazado según la imagen corregida)
# ############################
# # Defino la constante de la velocidad de la luz en unidades naturales.
c = 1.0
# # Defino la carga de la partícula.
q = 7.5284
# # Campo magnético constante B0 en la dirección z.
B0 = 0.438
# # Amplitud del campo eléctrico espacialmente variable.
E0 = 0.7423
# # Masa de la partícula.
m = 3.8428
# # Constante de onda espacial k.
k = 1.0014

# # Defino el lado derecho del sistema de ecuaciones como función para el integrador.
# # El sistema de primer orden usa el vector state = [x, y, vx, vy].
def landau_rhs(t, state):
    # # Extraigo posiciones y velocidades del vector de estado.
    x = state[0]
    y = state[1]
    vx = state[2]
    vy = state[3]
    # # Campo eléctrico espacial Ex(x) = E0*(sin(k x) + k x cos(k x)) según la ecuación corregida.
    Ex = E0 * (np.sin(k * x) + k * x * np.cos(k * x))
    # # Ecuación para la aceleración en x: m ẍ = q Ex - (q B0 / c) ẏ -> ẍ = (q/m)*Ex - (q B0/(m c))*vy
    ax = (q / m) * Ex - (q * B0 / (m * c)) * vy
    # # Ecuación para la aceleración en y: m ÿ = (q B0 / c) ẋ -> ÿ = (q B0/(m c)) * vx
    ay = (q * B0 / (m * c)) * vx
    # # Devuelvo las derivadas [dx/dt, dy/dt, dvx/dt, dvy/dt].
    return np.array([vx, vy, ax, ay])

# # Condiciones iniciales (ajustables; si el enunciado trae otras, cámbialas aquí).
# # Posición inicial en x.
x0_b = 0.1
# # Posición inicial en y.
y0_b = 0.0
# # Velocidad inicial en x.
vx0_b = 0.0
# # Velocidad inicial en y.
vy0_b = 0.0
# # Vector de condiciones iniciales para el integrador.
state0_b = np.array([x0_b, y0_b, vx0_b, vy0_b])

# # Intervalo de tiempo de integración y puntos de evaluación uniformes.
t_span_b = (0.0, 30.0)
# # Tiempo de salida deseado (muestreo denso para trazar).
t_eval_b = np.linspace(t_span_b[0], t_span_b[1], 10001)

# # Integro con solve_ivp (RK45) usando tolerancias estrictas para buena precisión de las cantidades conservadas.
sol_b_ivp = solve_ivp(landau_rhs, t_span_b, state0_b, t_eval=t_eval_b, rtol=1e-9, atol=1e-12)

# # Extraigo arrays temporales y de estado de la solución devuelta por solve_ivp.
t_arr_b = sol_b_ivp.t
x_b = sol_b_ivp.y[0, :]
y_b = sol_b_ivp.y[1, :]
vx_b = sol_b_ivp.y[2, :]
vy_b = sol_b_ivp.y[3, :]

# # Calculo el momento conjugado Pi_y = m * ẏ - (q B0 / c) * x (cantidad conservada según el enunciado).
Pi_y = m * vy_b - (q * B0 / c) * x_b

# # Calculo la energía total K + U con K = (m/2)(vx^2 + vy^2) y U = - q E0 x sin(k x).
K_b = 0.5 * m * (vx_b ** 2 + vy_b ** 2)
U_b = - q * E0 * x_b * np.sin(k * x_b)
E_total_b = K_b + U_b

# # Gráficas: posiciones x(t), y(t), momento conjugado Pi_y(t) y energía total E(t).
fig_b, axs_b = plt.subplots(4, 1, figsize=(6, 12))
# # Grafico x(t).
axs_b[0].plot(t_arr_b, x_b, '-b')
# # Etiqueta eje vertical.
axs_b[0].set_ylabel('x(t)')
# # Título del primer subplot.
axs_b[0].set_title('posiciones, momento conjugado y energía')
# # Grafico y(t).
axs_b[1].plot(t_arr_b, y_b, '-g')
# # Etiqueta eje vertical.
axs_b[1].set_ylabel('y(t)')
# # Grafico Pi_y(t).
axs_b[2].plot(t_arr_b, Pi_y, '-m')
#axs_b[2].set_ylim(np.min(Pi_y) * 3, np.max(Pi_y) * 3)
# # Etiqueta eje vertical.
axs_b[2].set_ylabel('Pi_y(t)')

# # Grafico energía total.
axs_b[3].plot(t_arr_b, E_total_b, '-k')
# # Etiqueta eje vertical.
axs_b[3].set_ylabel('E total')
# # Etiqueta eje horizontal del último subplot.
axs_b[3].set_xlabel('t')
# # Ajusto layout para evitar solapamiento y guardo la figura en PDF.
fig_b.tight_layout()
fig_b.savefig('Taller 3 - ODE/1.b.pdf', dpi=200)


# ############################
# # 1.c Sistema binario (dos masas iguales m)
# ############################
# # Parámetros: G=1, ambas masas m=1.7, posiciones y velocidades iniciales dadas.
G = 1.0
m_star = 1.7
# # Posiciones iniciales r1=(0,0), r2=(1,1)
r1_0 = np.array([0.0, 0.0])
r2_0 = np.array([1.0, 1.0])
# # Velocidades iniciales v1=(0,0.5), v2=(0,-0.5)
v1_0 = np.array([0.0, 0.5])
v2_0 = np.array([0.0, -0.5])

# # Integrador symplectic: velocity-Verlet para interacción gravitacional entre dos partículas.
t_max_c = 10.0
dt_c = 1e-3
t_arr_c = np.arange(0.0, t_max_c + dt_c, dt_c)
# # Inicializo arrays para posiciones y velocidades.
r1 = np.zeros((len(t_arr_c), 2))
r2 = np.zeros((len(t_arr_c), 2))
v1 = np.zeros((len(t_arr_c), 2))
v2 = np.zeros((len(t_arr_c), 2))
# # Condiciones iniciales.
r1[0, :] = r1_0
r2[0, :] = r2_0
v1[0, :] = v1_0
v2[0, :] = v2_0

# # Función que calcula las aceleraciones debidas a la gravedad mutua.
def accel_grav(r1_vec, r2_vec):
    # # Vector relativo r1 - r2
    r12 = r1_vec - r2_vec
    # # Distancia entre cuerpos.
    dist = np.linalg.norm(r12)
    # # Evito división por cero si se acercan demasiado.
    if dist < 1e-12:
        return np.zeros(2), np.zeros(2)
    # # Fuerza sobre 1 por 2: -G m1 m2 * (r1 - r2)/|r1-r2|^3 ; aceleración a1 = F1/m1 = -G m2 * (r1-r2)/|r|^3
    a1 = - G * m_star * r12 / dist ** 3
    # # Para la otra masa, a2 = -G m1 * (r2 - r1)/|r|^3 = -a1 (simétrico con masas iguales).
    a2 = -a1
    # # Devuelvo aceleraciones sobre cada cuerpo.
    return a1, a2

# # Primeros aceleraciones iniciales.
a1_curr, a2_curr = accel_grav(r1[0, :], r2[0, :])

# # Iteración velocity-Verlet.
for i in range(len(t_arr_c) - 1):
    # # Actualizo posiciones con paso de Verlet: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
    r1[i + 1, :] = r1[i, :] + v1[i, :] * dt_c + 0.5 * a1_curr * dt_c ** 2
    r2[i + 1, :] = r2[i, :] + v2[i, :] * dt_c + 0.5 * a2_curr * dt_c ** 2
    # # Calculo nuevas aceleraciones en posiciones actualizadas.
    a1_next, a2_next = accel_grav(r1[i + 1, :], r2[i + 1, :])
    # # Actualizo velocidades: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
    v1[i + 1, :] = v1[i, :] + 0.5 * (a1_curr + a1_next) * dt_c
    v2[i + 1, :] = v2[i, :] + 0.5 * (a2_curr + a2_next) * dt_c
    # # Paso a la siguiente aceleración.
    a1_curr, a2_curr = a1_next, a2_next

# # Calculo cantidades conservadas: energía total y momento angular total.
# # Energía cinética total K = 0.5*m*(|v1|^2 + |v2|^2)
K_c = 0.5 * m_star * (np.sum(v1 ** 2, axis=1) + np.sum(v2 ** 2, axis=1))
# # Energía potencial U = -G * m1 * m2 / |r1 - r2|
rel_vecs = r1 - r2
distances = np.linalg.norm(rel_vecs, axis=1)
U_c = - G * m_star * m_star / np.maximum(distances, 1e-12)
E_total_c = K_c + U_c
# # Momento angular total (z-component) L_z = sum_i m * (x_i * v_y_i - y_i * v_x_i)
Lz_c = m_star * (r1[:, 0] * v1[:, 1] - r1[:, 1] * v1[:, 0]) + m_star * (r2[:, 0] * v2[:, 1] - r2[:, 1] * v2[:, 0])

# # Grafico y guardo 1.c.pdf: posiciones (trayectorias) y las cantidades conservadas en función del tiempo.
fig_c, axs_c = plt.subplots(3, 1, figsize=(7, 12))
# # Subplot 1: trayectorias en el plano.
axs_c[0].plot(r1[:, 0], r1[:, 1], '-b', label='estrella 1')
axs_c[0].plot(r2[:, 0], r2[:, 1], '-r', label='estrella 2')
axs_c[0].set_xlabel('x')
axs_c[0].set_ylabel('y')
axs_c[0].set_title('trayectorias y cantidades conservadas')
axs_c[0].legend()
# # Subplot 2: energía total.
axs_c[1].plot(t_arr_c, E_total_c, '-k')
axs_c[1].set_ylabel('E total')
# # Subplot 3: momento angular L_z
axs_c[2].plot(t_arr_c, Lz_c, '-m')
axs_c[2].set_ylabel('L_z')
axs_c[2].set_xlabel('t')
fig_c.tight_layout()
fig_c.savefig('Taller 3 - ODE/1.c.pdf', dpi=200)

# # Impresión breve indicando que los archivos se generaron (útil en el terminal integrado de VS Code).
print("Generados: 1.a.pdf, 1.b.pdf, 1.c.pdf")