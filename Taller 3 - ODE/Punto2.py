import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

m = 10.01      # kg
g = 9.773      # m/s^2
A = 1.642
B = 40.624     # m
D = 2.36
#____________________________________________Punto 2.a____________________________________________
def beta(y):
    return A*max(0.0,1-y/B) ** D

def derivada(t, s):
    # s = [x, y, vx, vy]
    return [s[2], s[3], 0, -g]

#alcance
def alcance(v0, theta):
    ang = np.deg2rad(theta)
    vx0 = v0 * np.cos(ang) 
    vy0 = v0 * np.sin(ang)
    s0 = np.array([0, 0, vx0, vy0])  # estado inicial: x, y, vx, vy


    def impact(t, s):
        return s[1] #(devuelve la altura y se detiene cuando es 0)

    impact.terminal = True
    impact.direction = -1

    solucion = solve_ivp(derivada, [0, 200], s0,
                    events=impact,
                    max_step=0.05,
                    rtol=1e-6, atol=1e-8)

    if solucion.status == 1 and len(solucion.t_events[0]) > 0:
        return solucion.y_events[0][0][0] #y_events[0] es el array con el primer evento de esta naturaleza en ocurrir, y_events[0][0] el evento puede repetirse segun la función, solo nos interesa el primer rebote, y_events[0][0][0] es la compoenente x de la vez que pasó eso
    else:
        return None
    

v0_vals = np.linspace(1, 140, 140)  # valores de velocidad inicial
angulos = np.linspace(10, 80, 100)

x_max = []
mejor_angulo = []

#Primero se recorren las Velociodades iniciales
for v0 in v0_vals:
    x_bueno = -1.0
    angulo_bueno = None
    #Luego se recorren los angulos
    for theta in angulos:
        x_piso = alcance(v0, theta)
        if x_piso is not None and x_piso > x_bueno:
            x_bueno = x_piso
            angulo_bueno = theta
    x_max.append(x_bueno)
    mejor_angulo.append(angulo_bueno)


plt.figure(figsize=(8,5))
plt.plot(v0_vals, x_max)
plt.xlabel("Velocidad inicial v0 (m/s)")
plt.ylabel("Alcance máximo x_max (m)")
plt.title("Alcance máximo vs velocidad inicial")
plt.savefig("Taller 3 - ODE/2.a.pdf")

#____________________________________________Punto 2.b____________________________________________
def angle_to_hit_target(v0, target_x, target_y):
    """
    Encuentra el ángulo (grados) para que el proyectil pase
    exactamente por (target_x, target_y).
    """

    def f(theta_deg):
        theta = np.deg2rad(theta_deg)
        vx0 = v0 * np.cos(theta)
        vy0 = v0 * np.sin(theta)

        # tiempo en que se alcanza target_x con esa velocidad horizontal
        t_x = target_x / vx0

        # y que tendría el proyectil en ese instante
        y_pred = vy0 * t_x - 0.5 * g * t_x**2

        return y_pred - target_y  # queremos que sea 0

    # verificamos que en [0,90] haya solución
    f0 = f(1.0)
    f1 = f(89.0)
    if f0 * f1 > 0:
        raise ValueError("No hay ángulo que pase por ese punto con esta velocidad")

    ang_sol = brentq(f, 1.0, 89.0)
    return ang_sol

#___________________________________________Punto 2.c____________________________________________

x_target = 12.0
y_target = 0.0

# rango de velocidades (desde el minimo físicamente posible)
v0_min = np.sqrt(g * x_target)
v0_max = 140.0
v0_vals = np.linspace(v0_min, v0_max, 600)

# calcular las dos ramas de ángulos (en rad)
s = (g * x_target) / (v0_vals**2)

# condición física: solo para s <= 1 hay soluciones reales
mask = s <= 1.0

theta1 = np.full_like(v0_vals, np.nan)
theta2 = np.full_like(v0_vals, np.nan)

theta1[mask] = 0.5 * np.arcsin(s[mask])
theta2[mask] = 0.5 * (np.pi - np.arcsin(s[mask]))

# pasar a grados
theta1_deg = np.degrees(theta1)
theta2_deg = np.degrees(theta2)

# plot
plt.figure(figsize=(8,5))
plt.plot(v0_vals, theta1_deg, label='Solución baja (θ1)', lw=2)
plt.plot(v0_vals, theta2_deg, label='Solución alta (θ2)', lw=2)
plt.axhline(80, color="k", label="Limite de angulo")
plt.axhline(10, color="green", label="Limite de angulo")
plt.axvline(v0_min, color='gray', linestyle='--', label=f'v0_min={v0_min:.2f} m/s')
plt.xlabel('Velocidad inicial $v_0$ (m/s)')
plt.ylabel('Ángulo $\\,\\theta$ (grados)')
plt.title(r'Ángulos $\theta_1,\theta_2$ que alcanzan $(x,y)=(12,0)$')
plt.xlim(v0_min-0.5, v0_max)
plt.ylim(9,81)
plt.legend()
plt.tight_layout()
plt.savefig("Taller 3 - ODE/2.c.pdf")