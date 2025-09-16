import numpy as np                                  # para cálculo numérico eficiente
import matplotlib.pyplot as plt                     # para graficar y guardar 6.pdf
from scipy.special import betainc                   # q(x) = betainc(3, 6, x)
from scipy.integrate import cumulative_trapezoid, trapezoid  # integrales numéricas precisas

# -------------------------------------------------------------------------
# Punto 6: Viga de Timoshenko–Ehrenfest (quasiestática) en [0, 1]
# Datos:
#   - E = 1, I = 1  (módulo elástico y 2do momento de inercia)
#   - A = 1, G = 1  (área transversal y módulo de cizalla)
#   - kappa = 5/6   (constante de la viga)
#   - q(x) = betainc(3, 6, x) (beta incompleta regularizada)
#
# Ecuaciones (del enunciado):
#   1) d^2/dx^2 (E I dφ/dx) = q(x)  => con E=I=1,  φ'''(x) = q(x)
#   2) dw/dx = φ - (1/(kappa*A*G)) d/dx (E I dφ/dx) => con E=I=A=G=1,  w' = φ - (1/kappa) φ''
#
# Condiciones de borde:
#   - Empotrada en x=0:  φ(0) = 0,  w(0) = 0
#   - Libre en x=1:      φ'(1) = 0 (momento nulo),  φ''(1) = 0 (cortante nula)
#
# Idea SIMPLE y OPTIMIZADA:
#   - Como φ''' = q(x), imponemos los BCs en x=1 para obtener a = φ'(0) y b = φ''(0):
#       b = - ∫_0^1 q(s) ds
#       a =   ∫_0^1 s q(s) ds
#   - Con a, b y q(x) reconstruimos φ'', φ', φ y w' por integraciones acumuladas (trapezoidal).
#   - Finalmente graficamos la viga antes y después con u_x = -y φ, u_y = w.
# -------------------------------------------------------------------------

# ----------------------------
# 1) Parámetros y malla numérica
# ----------------------------
kappa = 5.0 / 6.0                     # constante de la viga (del enunciado)
L = 1.0                               # largo de la viga
y_top = 0.2                           # cara superior de la viga
y_bot = -0.2                          # cara inferior de la viga

N = 2001                              # número de puntos en la malla (suficiente y rápido)
x = np.linspace(0.0, L, N)            # malla a lo largo de la viga

# ----------------------------
# 2) Definimos la carga distribuida q(x)
# ----------------------------
q = betainc(3.0, 6.0, x)              # q(x) = I_x(3, 6) (beta incompleta regularizada)

# ----------------------------
# 3) Calculamos a y b que satisfacen los BCs libres en x=1
#    b = φ''(0) = - ∫_0^1 q(s) ds
#    a = φ'(0)  =   ∫_0^1 s q(s) ds
#    (usamos misma malla x para consistencia numérica con las integraciones posteriores)
# ----------------------------
Q1 = trapezoid(q, x)                  # ∫_0^1 q(s) ds
Qs = trapezoid(x * q, x)              # ∫_0^1 s q(s) ds

b = -Q1                               # φ''(0)
a = Qs                                # φ'(0)

# ----------------------------
# 4) Reconstruimos φ'', φ', φ por integraciones acumuladas
#    - F(x)   = ∫_0^x q(s) ds
#    - ∫_0^x F(t) dt nos da el término doble integral para φ'
# ----------------------------
F = cumulative_trapezoid(q, x, initial=0.0)          # F(x) = ∫_0^x q
phi_dd = b + F                                        # φ''(x) = b + ∫_0^x q
IntF = cumulative_trapezoid(F, x, initial=0.0)        # ∫_0^x F(t) dt
phi_d = a + b * x + IntF                              # φ'(x) = a + b x + ∫_0^x F
phi = cumulative_trapezoid(phi_d, x, initial=0.0)     # φ(x) = ∫_0^x φ'(t) dt  con φ(0)=0

# ----------------------------
# 5) Calculamos w'(x) y w(x) usando w' = φ - (1/kappa) φ''
# ----------------------------
w_prime = phi - (1.0 / kappa) * phi_dd
w = cumulative_trapezoid(w_prime, x, initial=0.0)      # w(0)=0

# ----------------------------
# 6) Construimos las coordenadas antes y después (deformación de Timoshenko)
#    u_x(x,y) = - y φ(x),   u_y(x,y) = w(x)
#    Cara superior y=+0.2: X = x - 0.2 φ(x),  Y = 0.2 + w(x)
#    Cara inferior y=-0.2: X = x + 0.2 φ(x),  Y = -0.2 + w(x)
# ----------------------------
X_top_undeformed = x
Y_top_undeformed = np.full_like(x, y_top)
X_bot_undeformed = x
Y_bot_undeformed = np.full_like(x, y_bot)

X_top_deformed = x - y_top * phi
Y_top_deformed = y_top + w
X_bot_deformed = x - y_bot * phi   # ojo: y_bot = -0.2 ⇒ X = x - (-0.2)*φ = x + 0.2 φ
Y_bot_deformed = y_bot + w

# ----------------------------
# 7) Graficamos viga antes y después y guardamos 6.pdf
# ----------------------------
plt.figure(figsize=(7, 3.5))

# Viga original (rectángulo)
plt.plot(X_top_undeformed, Y_top_undeformed, 'k--', lw=1.2, label='Viga original (superior)')
plt.plot(X_bot_undeformed, Y_bot_undeformed, 'k--', lw=1.2, label='Viga original (inferior)')
# Unimos bordes para visualizar mejor el contorno original
plt.plot([0, 0], [y_bot, y_top], 'k--', lw=1.0)
plt.plot([L, L], [y_bot, y_top], 'k--', lw=1.0)

# Viga deformada
plt.plot(X_top_deformed, Y_top_deformed, 'b', lw=2.0, label='Viga deformada (superior)')
plt.plot(X_bot_deformed, Y_bot_deformed, 'r', lw=2.0, label='Viga deformada (inferior)')
# Unimos bordes deformados (aprox. líneas entre extremos)
plt.plot([X_top_deformed[0], X_bot_deformed[0]], [Y_top_deformed[0], Y_bot_deformed[0]], 'g-', lw=1.0)
plt.plot([X_top_deformed[-1], X_bot_deformed[-1]], [Y_top_deformed[-1], Y_bot_deformed[-1]], 'g-', lw=1.0)

plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Viga de Timoshenko–Ehrenfest: antes y después (6.pdf)')
plt.legend(loc='best')
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig('6.pdf', dpi=300)

print("Generado: 6.pdf")