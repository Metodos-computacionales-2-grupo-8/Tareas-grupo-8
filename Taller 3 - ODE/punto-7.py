import numpy as np                                        # cálculos numéricos eficientes
from scipy.integrate import solve_ivp                      # integrador ODE con eventos
import csv                                                 # para exportar 7.csv

# -------------------------------------------------------------------------
# Punto 7: Ecuación de Lane–Emden
#    (1/x^2) d/dx ( x^2 dθ/dx ) = - θ^n
# Condiciones iniciales en x=0: θ(0)=1, θ'(0)=0.
# El radio de la estrella es x* tal que θ(x*)=0 (primer cero).
# Masa total (proporcional):  M ∝ - x*^2 θ'(x*).
# Relación densidad central a densidad promedio: ρc/⟨ρ⟩ = - x* / (3 θ'(x*)).
#
# Nota numérica: la ecuación es singular en x=0 por el término (2/x)θ'.
# Para evitarlo, iniciamos la integración en x=ε>0 usando la expansión en serie:
#   θ(x)   ≈ 1 - x^2/6
#   θ'(x)  ≈ -x/3
# -------------------------------------------------------------------------

def lane_emden_rhs(n):
    """
    Arma el sistema de 1er orden equivalente:
      y = [θ, φ] con φ = dθ/dx
      dθ/dx = φ
      dφ/dx = - (2/x) φ - θ^n
    """
    def f(x, y):
        theta, phi = y
        return np.array([phi, - (2.0/x) * phi - theta**n], dtype=float)
    return f

def theta_zero_event(x, y):
    """
    Evento para detectar el primer cero de θ: θ(x) = 0.
    Es terminal (detiene la integración) y lo buscamos en dirección descendente.
    """
    return y[0]  # raíz cuando θ = 0
theta_zero_event.terminal = True
theta_zero_event.direction = -1.0

def solve_lane_emden_for_n(n, rtol=1e-10, atol=1e-12):
    """
    Integra Lane–Emden para el índice n y devuelve:
      (x_star, mass_like, rho_ratio)
    donde:
      x_star     = primer cero de θ (radio)
      mass_like  = - x_star^2 * θ'(x_star)         (proporcional a la masa)
      rho_ratio  = - x_star / (3 * θ'(x_star))     (= ρc / ⟨ρ⟩)
    Si no se encuentra x* finito (p. ej. n=5), retorna (None, None, None).
    """
    f = lane_emden_rhs(n)

    # 1) Arranque cerca del origen (evita singularidad en x=0) usando serie
    eps = 1e-6
    theta0 = 1.0 - (eps**2)/6.0   # θ(ε) ≈ 1 - ε^2/6
    phi0   = - eps / 3.0          # θ'(ε) ≈ -ε/3
    y0 = np.array([theta0, phi0], dtype=float)

    # 2) Integramos aumentando x_max si no encontramos el cero a la primera
    x_star = None
    theta_prime_at_root = None
    for x_max in [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]:
        sol = solve_ivp(
            f, (eps, x_max), y0,
            method='LSODA',            # robusto para problemas suaves y con términos "1/x"
            events=theta_zero_event,   # detiene cuando θ cruza cero
            rtol=rtol, atol=atol,
            dense_output=False
        )
        # ¿Se detectó el primer cero?
        if sol.t_events[0].size > 0:
            x_star = float(sol.t_events[0][0])               # x* (radio)
            theta_prime_at_root = float(sol.y_events[0][0][1])  # θ'(x*) = φ en el evento
            break

    # 3) Si no hay cero finito (caso n=5), devolvemos Nones
    if x_star is None or theta_prime_at_root is None:
        return None, None, None

    # 4) Calculamos magnitudes pedidas
    mass_like = - (x_star**2) * theta_prime_at_root
    rho_ratio = - x_star / (3.0 * theta_prime_at_root)

    return x_star, mass_like, rho_ratio

def format_val_or_empty(val, fmt="{:.5f}"):
    """
    Formatea val si es válido; en caso contrario, devuelve cadena vacía.
    Útil para el caso n=5 (sin radio finito).
    """
    if val is None:
        return ""
    if isinstance(val, float) and not np.isfinite(val):
        return ""
    return fmt.format(val)

if __name__ == "__main__":
    # Índices a resolver (según enunciado)
    ns = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    # Resolvemos y almacenamos resultados
    results = []
    for n in ns:
        x_star, mass_like, rho_ratio = solve_lane_emden_for_n(n)
        results.append((n, x_star, mass_like, rho_ratio))

    # Exportamos a 7.csv con el formato pedido
    out_path = "Taller 3 - ODE/7.csv"
    with open(out_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Indice n", "Radio", "Masa", "rho_c/<rho>"])
        for (n, x_star, mass_like, rho_ratio) in results:
            writer.writerow([
                "{:.1f}".format(n),
                format_val_or_empty(x_star, "{:.5f}"),
                format_val_or_empty(mass_like, "{:.5f}"),
                format_val_or_empty(rho_ratio, "{:.5f}")
            ])



    print("Archivo generado:", out_path)