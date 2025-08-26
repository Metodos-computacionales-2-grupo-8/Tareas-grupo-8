import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lombscargle


# -----------------------------------------------------------------------------
# Utilidades de carga y chequeo
# -----------------------------------------------------------------------------
def load_ogle_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga archivo OGLE con 3 columnas: tiempo (d), brillo, sigma."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("El archivo debe tener al menos 3 columnas (t, brillo, sigma)")

    t = data[:, 0].astype(float)
    y = data[:, 1].astype(float)
    s = data[:, 2].astype(float)

    # Ordenar por tiempo por seguridad
    order = np.argsort(t)
    return t[order], y[order], s[order]


def sampling_report(t: np.ndarray) -> dict:
    """Calcula métricas del muestreo para evaluar si ~1/día."""
    dt = np.diff(t)
    # Eliminar saltos extremadamente grandes para estadísticas robustas
    # (pero igual se reportan percentiles)
    stats = {
        "n": int(t.size),
        "baseline_days": float(t.max() - t.min()) if t.size else 0.0,
        "dt_count": int(dt.size),
        "dt_median": float(np.median(dt)) if dt.size else np.nan,
        "dt_mean": float(np.mean(dt)) if dt.size else np.nan,
        "dt_std": float(np.std(dt)) if dt.size else np.nan,
        "dt_p5": float(np.percentile(dt, 5)) if dt.size else np.nan,
        "dt_p95": float(np.percentile(dt, 95)) if dt.size else np.nan,
        "frac_within_1d_pm_0p1": float(np.mean(np.abs(dt - 1.0) <= 0.1)) if dt.size else np.nan,
        "frac_within_0p5_to_1p5": float(np.mean((dt >= 0.5) & (dt <= 1.5))) if dt.size else np.nan,
    }
    return stats


# -----------------------------------------------------------------------------
# Espectro Lomb–Scargle (muestreo irregular)
# -----------------------------------------------------------------------------
def find_dominant_frequency(
    t: np.ndarray,
    y: np.ndarray,
    fmin: float = 0.02,   # 1/50 d^{-1}  -> periodos hasta 50 días
    fmax: float = 5.0,    # 5 d^{-1}     -> periodos de 0.2 días
    n_freq: int = 20000,  # resolución fina
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Calcula Lomb–Scargle (SciPy) y devuelve:
      - f_peak (1/día)
      - f_grid (1/día)
      - P (potencia normalizada)
    """
    # Precentrado: SciPy también permite precenter=True, pero restamos media por claridad
    y0 = y - np.mean(y)

    f_grid = np.linspace(fmin, fmax, n_freq)
    w = 2.0 * np.pi * f_grid  # SciPy usa frecuencia angular
    P = lombscargle(t, y0, w, precenter=True, normalize=True)

    # Frecuencia dominante
    i_peak = int(np.argmax(P))
    f_peak = float(f_grid[i_peak])
    return f_peak, f_grid, P


# -----------------------------------------------------------------------------
# Gráfica fase–brillo
# -----------------------------------------------------------------------------
def plot_phase_curve(
    t: np.ndarray,
    y: np.ndarray,
    s: np.ndarray | None,
    f: float,
    out_pdf: str,
    invert_y: bool = True,
) -> None:
    """Genera el gráfico brillo vs fase φ = mod(f*t, 1) y lo guarda en PDF."""
    phi = np.mod(f * t, 1.0)

    plt.figure(figsize=(7.5, 5.5))
    if s is not None:
        plt.errorbar(phi, y, yerr=s, fmt=".", ms=3, elinewidth=0.6, alpha=0.8, label="Datos")
        # Repetir en [1,2) para visualizar continuidad
        plt.errorbar(phi + 1.0, y, yerr=s, fmt=".", ms=3, elinewidth=0.6, alpha=0.3)
    else:
        plt.plot(phi, y, ".", ms=3, alpha=0.8, label="Datos")
        plt.plot(phi + 1.0, y, ".", ms=3, alpha=0.3)

    if invert_y:
        # En magnitudes, valores menores son más brillantes
        plt.gca().invert_yaxis()

    period = 1.0 / f if f > 0 else np.nan
    plt.xlabel("Fase φ = mod(f · t, 1)")
    plt.ylabel("Brillo (mag)")
    plt.title(f"Curva de luz plegada – f = {f:.6f} d⁻¹ (P ≈ {period:.4f} d)")
    plt.xlim(0.0, 2.0)
    plt.grid(alpha=0.25)
    plt.legend(loc="best")

    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()
    print(f"Gráfico de fase guardado en: {out_pdf}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Rutas (según enunciado)
    data_path = r"Taller 2 - Fourier/Datos/OGLE-LMC-CEP-0001.dat"
    out_pdf = r"Taller 2 - Fourier/4.pdf"

    # 1) Cargar datos
    t, y, s = load_ogle_data(data_path)

    # 2) Evaluación del muestreo (~1/día)
    stats = sampling_report(t)
    print("Resumen de muestreo")
    print(f"  Número de puntos:                {stats['n']}")
    print(f"  Baseline (días):                 {stats['baseline_days']:.3f}")
    print(f"  Δt median (días):                {stats['dt_median']:.3f}")
    print(f"  Δt mean ± std (días):            {stats['dt_mean']:.3f} ± {stats['dt_std']:.3f}")
    print(f"  Δt p5–p95 (días):                {stats['dt_p5']:.3f} – {stats['dt_p95']:.3f}")
    print(f"  Fracción |Δt−1| ≤ 0.1:           {stats['frac_within_1d_pm_0p1']:.3f}")
    print(f"  Fracción 0.5 ≤ Δt ≤ 1.5:         {stats['frac_within_0p5_to_1p5']:.3f}")

    if stats["frac_within_0p5_to_1p5"] >= 0.7:
        print("Evaluación: El muestreo puede considerarse aproximadamente 1 medición por día.")
    else:
        print("Evaluación: El muestreo es claramente irregular y no estrictamente 1/día.")

    # 3) Frecuencia dominante (Lomb–Scargle)
    # Rango de búsqueda razonable para Cefeidas: 0.02–5 d^{-1}
    f_peak, f_grid, P = find_dominant_frequency(t, y, fmin=0.02, fmax=5.0, n_freq=20000)
    period = 1.0 / f_peak
    print(f"\nFrecuencia dominante estimada: f = {f_peak:.8f} d^-1  (Periodo ≈ {period:.6f} días)")

    # 4) Verificación de fase y visualización (guarda 4.pdf)
    plot_phase_curve(t, y, s, f_peak, out_pdf)