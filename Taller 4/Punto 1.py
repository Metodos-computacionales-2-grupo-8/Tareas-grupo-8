# ...existing code...

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import shutil   # # nuevo: para detectar ffmpeg en PATH

# ---------------------------------------------------------------------
# Funciones y parámetros comunes
# ---------------------------------------------------------------------

# # Crea el paquete inicial: gaussiana localizada en x0 con ancho sigma y fase (velocidad) k0
def gaussian_packet(x, x0=10.0, sigma=1/np.sqrt(4.0), k0=2.0):
    # # exp(-2 (x-x0)^2) corresponde a sigma relacionado con el enunciado
    # # la fase compleja exp(-i k0 x) le da velocidad al paquete
    psi = np.exp(-2.0 * (x - x0) ** 2) * np.exp(-1j * k0 * x)
    # # normalizar la función de onda (usar trapezoid para evitar DeprecationWarning)
    psi /= np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
    return psi

# # Construye el operador Laplaciano discreto (segunda derivada) con condiciones Dirichlet
# ...existing code...
def laplacian_matrix(N, dx):
    # # matriz tridiagonal con stencil [1, -2, 1]/dx^2
    main = -2.0 * np.ones(N)
    off = np.ones(N - 1)
    # # no usar vstack porque off tiene longitud N-1 y main N
    data = [off, main, off]                           # <-- usar lista de arrays
    A = sp.diags(data, offsets=[-1, 0, 1], format='csc') / (dx ** 2)
    return A
# ...existing code...

# # Evolve usando Crank-Nicolson: (I - i dt/2 H) psi_{n+1} = (I + i dt/2 H) psi_n
def crank_nicolson_step_factorized(A_factor, B, psi):
    # # usa la factorización (A_factor) para resolver A psi_next = B psi
    rhs = B.dot(psi)
    psi_next = A_factor(rhs)
    return psi_next

# ---------------------------------------------------------------------
# Función principal de simulación
# ---------------------------------------------------------------------
def run_simulation(V_func, title, x_min=-20.0, x_max=20.0, Nx=801, alpha=0.1,
                   dt=0.01, tmax=150.0, save_mp4=True, mp4_name=None, fps=25):
    # # definir malla espacial
    x = np.linspace(x_min, x_max, Nx)
    dx = x[1] - x[0]

    # # potencial en la malla
    V = V_func(x)

    # # operador Laplaciano discreto y Hamiltoniano H = -alpha * lap + diag(V)
    Lap = laplacian_matrix(Nx, dx)                # # matriz Laplaciana
    H = (-alpha) * Lap + sp.diags(V, 0, format='csc')  # # Hamiltoniano discreto (sparse)

    # # matrices Crank-Nicolson
    I = sp.eye(Nx, format='csc')
    A = (I - 1j * dt / 2.0 * H).tocsc()  # # izquierda: A psi_{n+1}
    B = (I + 1j * dt / 2.0 * H).tocsc()  # # derecha: B psi_n

    # # factorizar A para resolver de manera eficiente en cada paso
    A_lu = spla.factorized(A)  # # devuelve función que resuelve A x = b

    # # condiciones iniciales: paquete gaussiano (ver enunciado)
    psi = gaussian_packet(x, x0=10.0, sigma=1/np.sqrt(4.0), k0=2.0)

    # # arreglos para guardar la evolución de <x> y sigma
    times = np.arange(0.0, tmax, dt)
    save_interval = max(1, int(1.0 / (fps * dt)))  # # cuántos pasos entre frames (intenta ~fps)
    frames = []

    mu_list = []
    sigma_list = []
    t_list = []

    # # preparar figura para animación (probabilidad |psi|^2)
    fig, ax = plt.subplots()
    line, = ax.plot(x, np.abs(psi) ** 2, color='tab:blue')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.2 * np.max(np.abs(psi) ** 2))
    ax.set_xlabel('x')
    ax.set_ylabel('|psi|^2')
    ax.set_title(title)

    # # función que actualiza un frame (usada por FuncAnimation)
    def update(frame_idx):
        nonlocal psi
        # # avanzar 'save_interval' pasos de tiempo usando Crank-Nicolson
        for _ in range(save_interval):
            psi = crank_nicolson_step_factorized(A_lu, B, psi)

        # # normalizar numéricamente (para evitar error acumulado)
        norm = np.trapezoid(np.abs(psi) ** 2, x)
        psi /= np.sqrt(norm)

        # # calcular probabilidad, media y varianza
        prob = np.abs(psi) ** 2
        mu = np.trapezoid(x * prob, x)
        x2 = np.trapezoid((x ** 2) * prob, x)
        sigma = np.sqrt(np.abs(x2 - mu ** 2))

        mu_list.append(mu)
        sigma_list.append(sigma)
        t_list.append(frame_idx * save_interval * dt)

        # # actualizar curva
        line.set_ydata(prob)
        ax.set_ylim(0, 1.2 * max(1e-8, prob.max()))
        ax.set_title(f"{title}\n t = {t_list[-1]:.2f}, <x> = {mu:.3f}, sigma = {sigma:.3f}")
        return (line,)

    # # crear la animación
    n_frames = int(len(times) / save_interval)
    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)

    # # guardar video (requiere ffmpeg instalado y en PATH). Si no hay ffmpeg, fallback a GIF
    if save_mp4:
        if mp4_name is None:
            mp4_name = f"{title.replace(' ', '_')}.mp4"

        # # usar ffmpeg si está disponible
        if shutil.which('ffmpeg') is not None:
            print(f"ffmpeg detectado: guardando MP4 -> {mp4_name} (puede tardar)...")
            writer = animation.FFMpegWriter(fps=fps, codec='libx264')
            ani.save(mp4_name, writer=writer, dpi=150)
            print("Vídeo MP4 guardado.")
        else:
            # # fallback: guardar GIF con Pillow
            gif_name = mp4_name.replace('.mp4', '.gif')
            print("ffmpeg no disponible en PATH. Guardando GIF en su lugar:", gif_name)
            ani.save(gif_name, writer='pillow', fps=fps, dpi=150)
            print("GIF guardado.")
            mp4_name = gif_name

    plt.close(fig)  # # cerramos la figura para no mostrarla aquí

# ...existing code...
    # # Tras la simulación, graficar <x> con barras de error mu ± sigma
    plt.figure()
    plt.plot(t_list, mu_list, color='black', label=r'$\mu(t)=\langle x\rangle$')
    plt.fill_between(t_list, np.array(mu_list) - np.array(sigma_list),
                     np.array(mu_list) + np.array(sigma_list),
                     color='gray', alpha=0.4, label=r'$\mu \pm \sigma$')
    plt.xlabel('t')
    plt.ylabel('<x>(t)')
    plt.title(f"{title} — posición media e incertidumbre")
    plt.legend()

    # # corregir extensión: siempre guardar la imagen como .png aunque el video sea .mp4 o .gif
    base_name = (mp4_name if mp4_name is not None else title)   # # usar mp4_name si existe, sino el título
    root, _ext = os.path.splitext(base_name)                     # # separar la extensión actual
    png_name = root + '.png'                                     # # forzar .png para savefig

    plt.savefig(png_name, dpi=150)
    plt.show()
# ...existing code...
    # # devolver resultados esenciales por si se quieren analizar más
    return {'x': x, 'psi': psi, 't': t_list, 'mu': mu_list, 'sigma': sigma_list, 'video': mp4_name}

# ---------------------------------------------------------------------
# Potenciales para los incisos
# ---------------------------------------------------------------------

# 1.a Oscilador armónico: V(x) = -x^2 / 50  (según enunciado de la imagen)
def V_harmonic(x):
    # # potencial armónico (negativo en la imagen)
    return - (x ** 2) / 50.0

# 1.b Oscilador cuártico: V(x) = (x/5)^4  (no armónico)
def V_quartic(x):
    return (x / 5.0) ** 4

# 1.c Potencial sombrero: V(x) = 1/50 * (x^4 / 100 - x^2)
def V_hat(x):
    return (1.0 / 50.0) * (x ** 4 / 100.0 - x ** 2)

# ---------------------------------------------------------------------
# Ejecutar los tres incisos
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # # Nota: para 1.a en el enunciado se pide tmax=150 y alpha=0.1
    # # Guardaremos cada animación en un mp4 distinto
    out1 = run_simulation(V_harmonic, "1.a_Oscilador_armonico", tmax=150.0, dt=0.01, fps=25, Nx=801)
    out2 = run_simulation(V_quartic, "1.b_Oscilador_cuartico", tmax=50.0, dt=0.01, fps=25, Nx=801)
    out3 = run_simulation(V_hat, "1.c_Potencial_sombrero", tmax=150.0, dt=0.01, fps=25, Nx=801)

# ...existing code...