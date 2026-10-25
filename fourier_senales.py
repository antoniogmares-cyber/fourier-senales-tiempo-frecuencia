"""
Actividad Formativa - Transformada de Fourier en el dominio del tiempo y la frecuencia
Autor: Antonio de Jesús Guevara Mares
Librerías: NumPy, Matplotlib
Estructura: 1) señal en el tiempo  2) FFT  3) visualización  4) propiedades  5) aplicación real
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Carpeta donde se guardan las gráficas (se crea si no existe)
os.makedirs("figuras", exist_ok=True)

# ---------------------------------------------------------------
# Parámetros generales
# ---------------------------------------------------------------
fs = 1000                      # frecuencia de muestreo (Hz)
T = 2.0                        # duración (s)
N = int(fs * T)                # número de muestras
t = np.arange(N) / fs - T / 2  # eje de tiempo centrado en 0
f = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))  # eje de frecuencia centrado en 0


def espectro(x):
    """Calcula la FFT centrada, normalizada por fs (aprox. a la TF continua)."""
    X = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x))) / fs
    return X


def graficar(x, X, titulo, archivo, fmax=50):
    """Gráfica en 3 paneles: señal, magnitud y fase."""
    mag = np.abs(X)
    fase = np.angle(X)
    fase[mag < 1e-3 * mag.max()] = 0   # se oculta la fase donde la magnitud es despreciable
    fig, ax = plt.subplots(3, 1, figsize=(9, 8))
    ax[0].plot(t, x); ax[0].set_title(f"{titulo} - dominio del tiempo")
    ax[0].set_xlabel("Tiempo (s)"); ax[0].set_ylabel("x(t)"); ax[0].grid(alpha=.3)
    ax[1].plot(f, mag); ax[1].set_xlim(-fmax, fmax)
    ax[1].set_title("Magnitud |X(f)|"); ax[1].set_xlabel("Frecuencia (Hz)"); ax[1].grid(alpha=.3)
    ax[2].plot(f, fase); ax[2].set_xlim(-fmax, fmax)
    ax[2].set_title("Fase ∠X(f)"); ax[2].set_xlabel("Frecuencia (Hz)")
    ax[2].set_ylabel("rad"); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"figuras/{archivo}", dpi=130); plt.close()


# ---------------------------------------------------------------
# 1) Señales elementales (dominio del tiempo)
# ---------------------------------------------------------------
pulso = np.where(np.abs(t) <= 0.25, 1.0, 0.0)      # pulso rectangular de ancho 0.5 s
escalon = np.where(t >= 0, 1.0, 0.0)               # escalón unitario u(t)
senoidal = np.sin(2 * np.pi * 5 * t)               # seno de 5 Hz

# ---------------------------------------------------------------
# 2) Transformada de Fourier
# ---------------------------------------------------------------
X_pulso, X_esc, X_sen = espectro(pulso), espectro(escalon), espectro(senoidal)

# ---------------------------------------------------------------
# 3) Visualización
# ---------------------------------------------------------------
graficar(pulso, X_pulso, "Pulso rectangular", "01_pulso.png")
graficar(escalon, X_esc, "Función escalón", "02_escalon.png")
graficar(senoidal, X_sen, "Señal senoidal (5 Hz)", "03_senoidal.png")

# ---------------------------------------------------------------
# 4) Propiedades de la transformada
# ---------------------------------------------------------------
# 4.1 Linealidad: TF{a*x1 + b*x2} = a*X1 + b*X2
a, b = 2.0, 0.5
combinada = a * pulso + b * senoidal
err_lin = np.max(np.abs(espectro(combinada) - (a * X_pulso + b * X_sen)))
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(t, combinada); ax[0].set_title("Linealidad: 2·pulso + 0.5·seno(5 Hz)")
ax[0].set_xlabel("Tiempo (s)"); ax[0].grid(alpha=.3)
ax[1].plot(f, np.abs(espectro(combinada)), label="TF de la combinación")
ax[1].plot(f, np.abs(a * X_pulso + b * X_sen), "--", label="a·X1 + b·X2")
ax[1].set_xlim(-30, 30); ax[1].legend(); ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_title(f"Magnitud (error máximo = {err_lin:.2e})"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("figuras/04_linealidad.png", dpi=130); plt.close()

# 4.2 Desplazamiento en el tiempo: x(t - t0) -> X(f)·e^{-j2πf t0}  (misma magnitud, fase cambia)
t0 = 0.3
pulso_des = np.where(np.abs(t - t0) <= 0.25, 1.0, 0.0)
X_des = espectro(pulso_des)
X_teorico = X_pulso * np.exp(-1j * 2 * np.pi * f * t0)
err_des = np.max(np.abs(X_des - X_teorico))
fig, ax = plt.subplots(3, 1, figsize=(9, 8))
ax[0].plot(t, pulso, label="original"); ax[0].plot(t, pulso_des, "--", label=f"desplazado {t0} s")
ax[0].legend(); ax[0].set_title("Desplazamiento en el tiempo"); ax[0].grid(alpha=.3)
ax[1].plot(f, np.abs(X_pulso), label="original"); ax[1].plot(f, np.abs(X_des), "--", label="desplazado")
ax[1].set_xlim(-20, 20); ax[1].legend(); ax[1].set_title("Magnitud: no cambia"); ax[1].grid(alpha=.3)
mag = np.abs(X_pulso); fo = np.angle(X_pulso); fd = np.angle(X_des)
fo[mag < 1e-3 * mag.max()] = 0; fd[mag < 1e-3 * mag.max()] = 0
ax[2].plot(f, fo, label="original"); ax[2].plot(f, fd, label="desplazado", alpha=.8)
ax[2].set_xlim(-20, 20); ax[2].legend(); ax[2].set_title(f"Fase: cambia (error vs. teoría = {err_des:.2e})")
ax[2].set_xlabel("Frecuencia (Hz)"); ax[2].grid(alpha=.3)
plt.tight_layout(); plt.savefig("figuras/05_desplazamiento.png", dpi=130); plt.close()

# 4.3 Escalamiento: x(a t) -> (1/|a|) X(f/a). Comprimir en el tiempo ensancha el espectro
pulso_comp = np.where(np.abs(t) <= 0.125, 1.0, 0.0)   # equivale a x(2t)
X_comp = espectro(pulso_comp)
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(t, pulso, label="x(t) ancho 0.5 s"); ax[0].plot(t, pulso_comp, label="x(2t) ancho 0.25 s")
ax[0].legend(); ax[0].set_title("Escalamiento: compresión en el tiempo"); ax[0].grid(alpha=.3)
ax[1].plot(f, np.abs(X_pulso), label="|X(f)|"); ax[1].plot(f, np.abs(X_comp), label="|X2(f)| (más ancho)")
ax[1].set_xlim(-30, 30); ax[1].legend(); ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].set_title("Espectro: comprimir en tiempo = ensanchar en frecuencia"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("figuras/06_escalamiento.png", dpi=130); plt.close()

# Desplazamiento en frecuencia (modulación): x(t)·cos(2π f0 t)
f0 = 20
mod = pulso * np.cos(2 * np.pi * f0 * t)
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
ax[0].plot(t, mod); ax[0].set_title("Pulso modulado con coseno de 20 Hz"); ax[0].grid(alpha=.3)
ax[1].plot(f, np.abs(espectro(mod))); ax[1].set_xlim(-50, 50)
ax[1].set_title("El espectro del pulso se desplaza a ±20 Hz"); ax[1].set_xlabel("Frecuencia (Hz)")
ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("figuras/07_modulacion.png", dpi=130); plt.close()

# ---------------------------------------------------------------
# 5) Aplicación real: señal tipo ECG con ruido de línea eléctrica de 60 Hz
# ---------------------------------------------------------------
fs2 = 500
t2 = np.arange(0, 4, 1 / fs2)
ecg = np.zeros_like(t2)
for latido in np.arange(0.4, 4, 0.8):      # ~75 latidos por minuto
    ecg += 1.0 * np.exp(-((t2 - latido) ** 2) / (2 * 0.012 ** 2))        # pico R
    ecg += 0.25 * np.exp(-((t2 - latido + 0.2) ** 2) / (2 * 0.04 ** 2))  # onda P
    ecg += 0.35 * np.exp(-((t2 - latido - 0.25) ** 2) / (2 * 0.06 ** 2)) # onda T
ruido = 0.3 * np.sin(2 * np.pi * 60 * t2)            # interferencia de 60 Hz
ecg_ruido = ecg + ruido

F = np.fft.fft(ecg_ruido)
f2 = np.fft.fftfreq(len(t2), 1 / fs2)
F_filtrada = F.copy()
F_filtrada[(np.abs(f2) > 55) & (np.abs(f2) < 65)] = 0   # filtro "notch" en frecuencia
ecg_limpia = np.real(np.fft.ifft(F_filtrada))

fig, ax = plt.subplots(3, 1, figsize=(10, 9))
ax[0].plot(t2, ecg_ruido); ax[0].set_title("ECG simulado con ruido de 60 Hz"); ax[0].set_xlabel("Tiempo (s)")
ax[1].plot(f2[:len(f2)//2], np.abs(F)[:len(f2)//2] / len(t2))
ax[1].set_xlim(0, 100); ax[1].set_title("Espectro: se identifica el pico de 60 Hz"); ax[1].set_xlabel("Frecuencia (Hz)")
ax[2].plot(t2, ecg_limpia); ax[2].set_title("ECG después de eliminar 60 Hz en el dominio de la frecuencia")
ax[2].set_xlabel("Tiempo (s)")
for a_ in ax: a_.grid(alpha=.3)
plt.tight_layout(); plt.savefig("figuras/08_ecg.png", dpi=130); plt.close()

print(f"Error linealidad: {err_lin:.2e}")
print(f"Error desplazamiento: {err_des:.2e}")
print(f"Pico de ruido detectado en: {abs(f2[np.argmax(np.abs(F)[:len(f2)//2][20:])+20]):.1f} Hz")
