# Transformada de Fourier: señales en el dominio del tiempo y de la frecuencia

Actividad formativa de la carrera de Ingeniería en Diseño de Software (Universidad Ciudadana de Nuevo León).
Autor: Antonio de Jesús Guevara Mares

## Propósito
Analizar señales elementales (pulso rectangular, escalón y senoidal) en el dominio del tiempo y de la frecuencia usando la Transformada de Fourier con `np.fft.fft()`, verificar sus propiedades y aplicarla a un caso real: limpiar una señal de ECG con ruido de 60 Hz.

## Contenido
- `fourier_senales.py`: código comentado con todo el análisis.
- `figuras/`: gráficas generadas por el código.

## Qué hace el código
1. Define las señales en el tiempo.
2. Calcula la FFT y su espectro (magnitud y fase).
3. Grafica los resultados.
4. Verifica linealidad, desplazamiento en el tiempo, escalamiento y desplazamiento en frecuencia.
5. Aplica la FFT a un ECG simulado con ruido de línea eléctrica y lo filtra en frecuencia.

## Requisitos
```
pip install numpy matplotlib
```

## Ejecución
```
python fourier_senales.py
```
Las gráficas se guardan en la carpeta `figuras/`.
