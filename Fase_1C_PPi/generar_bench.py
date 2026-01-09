import matplotlib.pyplot as plt
import numpy as np

# Configuración para entorno sin monitor (Headless)
plt.switch_backend('Agg')

# 1. Datos obtenidos en tus experimentos (en milisegundos)
labels = ['Detección\n(YOLOv11)', 'Super Resolución\n(Real-ESRGAN)', 'Preprocesamiento\n(C++/CUDA)']
cpu_times = [555.5, 2000.0, 28.29]  # Datos Fase B y C
gpu_times = [3.11, 35.33, 15.26]    # Datos Fase B y C

x = np.arange(len(labels))
width = 0.35  # Ancho de las barras

# 2. Creación del gráfico
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Colores institucionales / profesionales
rects1 = ax.bar(x - width/2, cpu_times, width, label='CPU (Intel i9)', color='#e53935', edgecolor='black')
rects2 = ax.bar(x + width/2, gpu_times, width, label='GPU (RTX 3070)', color='#43a047', edgecolor='black')

# 3. Personalización técnica
ax.set_ylabel('Tiempo de ejecución (ms) - ESCALA LOG')
ax.set_title('Análisis de Latencia: CPU vs GPU (RTX 3070)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend()

# Aplicar escala logarítmica porque la diferencia es de hasta 178x
ax.set_yscale('log')

# Añadir etiquetas de valor sobre las barras
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}ms',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Añadir una rejilla sutil para facilitar la lectura
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.tight_layout()

# 4. Guardar archivo
output_name = "bench_speedup.png"
plt.savefig(output_name)
print(f"¡Éxito! Gráfica de rendimiento guardada como: {output_name}")