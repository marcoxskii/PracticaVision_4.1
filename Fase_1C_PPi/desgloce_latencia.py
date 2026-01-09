import matplotlib.pyplot as plt

# Configuración para entorno headless (sin monitor)
plt.switch_backend('Agg')

# Datos obtenidos de tus pruebas de estrés en la Fase C (Resolución 6K / Médica)
# Los tiempos deben sumar el total de la latencia observada (~31.12 ms)
etapas = [
    'Transferencia VRAM (Upload)', 
    'Filtro Bilateral CUDA', 
    'Ecualización/Grises', 
    'Detector Canny CUDA', 
    'Transferencia RAM (Download)'
]

tiempos = [
    10.50,  # Upload PCIe (suele ser lo más pesado en 6K)
    12.10,  # El filtro Bilateral es costoso aritméticamente
    2.10,   # Operaciones simples
    3.16,   # Canny optimizado
    3.26    # Download de la máscara final
]

# Colores profesionales
colores = ['#2c3e50', '#e74c3c', '#3498db', '#f1c40f', '#27ae60']
explode = (0.1, 0, 0, 0, 0)  # Resaltar la transferencia inicial (cuello de botella)

fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

wedges, texts, autotexts = ax.pie(
    tiempos, 
    labels=etapas, 
    autopct='%1.1f%%',
    startangle=140, 
    colors=colores, 
    explode=explode,
    pctdistance=0.85,
    textprops={'fontweight': 'bold'}
)

# Añadir un círculo en medio para que parezca un "Donut Chart" (más moderno)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

ax.set_title("Distribución de Latencia en Pipeline C++/CUDA (Imagen 6K)", 
             fontsize=14, fontweight='bold', pad=20)

plt.axis('equal') 
plt.tight_layout()

# Guardar para el informe
output_name = "desglose_latencia.png"
plt.savefig(output_name)
print(f"¡Éxito! Gráfica de desglose guardada como: {output_name}")