import cv2
import matplotlib.pyplot as plt
import os

# Forzar backend sin GUI
plt.switch_backend('Agg') 

def generar_comparativa_final(segundo=2):
    directorio_base = "/home/mkt/data/projects/ProyectoYOLO/Fase_1B_SuperRes" 
    
    video_paths = {
        "Original (Raw)": os.path.join(directorio_base, "videos", "raw", "UPS.mp4"),
        "Super Res (GPU)": os.path.join(directorio_base, "videos", "super_res", "gpu.mp4"),
        "YOLO Final (GPU)": os.path.join(directorio_base, "videos", "yolo", "gpu.mp4")
    }

    def extraer_frame_robusto(path, seg_objetivo):
        if not os.path.exists(path):
            print(f"ERROR: No existe: {path}")
            return None
        
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        
        frame_objetivo = int(fps * seg_objetivo)
        
        # Leemos secuencialmente hasta el frame deseado (más seguro en Linux Headless)
        count = 0
        success = False
        frame = None
        
        while count <= frame_objetivo:
            success, frame = cap.read()
            if not success:
                break
            count += 1
        
        cap.release()
        
        if not success or frame is None:
            print(f"ERROR: No se pudo alcanzar el frame {frame_objetivo} en {path}")
            return None
            
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frames = {}
    for nombre, path in video_paths.items():
        print(f"Procesando {nombre}...")
        frames[nombre] = extraer_frame_robusto(path, segundo)

    if any(f is None for f in frames.values()):
        print("\n[!] Fallo en la extracción. Prueba con un segundo menor (ej. segundo=1).")
        return

    # Gráfico
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor='white')
    titulos = list(frames.keys())
    imagenes = list(frames.values())
    
    for i, ax in enumerate(axes):
        ax.imshow(imagenes[i])
        ax.set_title(titulos[i], fontsize=16, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    output_file = "comparativa_sr_yolo.png"
    plt.savefig(output_file, dpi=300)
    print(f"\n¡ÉXITO! Imagen guardada en: {os.path.abspath(output_file)}")

# Cambiamos a segundo 1 por si los videos son cortos
generar_comparativa_final(segundo=0.4)