import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import os
import glob
import tempfile
import time

# --- CONFIGURACIÓN E INTERFAZ ---
st.set_page_config(page_title="YOLOv8 Dashboard", page_icon="🎴", layout="wide")

# Estilos para ocultar logs molestos y mejorar la estética
st.markdown("""
    <style>
    .stButton>button {width: 100%; font-weight: bold;}
    /* Ocultar elementos de carga excesivos */
    .stDeployButton {display:none;}
    </style>
    """, unsafe_allow_html=True)

# --- CABECERA ---
col_logo, col_title = st.columns([1, 6])
with col_title:
    st.title("Sistema de Detección de Cartas")
    st.caption(f"🚀 Hardware: **{torch.cuda.get_device_name(0)}** | Modo: Renderizado vs Tiempo Real")

st.divider()

# --- BARRA LATERAL ---
st.sidebar.title("⚙️ Configuración")

# 1. Selector de Modelo
search_paths = ["*.pt", "runs/segment/*/weights/*.pt", "Proyecto_Cartas/*/weights/*.pt"]
model_files = []
for path in search_paths:
    model_files.extend(glob.glob(path))

# Aseguramos que el usuario vea claramente cuál es cuál
selected_model = st.sidebar.selectbox("Seleccionar Modelo", model_files, index=len(model_files)-2 if len(model_files)>1 else 0)

# 2. Configuración de Inferencia
confidence = st.sidebar.slider("Confianza (Threshold)", 0.0, 1.0, 0.5)

# 3. Fuente de Video
source_options = {"Video de Prueba": "video_prueba.mp4", "Webcam (Solo Local)": 0}
source_label = st.sidebar.selectbox("Fuente de Video", list(source_options.keys()))
source_path = source_options[source_label]

st.sidebar.info("💡 Consejo: Selecciona 'Proyecto_Cartas/.../best.pt' para ver tus cartas.")

# --- LÓGICA PRINCIPAL ---

# Pestañas para separar el modo "En Vivo" (Lento) del "Renderizado" (Rápido y Fluido)
tab1, tab2 = st.tabs(["Procesar Video Completo (Recomendado)", "Prueba en Tiempo Real"])

# --- MODO 1: RENDERIZADO COMPLETO ---
with tab1:
    st.write("### Generar Video Final")
    st.markdown("Esta opción procesa todo el video en el servidor y te muestra el resultado final fluido, sin cortes por internet.")
    
    if st.button("⚡ Procesar y Renderizar Video", type="primary"):
        if source_label == "Webcam (Solo Local)":
            st.error("Este modo solo funciona con archivos de video, no con webcam.")
        else:
            model = YOLO(selected_model)
            cap = cv2.VideoCapture(source_path)
            
            # Configuración de video de salida
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Crear archivo temporal para el video procesado
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
            
            # Usamos 'vp80' que es el codec estandar para webm
            fourcc = cv2.VideoWriter_fourcc(*'vp80') 
            
            out = cv2.VideoWriter(tfile.name, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            start_time = time.time()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inferencia (verbose=False elimina logs en consola)
                results = model(frame, conf=confidence, verbose=False)
                annotated_frame = results[0].plot()
                
                out.write(annotated_frame)
                
                # Actualizar barra de progreso
                frame_count += 1
                if frame_count % 10 == 0: # Actualizar UI cada 10 frames para no frenar
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Procesando frame {frame_count}/{total_frames}...")

            cap.release()
            out.release()
            
            duration = time.time() - start_time
            status_text.success(f"¡Listo! Procesado en {duration:.1f} segundos.")
            progress_bar.empty()
            
            # Mostrar video final
            st.video(tfile.name)

# --- MODO 2: TIEMPO REAL (LEGACY) ---
with tab2:
    st.write("### Stream en Vivo")
    col_video, col_ctrl = st.columns([3, 1])
    
    with col_ctrl:
        run_live = st.button("▶️ Iniciar Stream")
        stop_live = st.button("⏹️ Detener")
    
    with col_video:
        st_frame = st.empty()
        
    if run_live:
        model = YOLO(selected_model)
        cap = cv2.VideoCapture(source_path)
        
        while cap.isOpened() and not stop_live:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Reducir tamaño antes de enviar para mejorar velocidad en red lenta
            # frame = cv2.resize(frame, (640, 360)) 
            
            results = model(frame, conf=confidence, verbose=False)
            annotated_frame = results[0].plot()
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
        cap.release()

# Instrucciones finales
# source venv/bin/activate
# streamlit run app.py --server.address 0.0.0.0
# http://100.76.3.19:8501