# Pipeline de Visión Artificial: Segmentación, Super-Resolución y Optimización CUDA

![Badge Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Badge C++](https://img.shields.io/badge/C%2B%2B-17-00599C)
![Badge CUDA](https://img.shields.io/badge/CUDA-12.1-76B900)
![Badge YOLO](https://img.shields.io/badge/YOLO-v11-FFD700)
![Badge License](https://img.shields.io/badge/License-MIT-green)

Este repositorio contiene la implementación de un ecosistema de visión artificial de alto rendimiento desarrollado como parte de la práctica de investigación universitaria. El proyecto explora la integración de modelos de Deep Learning (YOLOv11, DOVE Diffusion) con optimización de bajo nivel en C++ y CUDA sobre una GPU NVIDIA RTX 3070.

## Arquitectura del Proyecto

El proyecto se divide en tres fases independientes, diseñadas para comparar arquitecturas secuenciales (CPU) vs. paralelas (GPU):

### -> Fase 1A: Segmentación de Instancias
Implementación de **YOLOv11-seg** para la detección y segmentación semántica de objetos en tiempo real (Baraja de cartas).
- **Entrada:** Video en tiempo real (Webcam/Smartphone).
- **Salida:** Máscaras binarias y Bounding Boxes con confianza > 0.8.

### -> Fase 1B: Detección y Super Resolución (Video Enhancement)
Pipeline híbrido para la reconstrucción de video de tráfico degradado y baja resolución.
- **Tecnología:** Se sustituyó el enfoque GAN por **DOVE (One-Step Diffusion)** para garantizar coherencia temporal.
- **Flujo:** Video Low-Res -> Upscaling x4 (DOVE) -> Inferencia YOLOv11.
- **Resultado:** Recuperación de detección de objetos pequeños (coches lejanos) que eran invisibles en el video original.

### -> Fase 1C: Optimización C++ / CUDA (Imagen Médica)
Implementación de filtros de procesamiento de imagen "GPU-Only" para eliminar la latencia del bus PCIe.
- **Algoritmos:** Filtro Bilateral (Denoising) + Detector de Bordes Canny.
- **Hardware:** Uso de memoria `cv::cuda::GpuMat` para mantener el procesamiento 100% en VRAM.
- **Aplicación:** Procesamiento de Tomografías de Tórax (Chest CT) de alta resolución (6K).

---

## Resultados de Rendimiento (Benchmark)

Las pruebas se realizaron comparando una CPU Intel Core i9 vs. NVIDIA RTX 3070.

| Etapa del Pipeline | CPU (FPS) | GPU (FPS) | Speedup (Aceleración) |
|--------------------|-----------|-----------|-----------------------|
| **Inferencia YOLOv11** | 1.8 FPS | **321.4 FPS** | **178x** |
| **Super Resolución (DOVE)** | < 0.5 FPS | **219.2 FPS** | **> 400x** |
| **Filtro Médico (Bilateral)** | ~35 FPS | **65.5 FPS** | **1.85x** |

> **Nota:** La implementación en C++ redujo la latencia de procesamiento médico de 28.29ms (CPU) a 15.26ms (GPU).

---

## Requisitos e Instalación

### Prerrequisitos
- Drivers NVIDIA (535+) y CUDA Toolkit 12.1.
- Python 3.9+
- Compilador C++ (g++ / Visual Studio) con soporte CMake.
- OpenCV compilado con soporte CUDA (`WITH_CUDA=ON`).

### Instalación (Python)
```bash
git clone https://github.com/marcoxskii/PracticaVision_4.1.git
cd PracticaVision_4.1
pip install -r requirements.txt
```

### Compilación (C++ Fase 1C)
```bash
cd Fase_1C_PPi
mkdir build && cd build
cmake ..
make
./gpu_benchmark ../images/med2.png
```

---

## Demostración y Recursos

- **Video Explicativo (YouTube):** [Ver demostración de Fase 1B](https://youtu.be/x38ov16CQpM)
- **Paper Modelo DOVE:** [arXiv:2505.16239](https://arxiv.org/abs/2505.16239)

## Autor

**Marco Cajamarca**
- *Estudiante de Ciencias de la Computación*
- *Universidad Politécnica Salesiana - Cuenca, Ecuador*

---
*Este proyecto fue desarrollado utilizando un entorno remoto SSH sobre servidor Linux (Ubuntu) y cliente macOS, puede que al intentar ejecutar nuevamente sucedan errores.*
