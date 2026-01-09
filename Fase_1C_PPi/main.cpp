#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

using namespace std;
using namespace cv;

// Estructura para almacenar tiempos
struct BenchResult {
    double cpu_time;
    double gpu_time;
};

// --- PIPELINE DE PROCESAMIENTO ---
// Esta función encapsula la lógica para asegurar que CPU y GPU hagan lo mismo
BenchResult run_benchmark(const Mat& input, const string& filename) {
    string out_dir = "images_out/";
    BenchResult res;

    // --- FASE CPU (i9) ---
    Mat cpu_gray, cpu_eq, cpu_blur, cpu_morph, cpu_edges;
    double t0 = (double)getTickCount();

    cvtColor(input, cpu_gray, COLOR_BGR2GRAY);
    equalizeHist(cpu_gray, cpu_eq);
    // Filtro Bilateral: Diámetro 15, SigmaColor 80, SigmaSpace 80 (Muy pesado para CPU)
    bilateralFilter(cpu_eq, cpu_blur, 15, 80, 80);
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    erode(cpu_blur, cpu_morph, element);
    Canny(cpu_morph, cpu_edges, 20, 50, 3, true);

    res.cpu_time = ((double)getTickCount() - t0) / getTickFrequency() * 1000;

    // --- FASE GPU (RTX 3070) ---
    cuda::GpuMat d_src, d_gray, d_eq, d_blur, d_morph, d_edges;
    Ptr<cuda::CannyEdgeDetector> canny = cuda::createCannyEdgeDetector(20, 50, 3, true);
    Ptr<cuda::Filter> erosion = cuda::createMorphologyFilter(MORPH_ERODE, CV_8UC1, element);

    double t1 = (double)getTickCount();

    d_src.upload(input);
    cuda::cvtColor(d_src, d_gray, COLOR_BGR2GRAY);
    cuda::equalizeHist(d_gray, d_eq);
    // El filtro bilateral en CUDA es donde se nota la mayor potencia
    cuda::bilateralFilter(d_gray, d_blur, 15, 80, 80);
    erosion->apply(d_blur, d_morph);
    canny->detect(d_morph, d_edges);

    Mat gpu_final_edges;
    d_edges.download(gpu_final_edges);

    res.gpu_time = ((double)getTickCount() - t1) / getTickFrequency() * 1000;

    // Guardar resultados para comparación cualitativa
    imwrite(out_dir + "cpu_res_" + filename, cpu_edges);
    imwrite(out_dir + "gpu_res_" + filename, gpu_final_edges);

    return res;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Error: Proporcione la ruta de la imagen." << endl;
        return -1;
    }

    // Cargar imagen
    string path = argv[1];
    Mat img = imread(path);
    if (img.empty()) {
        cout << "Error: No se pudo cargar la imagen " << path << endl;
        return -1;
    }

    // Extraer solo el nombre del archivo de la ruta
    string filename = path.substr(path.find_last_of("/\\") + 1);

    // Asegurar directorio de salida
    system("mkdir -p images_out");

    cout << "\n===============================================" << endl;
    cout << "  BENCHMARK VISION POR COMPUTADOR (IEEE)       " << endl;
    cout << "  Imagen: " << filename << " [" << img.cols << "x" << img.rows << "]" << endl;
    cout << "===============================================" << endl;

    // Warmup inicial (Indispensable para métricas reales en GPU)
    cuda::GpuMat w; w.upload(img);

    // Ejecutar Pruebas
    BenchResult results = run_benchmark(img, filename);

    // Mostrar Métricas
    cout << fixed << setprecision(3);
    cout << "[RESULTADO] Tiempo CPU (i9):      " << results.cpu_time << " ms" << endl;
    cout << "[RESULTADO] Tiempo GPU (RTX 3070): " << results.gpu_time << " ms" << endl;
    cout << "[ANALISIS]  Speedup:               " << results.cpu_time / results.gpu_time << "x" << endl;
    cout << "-----------------------------------------------" << endl;
    cout << "Archivos guardados en carpeta: images_out/" << endl;

    return 0;
}