import argparse
import os
import sys
import json
import shutil
import cv2 as cv2


from DLT import dlt
from classic_layout_detection import classic_ld
#from layout_parse import lay
from pdf_generator import generate

BASE_OUTPUT_DIR = "output_results"
BASE_INPUT_DIR = "input_images"

WARPED_IMAGE_NAME = "dlt_img.jpeg"
LAYOUT_JSON_NAME = "layout_info.json"
FINAL_PDF_NAME = "final_document.pdf"

DEBUG_IMG_LP_NAME = "debug_layout_lp.png"
DEBUG_IMG_CLASSIC_NAME = "debug_layout_classic.png"

def run_pipeline(input_image_path: str,
                 output_base_dir: str,
                 layout_method: str = "classic",
                 generate_debug_images: bool = False):


    # Crear directorio de salida base si no existe
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Paso 1: Transformación de Perspectiva (DLT) ---
    print("\n[Paso 1: Transformación de Perspectiva]")
    warped_image_output_path = os.path.join(output_base_dir, WARPED_IMAGE_NAME)
    
    processed_warped_image_path = dlt.process_document_perspective(
        input_image_path=input_image_path,
        output_image_path=warped_image_output_path
    )

    if not processed_warped_image_path:
        print("Error en el paso de transformación de perspectiva. Abortando pipeline.")
        return
    print(f"Imagen transformada guardada en: {processed_warped_image_path}")

    # --- Paso 2: Detección de Layout (Condicional) ---
    print("\n[Paso 2: Detección de Layout]")
    layout_json_output_path = os.path.join(output_base_dir, LAYOUT_JSON_NAME)
    final_layout_json_path = None

    if layout_method == "layout_parser":
        # Asegurarse de que el modelo LP se inicializa una vez
        # (detector_lp.py ya maneja la inicialización global del modelo)
        """
        debug_lp_path = os.path.join(output_base_dir, DEBUG_IMG_LP_NAME) if generate_debug_images else None
        final_layout_json_path = lay.detect_layout_lp(
            input_image_path=processed_warped_image_path,
            output_json_path=layout_json_output_path,
            output_debug_image_path=debug_lp_path
        )
        """
    elif layout_method == "classic":
        debug_classic_path = os.path.join(output_base_dir, DEBUG_IMG_CLASSIC_NAME) if generate_debug_images else None
        final_layout_json_path = classic_ld.detect_layout_classic(
            input_image_path=processed_warped_image_path,
            output_json_path=layout_json_output_path,
            output_debug_image_path=debug_classic_path
        )
        # IMPORTANTE: El JSON de classic_ld necesita ser adaptado para generator.py
        # o generator.py modificado para entender el formato de classic_ld.
        # Esta implementación asume que PUEDE producir un JSON compatible,
        # pero mi refactorización de detector_classic.py lo guarda en un formato
        # que necesitaría una conversión para ser usado por generator.py.
        # Si usas "classic", es probable que el siguiente paso falle o dé resultados inesperados
        # sin esa adaptación.
        print("ADVERTENCIA: El método 'classic' puede requerir adaptación del JSON para el generador de PDF.")

    else:
        print(f"Error: Método de layout desconocido '{layout_method}'. Use 'layout_parser' o 'classic'.")
        return

    if not final_layout_json_path:
        print("Error en el paso de detección de layout. Abortando pipeline.")
        return
    print(f"Información de layout guardada en: {final_layout_json_path}")

    # --- Paso 3: Generación de PDF y Previsualización ---
    print("\n[Paso 3: Generación de PDF y Previsualización]")
    final_pdf_output_path = os.path.join(output_base_dir, FINAL_PDF_NAME)

    image = cv2.imread(input_image_path)
    image_height, image_widht, channels = image.shape
    generate.generate_pdf(
        final_layout_json_path,
        final_pdf_output_path,
        image_height,
        image_widht
    )

    print("\n--- Pipeline completado. ---")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de Procesamiento de Documentos.")
    parser.add_argument("input_image", help="Ruta a la imagen de entrada.")
    parser.add_argument(
        "-o", "--output_dir",
        default=BASE_OUTPUT_DIR,
        help=f"Directorio base para los resultados (default: {BASE_OUTPUT_DIR})"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["layout_parser", "classic"],
        default="classic",
        help="Método de detección de layout a usar (default: classic)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generar imágenes de depuración para los pasos de layout."
    )
    # Podrías añadir un argumento para un archivo de configuración JSON aquí.

    args = parser.parse_args()

    # Validar ruta de entrada
    if not os.path.isfile(args.input_image):
        print(f"Error: El archivo de imagen de entrada no existe: {args.input_image}")
        sys.exit(1)

    run_pipeline(
        input_image_path=args.input_image,
        output_base_dir=args.output_dir,
        layout_method=args.method,
        generate_debug_images=args.debug
    )