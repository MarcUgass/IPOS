# layout_parser_module/detector_lp.py
import layoutparser as lp
import cv2
import numpy as np
# import matplotlib.pyplot as plt # No para guardar, usar cv2.imwrite o similar
import json
import os

# Variable global para el modelo para no cargarlo cada vez si se llama múltiples veces
# Podrías pasar el modelo como argumento o hacerlo parte de una clase si es más complejo.
LP_MODEL = None

def initialize_lp_model():
    global LP_MODEL
    if LP_MODEL is None:
        print("LayoutParser: Initializing Detectron2 model...")
        try:
            LP_MODEL = lp.models.detectron2.Detectron2LayoutModel(
                'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"},
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.2] # Umbral de confianza
            )
            print("LayoutParser: Model initialized.")
        except Exception as e:
            print(f"LayoutParser: Error initializing model: {e}")
            LP_MODEL = "Error" # Para evitar reintentos fallidos
    return LP_MODEL

def detect_layout_lp(input_image_path: str, output_json_path: str, output_debug_image_path: str | None = None) -> str | None:
    """
    Detects layout using LayoutParser, extracts text with Tesseract, and saves to JSON.
    Returns path to JSON file if successful.
    """
    global LP_MODEL
    if LP_MODEL is None:
        initialize_lp_model()
    if LP_MODEL == "Error": # Si la inicialización falló
        return None

    if not os.path.exists(input_image_path):
        print(f"Error LP: Input image not found at {input_image_path}")
        return None

    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error LP: Could not load image from {input_image_path}")
        return None

    # LayoutParser espera imagen en RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("LayoutParser: Detecting layout...")
    layout = LP_MODEL.detect(image_rgb) # Aquí image_rgb

    if output_debug_image_path:
        # Para dibujar, LayoutParser también espera RGB, pero cv2.imwrite guardará BGR
        # así que lo convertimos de nuevo si es necesario.
        img_for_drawing = image_rgb.copy()
        img_with_boxes = lp.draw_box(img_for_drawing, layout, box_width=3)
        debug_dir = os.path.dirname(output_debug_image_path)
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
        if not cv2.imwrite(output_debug_image_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)):
            print(f"LayoutParser: Failed to save debug image to {output_debug_image_path}")
        else:
            print(f"LayoutParser: Debug image saved to {output_debug_image_path}")


    text_blocks = lp.Layout([b for b in layout if (b.type=='Text' or b.type=='Title')])
    # figure_blocks = lp.Layout([b for b in layout if b.type=='Figure']) # No se usa en tu código

    form_data = []
    print("LayoutParser: Initializing Tesseract OCR agent...")
    try:
        ocr_agent = lp.TesseractAgent(languages='eng') # o el idioma que necesites
    except Exception as e:
        print(f"LayoutParser: Error initializing TesseractAgent: {e}")
        print("Ensure Tesseract OCR is installed and in your PATH, and language data is available.")
        return None

    print("LayoutParser: Performing OCR on text blocks...")
    for i, block in enumerate(text_blocks):
        x1 = int(block.block.x_1)
        y1 = int(block.block.y_1)
        x2 = int(block.block.x_2)
        y2 = int(block.block.y_2)
        block_type = block.type

        # Crop image (LayoutParser usa la imagen original que le pasaste, que era RGB)
        segment_image_rgb = (block
                           .pad(left=5, right=5, top=5, bottom=5)
                           .crop_image(image_rgb)) # Usa image_rgb aquí

        text = ocr_agent.detect(segment_image_rgb, return_response=False) # Pasa segment_image_rgb
        
        # Tu ajuste de coordenadas original:
        # "x1": (x1+5)/2, "y1": (y1-5)/2, "x2": (x2+5)/2, "y2": y2/1.2,
        # Estas coordenadas parecen específicas para alguna transformación que no está clara.
        # Para ReportLab, normalmente se usan las coordenadas directas (x1, y1, x2, y2)
        # donde y1 es desde ABAJO. Tu flip_y en generator.py lo maneja.
        # Mantendré tus coordenadas por ahora, pero considera si son las correctas.
        form_data.append({
            "id": f"lp_block_{i}",
            "x1": (x1 + 5) / 2, # Estas coordenadas podrían necesitar ajuste
            "y1": (y1 - 5) / 2, # para que coincidan con las expectativas de generate.py
            "x2": (x2 + 5) / 2, # respecto al origen y unidades.
            "y2": y2 / 1.2,
            "text": text if text else "", # Asegurar que text no sea None
            "type": block_type.lower() # 'text' o 'title'
        })

    # Convertir al formato que espera generator.py
    converted_form_for_json = {}
    for idx, item in enumerate(form_data):
        converted_form_for_json[str(idx)] = {
            "type": "text", # Asumimos todo es texto para ReportLab por ahora
            "text": item["text"],
            "font": "Helvetica", # Default, podría ser configurable
            "size": 10,          # Default
            "color": [0, 0, 0],  # Default
            "position": [item["x1"], item["y1"], item["x2"], item["y2"]],
            "align": "left",     # Default
            "valign": "top"      # Default
        }

    try:
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(converted_form_for_json, f, ensure_ascii=False, indent=4)
        print(f"LayoutParser: Layout and OCR data saved to {output_json_path}")
        return output_json_path
    except IOError as e:
        print(f"Error LP: Could not write JSON to {output_json_path}: {e}")
        return None