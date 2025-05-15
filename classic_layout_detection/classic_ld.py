import cv2 as cv2
import numpy as np
from scipy import ndimage
import os
import json

def characters_bbox(image):
    binary_for_contours = image * 255
    binary_for_contours = binary_for_contours.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    characters_bbox = []
    for i, cnt in enumerate(contours):
        hull = cv2.convexHull(cnt)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(hull)
        characters_bbox.append(binary_for_contours[x_rect:(x_rect + w_rect), y_rect:(y_rect + h_rect)])
        
    text = "prueba"


    return text


def detect_layout_classic(input_image_path: str, output_json_path: str, output_debug_image_path: str | None = None) -> str | None:
    """
    Processes an image using classic computer vision techniques to find layout boxes.
    Saves detected box coordinates to a JSON file.
    Optionally saves a debug image with boxes.
    Returns path to JSON file if successful.
    """
    if not os.path.exists(input_image_path):
        print(f"Error ClassicLD: Input image not found at {input_image_path}")
        return None

    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error ClassicLD: Could not load image from {input_image_path}")
        return None

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_h, img_w = image.shape[:2]

    median_filter = ndimage.median_filter(img, size=30)

    filtered = img - median_filter

    binary = img > 150
    binary = np.abs(binary -1)

    dilatation_cont = ndimage.binary_dilation(binary, iterations=1)
    dilatation = ndimage.binary_dilation(dilatation_cont, iterations=1)

    kernel5 = np.array([[1], [1], [1], [1], [1]]) 
    filtered_image = dilatation.astype(np.uint8)
    dilated_image = cv2.dilate(filtered_image, kernel5, iterations=3)

    kernel2 = np.array([[1], [1]])
    dilated_image_2 = cv2.dilate(dilated_image, kernel2, iterations=8)

    kernel_horitzontal = np.array([[1, 1, 1, 1]])
    dilated = cv2.dilate(dilated_image_2, kernel_horitzontal, iterations=8)

    binary_for_contours = dilated * 255
    binary_for_contours = binary_for_contours.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output_image_bounding_boxes_filtered = image.copy()
    hull_coordinates_list = []

    for i, cnt in enumerate(contours):
        hull = cv2.convexHull(cnt)
        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(hull)
        hull_coordinates_list.append({
            "id": f"hull_{i}",
            "hull_points_array": hull.tolist(), # Para serialización JSON
            "x1": x_rect, "y1": y_rect,
            "x2": x_rect + w_rect, "y2": y_rect + h_rect,
            "w": w_rect, "h": h_rect, "area": w_rect * h_rect
        })

    filtered_hull_coordinates_list_for_json = [] #

    if not hull_coordinates_list:
        print("ClassicLD: No hulls found to process.")
    else:
        last_x1 = hull_coordinates_list[0]["x1"]
        last_y1 = hull_coordinates_list[0]["y1"]
        last_x2 = hull_coordinates_list[0]["x2"]
        last_y2 = hull_coordinates_list[0]["y2"]

    #bbox2 = [] # No se usa para el JSON de salida principal

    for hull_info in hull_coordinates_list:
        w = hull_info["w"]
        h = hull_info["h"]
        # area = hull_info["area"] # No se usa en el if
        x1 = fx1 = hull_info["x1"]
        y1 = fy1 = hull_info["y1"]
        x2 = fx2 = hull_info["x2"]
        y2 = fy2 = hull_info["y2"]

        # aspect_ratio = float('inf') # No se usa
        # if h > 0:
        #     aspect_ratio = w / h

        is_marginal = (x1 < 15 or y1 < 15 or x2 > (img_w - 15) or y2 > (img_h - 15))
        is_last = 1 # Esta lógica necesita ser revisada, siempre es True

        if not is_marginal:
            if is_last == 1: # Siempre True con la lógica actual
                # bbox_binary = dilatation[fy1:fy2, fx1:fx2] # Esto es para OCR, no para el layout JSON
                # bbox = bbox_binary.astype(np.uint8) * 255
                # bbox2.append(bbox)
                # contours_bbox, _ = cv2.findContours(bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # len_contours = len(contours_bbox)

                # Adaptación para el JSON:
                # Necesitarías convertir esto al formato que espera `generator.py`
                # Aquí simplemente guardamos la info del bounding box.
                # El "text" y "type" no se extraen aquí.
                """
                 items antes:
                    "id": hull_info['id'], # Usar un ID único
                    "type": "unknown_classic", # Tipo genérico, ya que no lo clasificamos
                    "text": "", # No hay OCR en este script
                    "position": [x1, y1, x2, y2], # [x1, y1, x2, y2]
                    # "len_contours": len_contours # Información de debug
                """
                bbox = image[fy1:fy2, fx1:fx2]
                #cv2.imwrite('out/chracter' + hull_info["id"] + '.jpeg', bbox)

                #recordar: del bbox de sortida descartar coses rares per poder fer la separacio de caracters
                out_text = "Prueba" #characters_bbox(bbox)
                item_for_json = {
                    "type": "text", # Asumimos todo es texto para ReportLab por ahora
                    "text": out_text,
                    "font": "Helvetica", # Default, podría ser configurable
                    "size": 10,          # Default
                    "color": [0, 0, 0],  # Default
                    "position": [x1, y1, x2, y2],
                    "align": "left",     # Default
                    "valign": "top"      # Default
                }
                filtered_hull_coordinates_list_for_json.append(item_for_json)
                cv2.rectangle(output_image_bounding_boxes_filtered, (fx1, fy1), (fx2, fy2), (0, 0, 255), 1)
                
            # La parte 'else' de 'is_last' de tu código original parece que intentaba fusionar,
            # pero tenía un error al hacer pop() sin verificar si la lista estaba vacía
            # y luego append() sin la info de 'len_contours'.
            # Esta lógica de fusión es compleja y la omito por simplicidad aquí.
            # Deberías revisarla cuidadosamente.

        last_x1, last_y1, last_x2, last_y2 = x1, y1, x2, y2
    # --- FIN DE TU LÓGICA DE FILTRADO ---

    # Formato para generator.py: un diccionario donde las claves son strings de índices
    output_data_for_json = {
        str(i): item for i, item in enumerate(filtered_hull_coordinates_list_for_json)
    }

    
    
    try:
        output_dir = os.path.dirname(output_json_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(output_data_for_json, f, indent=4)
            cv2.imwrite(output_dir + 'bounding_box.jpeg', output_image_bounding_boxes_filtered)
        print(f"ClassicLD: Layout data saved to {output_json_path}")
    except IOError as e:
        print(f"Error ClassicLD: Could not write JSON to {output_json_path}: {e}")
        return None

    if output_debug_image_path:
        debug_dir = os.path.dirname(output_debug_image_path)
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir, exist_ok=True)
        if not cv2.imwrite(output_debug_image_path, output_image_bounding_boxes_filtered):
            print(f"ClassicLD: Failed to save debug image to {output_debug_image_path}")
        else:
            print(f"ClassicLD: Debug image saved to {output_debug_image_path}")

    return output_json_path