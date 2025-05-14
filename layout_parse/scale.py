import math

# --- Constantes ---
POINTS_PER_INCH = 72.0

def convert_jpg_coord_to_pdf(
    x_px: float, y_px: float,
    image_width_px: int, image_height_px: int, image_ppi: float,
    pdf_x_offset_pt: float = 0.0, pdf_y_offset_pt: float = 0.0,
    scale_factor: float = 1.0
) -> tuple[float, float]:
    """
    Convierte un par de coordenadas (X, Y) del espacio de píxeles JPG
    (origen arriba-izquierda) al espacio de puntos PDF (origen abajo-izquierda).

    Args:
        x_px: La coordenada X en píxeles (desde el borde izquierdo de la imagen).
        y_px: La coordenada Y en píxeles (desde el borde superior de la imagen).
        image_width_px: Ancho total de la imagen JPG en píxeles.
        image_height_px: Alto total de la imagen JPG en píxeles.
        image_ppi: Resolución de la imagen en Píxeles Por Pulgada (PPI/DPI).
        pdf_x_offset_pt: El desplazamiento X de la esquina inferior izquierda de la
                           imagen en la página PDF, en puntos. Por defecto 0.
        pdf_y_offset_pt: El desplazamiento Y de la esquina inferior izquierda de la
                           imagen en la página PDF, en puntos. Por defecto 0.
        scale_factor: El factor de escala aplicado a la imagen en el PDF.
                      Por defecto 1.0 (sin escala adicional más allá del PPI).

    Returns:
        Una tupla (x_pdf, y_pdf) representando las coordenadas en puntos PDF.

    Raises:
        ValueError: Si image_ppi es cero o negativo.
    """
    if image_ppi <= 0:
        raise ValueError("La resolución de la imagen (PPI) debe ser positiva.")

    # 1. Calcular la coordenada Y desde el borde inferior de la imagen en píxeles
    #    (Esto invierte el eje Y para que coincida con la dirección del PDF)
    y_from_bottom_px = image_height_px - y_px

    # 2. Convertir las distancias en píxeles a puntos relativos a la esquina
    #    inferior izquierda de la imagen.
    #    Formula: (píxeles / (píxeles/pulgada)) * (puntos/pulgada) = puntos
    x_pt_relative = (x_px / image_ppi) * POINTS_PER_INCH
    y_pt_relative = (y_from_bottom_px / image_ppi) * POINTS_PER_INCH

    # 3. Aplicar el factor de escala (si la imagen fue redimensionada en el PDF)
    x_pt_scaled = x_pt_relative * scale_factor
    y_pt_scaled = y_pt_relative * scale_factor

    # 4. Añadir el desplazamiento (offset) de la imagen en la página PDF
    x_pdf = pdf_x_offset_pt + x_pt_scaled
    y_pdf = pdf_y_offset_pt + y_pt_scaled

    return x_pdf, y_pdf




"""
# --- Ejemplo de uso ---

# **¡IMPORTANTE! Define estos parámetros según tu imagen y PDF específicos.**
# Estos son solo valores de ejemplo. Debes obtenerlos de tu imagen y
# del proceso de creación del PDF.
IMG_WIDTH_PX = 1240  # Ancho de tu imagen JPG en píxeles
IMG_HEIGHT_PX = 1754 # Alto de tu imagen JPG en píxeles
IMG_PPI = 96       # Resolución de tu imagen JPG (puedes verla en propiedades/metadata)

# Dónde se colocó la imagen en la página PDF (esquina inferior izquierda de la imagen)
# Si la imagen ocupa toda la página PDF desde la esquina inferior izquierda, ambos son 0.
# Si hay márgenes o la imagen está centrada, estos valores cambiarán.
PDF_X_OFFSET_PT = 0.0  # Desplazamiento X en puntos
PDF_Y_OFFSET_PT = 0.0  # Desplazamiento Y en puntos

# Si la imagen fue redimensionada al insertarla en el PDF (aparte de su PPI)
SCALE_FACTOR = 1.0   # 1.0 si no hubo redimensionamiento manual

# Tu estructura de datos de ejemplo (coordenadas JPG en píxeles)
# Asumiendo x1,y1 es la esquina superior izquierda y x2,y2 la inferior derecha del cuadro delimitador
form = [
    {"x1": 100, "y1": 50, "x2": 300, "y2": 80, "text": "Campo 1", "type": "TEXT"},
    {"x1": 100, "y1": 100, "x2": 300, "y2": 130, "text": "Campo 2", "type": "TEXT"},
]

form_pdf = []
for item in form:
    # Extraer coordenadas JPG del cuadro delimitador
    x1_px = item["x1"]
    y1_px = item["y1"] # Borde SUPERIOR en JPG
    x2_px = item["x2"]
    y2_px = item["y2"] # Borde INFERIOR en JPG

    # Convertir la esquina SUPERIOR-IZQUIERDA del JPG (x1_px, y1_px)
    # Esto dará el (x_pdf, y_pdf) de la esquina SUPERIOR-IZQUIERDA en el PDF
    pdf_x1_calc, pdf_y2_calc = convert_jpg_coord_to_pdf(
        x1_px, y1_px,
        IMG_WIDTH_PX, IMG_HEIGHT_PX, IMG_PPI,
        PDF_X_OFFSET_PT, PDF_Y_OFFSET_PT, SCALE_FACTOR
    )

    # Convertir la esquina INFERIOR-DERECHA del JPG (x2_px, y2_px)
    # Esto dará el (x_pdf, y_pdf) de la esquina INFERIOR-DERECHA en el PDF
    pdf_x2_calc, pdf_y1_calc = convert_jpg_coord_to_pdf(
        x2_px, y2_px,
        IMG_WIDTH_PX, IMG_HEIGHT_PX, IMG_PPI,
        PDF_X_OFFSET_PT, PDF_Y_OFFSET_PT, SCALE_FACTOR
    )

    # Las coordenadas PDF para un cuadro delimitador suelen ser (x1, y1) para
    # la esquina INFERIOR-IZQUIERDA y (x2, y2) para la esquina SUPERIOR-DERECHA.
    # Asignamos los valores calculados a las variables finales correctas.
    x1_pdf = pdf_x1_calc  # Coordenada X izquierda
    y1_pdf = pdf_y1_calc  # Coordenada Y inferior
    x2_pdf = pdf_x2_calc  # Coordenada X derecha
    y2_pdf = pdf_y2_calc  # Coordenada Y superior

    # Asegurarse de que x1 < x2 y y1 < y2 (opcional, depende de cómo uses las coords)
    # x1_pdf = min(pdf_x1_calc, pdf_x2_calc)
    # y1_pdf = min(pdf_y1_calc, pdf_y2_calc)
    # x2_pdf = max(pdf_x1_calc, pdf_x2_calc)
    # y2_pdf = max(pdf_y1_calc, pdf_y2_calc)

    form_pdf.append({
        "x1": x1_pdf,
        "y1": y1_pdf,
        "x2": x2_pdf,
        "y2": y2_pdf,
        "text": item["text"],
        "type": item["type"]
    })

# Imprimir resultado para verificar
print("Coordenadas convertidas a PDF (en puntos):")
for item_pdf in form_pdf:
    print(item_pdf)

"""