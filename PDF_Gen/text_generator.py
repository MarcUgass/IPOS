import random
import json
import math

# Tamaño A4 en puntos (portrait)
PAGE_WIDTH = 595
PAGE_HEIGHT = 842

FONTS = ["Helvetica", "Times-Roman"]
FONT_SIZE = {
    "Helvetica": 12,
    "Times-Roman": 12
}
FONT_WIDTH_ESTIMATES = {
    "Helvetica": 0.5,  # Estimación del ancho de un carácter en Helvetica
    "Times-Roman": 0.6  # Estimación del ancho de un carácter en Times-Roman
}

# Cargar el texto de Don Quijote
with open("el_quijote.txt", "r", encoding="utf-8") as f:
    DONQUIJOTE_TEXT = f.read().replace("\n", " ")  # quitar saltos de línea

# Obtener un fragmento aleatorio del texto
def get_random_fragment(min_length=100, max_length=300):
    start = random.randint(0, len(DONQUIJOTE_TEXT) - max_length - 1)
    end = start + random.randint(min_length, max_length)
    fragment = DONQUIJOTE_TEXT[start:end]
    
    # Cortar por última palabra si es posible
    if "." in fragment:
        fragment = fragment.rsplit(".", 1)[0] + "."
    return fragment.strip()

# Calcular el ancho del texto basándose en el número de caracteres
def calculate_text_width(text, font, font_size):
    # Estimación del ancho de cada carácter (en puntos)
    char_width = FONT_WIDTH_ESTIMATES[font] * font_size
    return len(text) * char_width

# Generar bloque de texto con tamaño y posición adaptados
def generate_random_text_block(block_id):
    # Seleccionar fragmento aleatorio
    fragment = get_random_fragment()
    
    # Seleccionar fuente y tamaño
    font = random.choice(FONTS)
    font_size = FONT_SIZE[font]
    
    # Calcular el ancho del texto
    text_width = calculate_text_width(fragment, font, font_size)
    
    # Ajustar la posición y el tamaño del recuadro para que el texto quepa
    x1 = random.randint(10, PAGE_WIDTH - 250)
    y1 = random.randint(10, PAGE_HEIGHT - 60)
    
    # Limitar el ancho del recuadro para que el texto no se salga de la página
    width = min(text_width + 20, PAGE_WIDTH - x1 - 10)
    
    # Estimar la altura del recuadro según el número de líneas que ocupe el texto
    max_chars_per_line = width // (FONT_WIDTH_ESTIMATES[font] * font_size)
    num_lines = math.ceil(len(fragment) / max_chars_per_line)
    height = num_lines * font_size + 10
    
    # Asegurar que la altura del recuadro no se salga de la página
    x2 = x1 + width
    y2 = min(y1 + height, PAGE_HEIGHT - 10)

    return {
        str(block_id): {
            "type": "text",
            "text": fragment,
            "font": font,
            "size": font_size,
            "color": [0,0,0],
            "position": [x1, y1, x2, y2],
            "align": "left",  # Alineación siempre a la izquierda
            "valign": "top"   # Alineación siempre en la parte superior
        }
    }

def generate_single_block():
    data = {}
    block = generate_random_text_block(0)  # Solo un bloque
    data.update(block)
    return data

if __name__ == "__main__":
    result = generate_single_block()  # Genera un único bloque
    with open("text_blocks.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print("Un único bloque generado en 'single_text_block.json'")
