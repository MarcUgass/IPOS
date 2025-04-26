from PIL import Image, ImageFilter
import numpy as np
import sys # Importar sys para mensajes de error
import random

def procesar_imagen_por_pasos(
    imagen: Image.Image,
    ruido_intensidad: int = 30,
    sombra_general_intensidad: int = 30, # Intensidad para la primera sombra (mezcla)
    sombra_general_desplazamiento: tuple = (10, 10), # Desplazamiento para la primera sombra
    color_sombra_general: tuple = (0, 0, 0), # Color para la primera sombra
    sombra_general_blur_radius: int = 10, # Radio de desenfoque para la primera sombra
    shadow_region: tuple = None, # REQUERIDO para el paso de sombra regional
    shadow_color_regional: tuple = (0, 0, 0), # Color para la segunda sombra (regional)
    shadow_opacity_regional: int = 128, # Opacidad para la segunda sombra (regional, 0-255)
    regional_shadow_blur_radius: int = 5 # Radio de desenfoque para la segunda sombra (regional)
) -> Image.Image:
    """
    Procesa una imagen a través de una secuencia de pasos:
    1. Agrega ruido (similar a sal y pimienta, pero usando distribución normal).
    2. Agrega un efecto de sombra general desplazada usando mezcla.
    3. Agrega un efecto de sombra regional (drop shadow) usando composición alfa.

    Args:
        imagen_path: Ruta al archivo de imagen de entrada.
        ruido_intensidad: Intensidad del ruido a agregar.
        sombra_general_intensidad: Intensidad de la mezcla para la sombra general (0-100).
        sombra_general_desplazamiento: Desplazamiento (x, y) para la sombra general.
        color_sombra_general: Color de la sombra general (tupla RGB).
        sombra_general_blur_radius: Radio de desenfoque para la sombra general.
        shadow_region: Una tupla (x1, y1, x2, y2) definiendo la región para la
                       sombra regional. **REQUERIDO** para este paso.
        shadow_color_regional: Color de la sombra regional (tupla RGB).
        shadow_opacity_regional: Opacidad de la sombra regional (0-255).
        regional_shadow_blur_radius: Radio de desenfoque para la sombra regional.

    Returns:
        Un objeto PIL Image después de aplicar todos los pasos.

    Raises:
        ValueError: Si shadow_region es None o si la región es inválida.
        FileNotFoundError: Si imagen_path es inválida.
    """
    print("--- Iniciando procesamiento de imagen por pasos ---")

    # Paso 1: Cargar la imagen
    print("Paso 1: Cargando imagen...")
    try:
        #img = Image.open(imagen_path)
        img = imagen.convert('RGB') # Comenzamos en modo RGB
        print("Paso 1 completado.")
    except FileNotFoundError:
        print(f"Error: No se encontró la imagen en '{imagen_path}'.", file=sys.stderr)
        raise # Relanzar la excepción para que el llamador sepa que falló

    # Paso 2: Agregar ruido (similar a salt_and_pepper original)
    print("Paso 2: Agregando ruido...")
    img_array = np.array(img)

    # Generar ruido aleatorio con distribución normal
    ruido = np.random.normal(0, ruido_intensidad, img_array.shape)

    # Agregar ruido a la imagen y asegurar que los valores estén en el rango 0-255
    img_array_con_ruido = np.clip(img_array + ruido, 0, 255).astype(np.uint8)

    # Convertir el array de nuevo a imagen
    img_step2 = Image.fromarray(img_array_con_ruido)
    print("Paso 2 completado.")

    # Paso 3: Agregar sombra general desplazada (lógica de agregar_sombra original)
    print("Paso 3: Agregando sombra general desplazada...")
    # Trabajamos sobre el resultado del paso anterior
    current_img_step3 = img_step2.copy()

    # Crear una imagen de sombra del mismo tamaño
    sombra = Image.new('RGB', current_img_step3.size, color_sombra_general)

    # Aplicar desenfoque si el radio es positivo
    if sombra_general_blur_radius > 0:
        sombra = sombra.filter(ImageFilter.GaussianBlur(radius=sombra_general_blur_radius))

    # Crear una imagen para la sombra desplazada (fondo negro por defecto)
    sombra_desplazada = Image.new('RGB', current_img_step3.size, (0, 0, 0))
    # Pegar la sombra desenfocada en la posición desplazada
    sombra_desplazada.paste(sombra, (sombra_general_desplazamiento[0], sombra_general_desplazamiento[1]))

    # Mezclar la imagen actual con la sombra desplazada
    # Asegurarse de que la intensidad esté entre 0 y 1 para Image.blend
    blend_alpha = max(0, min(1, sombra_general_intensidad / 100))
    img_step3 = Image.blend(current_img_step3, sombra_desplazada, blend_alpha)
    print("Paso 3 completado.")

    # Paso 4: Agregar sombra regional (lógica de add_shadow original)
    print("Paso 4: Agregando sombra regional...")
    # Trabajamos sobre el resultado del paso anterior y convertimos a RGBA
    current_img_step4 = img_step3.convert("RGBA")

    if shadow_region is None:
        print("Error: El parámetro 'shadow_region' es obligatorio para el paso de sombra regional.", file=sys.stderr)
        raise ValueError("shadow_region parameter is required for the regional shadow step.")

    # Validar que la región esté dentro de los límites de la imagen
    img_width, img_height = current_img_step4.size
    x1, y1, x2, y2 = shadow_region
    if not (0 <= x1 < x2 <= img_width and 0 <= y1 < y2 <= img_height):
         print(f"Error: La región de sombra regional {shadow_region} está fuera de los límites de la imagen {current_img_step4.size}.", file=sys.stderr)
         raise ValueError(f"Invalid shadow_region: {shadow_region} for image size {current_img_step4.size}")

    # Crear una capa de sombra transparente del tamaño completo de la imagen
    shadow_layer = Image.new("RGBA", current_img_step4.size, (0, 0, 0, 0))

    # Crear la pieza de sombra para la región específica
    region_width = x2 - x1
    region_height = y2 - y1
    # Crear la pieza de sombra con el color y opacidad especificados
    shadow_piece_color = shadow_color_regional + (shadow_opacity_regional,)
    shadow_piece = Image.new("RGBA", (region_width, region_height), shadow_piece_color)

    # Aplicar desenfoque a la pieza de sombra si el radio es positivo
    if regional_shadow_blur_radius > 0:
        shadow_piece = shadow_piece.filter(ImageFilter.GaussianBlur(regional_shadow_blur_radius))

    # Pegar la pieza de sombra desenfocada en la capa de sombra principal en la posición correcta
    # Usamos la propia pieza de sombra como máscara para pegar con su canal alfa
    shadow_layer.paste(shadow_piece, (x1, y1), shadow_piece)

    # Componer la capa de sombra sobre la imagen actual usando composición alfa
    img_step4 = Image.alpha_composite(current_img_step4, shadow_layer)
    print("Paso 4 completado.")

    print("--- Procesamiento de imagen finalizado ---")
    return img_step4

def generar_parametros_aleatorios_procesamiento(image_size: tuple) -> dict:
    """
    Genera un diccionario de parámetros aleatorios para la función
    procesar_imagen_por_pasos, basándose en rangos especificados.

    Args:
        image_size: Una tupla (ancho, alto) del tamaño de la imagen para
                    calcular las coordenadas de la región de la sombra.

    Returns:
        Un diccionario donde las claves son los nombres de los parámetros
        y los valores son los números aleatorios generados dentro de los
        rangos especificados.
    """
    img_width, img_height = image_size

    # Generar coordenadas de región de sombra (porcentajes 0-1)
    # Aseguramos x1 < x2 y y1 < y2
    x1_perc, x2_perc = sorted([random.random(), random.random()])
    y1_perc, y2_perc = sorted([random.random(), random.random()])

    # Convertir porcentajes a coordenadas de píxeles enteros
    # Aseguramos que el ancho/alto mínimo de la región sea al menos 1 píxel
    region_x1 = int(img_width * x1_perc)
    region_y1 = int(img_height * y1_perc)
    region_x2 = int(img_width * x2_perc)
    region_y2 = int(img_height * y2_perc)

    # Ajustar si la región es demasiado pequeña después de la conversión a int
    if region_x2 == region_x1:
        region_x2 = min(region_x1 + 1, img_width)
    if region_y2 == region_y1:
        region_y2 = min(region_y1 + 1, img_height)
     # Doble check para evitar regiones inválidas en los bordes
    if region_x1 >= region_x2: region_x1 = max(0, region_x2 - 1)
    if region_y1 >= region_y2: region_y1 = max(0, region_y2 - 1)


    sombra_region_coords = (region_x1, region_y1, region_x2, region_y2)


    # Generar los demás parámetros aleatorios según los rangos del comentario
    ruido_intensidad_rand = random.randint(0, 25)              # Entre 0-25
    sombra_general_intensidad_rand = random.randint(0, 50)       # Entre 0-50
    desplazamiento_x = random.randint(0, 20)
    desplazamiento_y = random.randint(0, 20)
    sombra_general_desplazamiento_rand = (desplazamiento_x, desplazamiento_y) # Entre 0-20 (para X e Y)

    color_r = random.randint(45, 55)
    color_g = random.randint(45, 55)
    color_b = random.randint(45, 55)
    color_sombra_general_rand = (color_r, color_g, color_b)   # Entre 45-55 (para R, G, B)

    sombra_general_blur_radius_rand = random.randint(0, 10)    # Entre 0-10

    # El color regional era fijo (0,0,0) en el ejemplo, lo mantenemos fijo a menos que se especifique un rango
    shadow_color_regional_rand = (0, 0, 0)

    # La opacidad regional era 150 en el ejemplo. Opacidad va de 0 a 255.
    # Asumiremos que quieres un valor aleatorio en el rango completo de opacidad.
    shadow_opacity_regional_rand = random.randint(0, 255)      # Entre 0-255

    regional_shadow_blur_radius_rand = random.randint(0, 10)   # Entre 0-10

    # Empaquetar los parámetros en un diccionario
    parametros = {
        'ruido_intensidad': ruido_intensidad_rand,
        'sombra_general_intensidad': sombra_general_intensidad_rand,
        'sombra_general_desplazamiento': sombra_general_desplazamiento_rand,
        'color_sombra_general': color_sombra_general_rand,
        'sombra_general_blur_radius': sombra_general_blur_radius_rand,
        'shadow_region': sombra_region_coords, # Pasamos las coordenadas calculadas
        'shadow_color_regional': shadow_color_regional_rand,
        'shadow_opacity_regional': shadow_opacity_regional_rand,
        'regional_shadow_blur_radius': regional_shadow_blur_radius_rand
    }

    return parametros

# --- Ejemplo de cómo usar la función ---
if __name__ == "__main__":
    # Este bloque solo se ejecuta si corres este script directamente
    test_image_path = "resulto_page-0001.jpg"
    output_image_path_jpg = "imagen_procesada_final_random.jpg"

    try:
        # Paso 1: Cargar la imagen *antes* de generar los parámetros dependientes del tamaño
        print(f"Cargando imagen de entrada: {test_image_path}")
        img_input = Image.open(test_image_path)
        print("Imagen cargada exitosamente.")

        # Obtener el tamaño de la imagen cargada
        image_width, image_height = img_input.size

        # --- Generar los parámetros aleatorios usando la nueva función ---
        print("\nGenerando parámetros aleatorios...")
        parametros_aleatorios = generar_parametros_aleatorios_procesamiento(img_input.size)


        # --- Llamar a la función de procesamiento con los parámetros aleatorios ---
        print(f"\nProcesando imagen con parámetros aleatorios...")
        # Usamos '**' para desempaquetar el diccionario de parámetros y pasarlos
        # a la función procesar_imagen_por_pasos.
        imagen_procesada = procesar_imagen_por_pasos(
            img_input, # Pasar el objeto Image
            **parametros_aleatorios # Pasar todos los demás parámetros desde el diccionario
        )
        imagen_procesada.show()  # Muestra la imagen procesada

        # Guarda la imagen resultante
        background = Image.new('RGB', imagen_procesada.size, (255, 255, 255)) # Fondo blanco

        # Pegar la imagen RGBA sobre el fondo. Pillow usa el canal alfa para la mezcla.
        background.paste(imagen_procesada, (0, 0), imagen_procesada) # El 3er argumento 'imagen_procesada_rgba' es la máscara (canal alfa)

        # Ahora guardar la imagen resultante (que está en modo RGB) como JPG
        imagen_final_rgb = background
        imagen_final_rgb.save("output.jpg", quality=95) 
        
    except FileNotFoundError:
        print(f"\nError: La imagen '{test_image_path}' no fue encontrada. Asegúrate de que la ruta es correcta.", file=sys.stderr)
    except ValueError as e:
         print(f"\nError en los parámetros de procesamiento: {e}", file=sys.stderr)
    except Exception as e:
         print(f"\nOcurrió un error inesperado durante el ejemplo: {e}", file=sys.stderr)