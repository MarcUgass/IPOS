import unicodedata

def normalizar_texto(texto):
    # Normaliza el texto (NFKD separa los acentos) y elimina los caracteres no ASCII
    texto_normalizado = unicodedata.normalize('NFKD', texto)
    return ''.join(c for c in texto_normalizado if not unicodedata.combining(c))

def limpiar_archivo(archivo_entrada, archivo_salida):
    with open(archivo_entrada, 'r', encoding='utf-8', errors='replace') as f:
        contenido = f.read()
    
    contenido_limpio = normalizar_texto(contenido)

    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write(contenido_limpio)

# Uso
archivo_original = 'el_quijote.txt'
archivo_limpio = 'el_quijote2.txt'
limpiar_archivo(archivo_original, archivo_limpio)
