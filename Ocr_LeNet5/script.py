import pandas as pd
import os

def eliminar_mitad_filas(ruta_archivo_csv):
    """
    Elimina la mitad de las filas de un archivo CSV.

    Args:
        ruta_archivo_csv (str): La ruta al archivo CSV.
    """
    try:
        # Lee el archivo CSV en un DataFrame de pandas
        df = pd.read_csv(ruta_archivo_csv)

        # Calcula el número de filas a conservar (la mitad)
        num_filas = len(df)
        num_filas_a_conservar = num_filas // 2

        # Selecciona las primeras 'num_filas_a_conservar' filas
        df_reducido = df.head(num_filas_a_conservar)

        # Construye la nueva ruta del archivo (añadiendo '_reducido')
        nombre_base, extension = os.path.splitext(ruta_archivo_csv)
        ruta_archivo_reducido = f"{nombre_base}_reducido{extension}"

        # Guarda el DataFrame reducido en un nuevo archivo CSV
        df_reducido.to_csv(ruta_archivo_reducido, index=False)

        print(f"Se ha creado un nuevo archivo: '{ruta_archivo_reducido}' con la mitad de las filas.")

    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en la ruta: '{ruta_archivo_csv}'")
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    ruta_archivo = input("Por favor, introduce la ruta completa de tu archivo CSV: ")
    eliminar_mitad_filas(ruta_archivo)