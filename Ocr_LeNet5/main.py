import pandas as pd
from sklearn.model_selection import train_test_split

def create_smaller_csv(input_filename, output_filename, fraction_to_keep=0.1, stratify_col_index=0, random_state=42):
    """
    Crea un CSV más pequeño a partir de uno grande, opcionalmente de forma estratificada.

    Args:
        input_filename (str): Nombre del archivo CSV de entrada.
        output_filename (str): Nombre del archivo CSV de salida.
        fraction_to_keep (float): Fracción de datos a conservar (e.g., 0.1 para 10%).
        stratify_col_index (int): Índice de la columna a usar para estratificación (la columna de etiquetas).
                                  Si es None, no se hace estratificación.
        random_state (int): Semilla para reproducibilidad.
    """
    print(f"Leyendo {input_filename}...")
    try:
        # Intentar leer en chunks si el archivo es masivo y causa problemas de memoria al leerlo entero
        # Sin embargo, para EMNIST, pd.read_csv suele manejarlo si tienes RAM decente.
        # Si falla por memoria aquí, necesitarías leer en chunks o usar dask.
        df = pd.read_csv(input_filename, header=None)
    except FileNotFoundError:
        print(f"Error: Archivo {input_filename} no encontrado.")
        return
    except Exception as e:
        print(f"Error leyendo {input_filename}: {e}")
        return

    if fraction_to_keep >= 1.0:
        print("La fracción a mantener es >= 1.0, guardando el archivo original.")
        df.to_csv(output_filename, header=False, index=False)
        print(f"Archivo guardado como {output_filename}")
        return

    if stratify_col_index is not None and stratify_col_index < df.shape[1]:
        print(f"Submuestreando {fraction_to_keep*100:.1f}% de forma estratificada por la columna {stratify_col_index}.")
        # train_test_split puede usarse para obtener una muestra estratificada
        # Aquí, 'test_size' será 1 - fraction_to_keep
        # Necesitamos las etiquetas (y) y las características (X) separadas para stratify
        y = df.iloc[:, stratify_col_index]
        X = df.drop(columns=[stratify_col_index])

        # Si solo queremos una fracción, y no dividir en train/test, podemos hacer esto:
        # Usamos X_sampled, y_sampled. Descartamos X_unused, y_unused.
        X_sampled, _, y_sampled, _ = train_test_split(
            X, y,
            train_size=fraction_to_keep, # O test_size = 1.0 - fraction_to_keep si se quiere el resto
            stratify=y,
            random_state=random_state
        )
        # Volver a unir las etiquetas y las características
        sampled_df = pd.concat([y_sampled.reset_index(drop=True), X_sampled.reset_index(drop=True)], axis=1)

    else:
        print(f"Submuestreando {fraction_to_keep*100:.1f}% de forma aleatoria (sin estratificación).")
        sampled_df = df.sample(frac=fraction_to_keep, random_state=random_state)

    print(f"Guardando {len(sampled_df)} filas en {output_filename}...")
    sampled_df.to_csv(output_filename, header=False, index=False)
    print(f"Archivo más pequeño guardado como {output_filename}")

# --- USO ---
# Asegúrate de que este script esté en el mismo directorio que tus CSVs
# o proporciona las rutas completas.

# Para el conjunto de entrenamiento (estratificado por la primera columna, que es la etiqueta)
create_smaller_csv('emnist-byclass-train.csv', 'emnist-byclass-train-small.csv', fraction_to_keep=0.1, stratify_col_index=0)

# Para el conjunto de test (también estratificado si quieres mantener la distribución)
create_smaller_csv('emnist-byclass-test.csv', 'emnist-byclass-test-small.csv', fraction_to_keep=0.1, stratify_col_index=0)

print("\n¡Proceso completado! Ahora puedes modificar tu script 'model2.py' para usar los archivos '-small.csv'.")