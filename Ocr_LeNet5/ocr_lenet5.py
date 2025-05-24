import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps # Pillow para manipulación de imágenes
import argparse # Para argumentos de línea de comandos
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


EMNIST_BYCLASS_MAPPING = {i: str(i) for i in range(10)} # 0-9
for i in range(26):
    EMNIST_BYCLASS_MAPPING[10 + i] = chr(ord('A') + i) # A-Z
    EMNIST_BYCLASS_MAPPING[10 + 26 + i] = chr(ord('a') + i) # a-z

# --- Definición de la Arquitectura del Modelo LeNet5PyTorch (debe ser idéntica) ---
class LeNet5PyTorch(nn.Module):
    def __init__(self, num_classes=62):
        super(LeNet5PyTorch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Nueva Función de Preprocesamiento para una imagen externa ---
def preprocess_external_image(img, mean_px, std_px, invert=False):
    """
    Carga, redimensiona, preprocesa y normaliza una imagen externa para el modelo EMNIST.
    Args:
        image_path (str): Ruta al archivo de imagen.
        mean_px (float): Media de píxeles del conjunto de entrenamiento.
        std_px (float): Desviación estándar de píxeles del conjunto de entrenamiento.
        invert (bool): Si es True, invierte los colores de la imagen (para pasar de negro sobre blanco a blanco sobre negro).
    Returns:
        torch.Tensor: Tensor de imagen listo para el modelo.
    """
    # Redimensionar a 28x28. Usar ANTIALIAS para mejor calidad al reducir.
    # Pillow >= 9.0.0 usa Resampling.LANCZOS en lugar de ANTIALIAS
    try:
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
    except AttributeError: # Para versiones más antiguas de Pillow
        img = img.resize((28, 28), Image.ANTIALIAS)


    # Convertir a NumPy array
    img_np = np.array(img).astype(np.float32)

    # Invertir colores si es necesario (EMNIST espera caracteres brillantes sobre fondo oscuro)
    if invert:
        img_np = 255.0 - img_np

    # Visualización de la imagen DESPUÉS de redimensionar e invertir (opcional)
    plt.imshow(img_np, cmap='gray')
    plt.title("Imagen después de redimensionar e invertir (si aplica)")
    plt.show()


    # Aplicar la misma rotación y volteo que en el entrenamiento de EMNIST
    # Esto es crucial porque las imágenes EMNIST originales están rotadas/volteadas
    # La función 'preprocess_emnist_images' en model2.py hacía:
    # img = np.rot90(img, k=1) # Rotar 90 grados en sentido antihorario
    # img = np.fliplr(img)     # Voltear horizontalmente
    # Aplicamos lo mismo aquí:
    img_np_processed = np.rot90(img_np, k=2)
    #img_np_processed = np.fliplr(img_np)

    
    # Visualización de la imagen DESPUÉS de la rotación/volteo de EMNIST (opcional)
    plt.imshow(img_np_processed, cmap='gray')
    plt.title("Imagen después de rotación/volteo EMNIST")
    plt.show()

    # Añadir dimensión de canal y lote (N, C, H, W) -> (1, 1, 28, 28)
    # La normalización se hace sobre esta forma
    reshaped_image_np = img_np_processed[np.newaxis, np.newaxis, :, :]

    # Normalización usando los parámetros del entrenamiento
    normalized_image_np = (reshaped_image_np - mean_px) / (std_px + 1e-6)
    
    return torch.from_numpy(normalized_image_np).float()


def predict_single_image(model, image_tensor, device):
    """
    Realiza una predicción en un tensor de imagen preprocesado.
    """
    model.eval() # Poner el modelo en modo evaluación
    with torch.no_grad(): # No calcular gradientes
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1) # Obtener probabilidades
        confidence, predicted_idx = torch.max(probabilities, 1)
    return predicted_idx.item(), confidence.item()


def ocr_lenet5(input_image, invert=True):
    MODEL_LOAD_PATH = os.path.join(CURRENT_DIR,"lenet5_emnist_pytorch.pth")
    PREPROCESSING_PARAMS_PATH = os.path.join(CURRENT_DIR, "preprocessing_params.npz")
    NUM_CLASSES = 62 # EMNIST ByClass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo para predicción: {device}")

    # --- Cargar Parámetros de Preprocesamiento ---
    try:
        params = np.load(PREPROCESSING_PARAMS_PATH)
        mean_px_loaded = params['mean_px']
        std_px_loaded = params['std_px']
    except FileNotFoundError:
        print(f"Error: Archivo de parámetros de preprocesamiento '{PREPROCESSING_PARAMS_PATH}' no encontrado.")
        print("Asegúrate de haber entrenado el modelo (model2.py) y guardado los parámetros primero.")
        exit()
    
    print(f"Parámetros de preprocesamiento cargados: mean={mean_px_loaded:.4f}, std={std_px_loaded:.4f}")

    # --- Cargar el  ---
    loaded_model = LeNet5PyTorch(num_classes=NUM_CLASSES)
    try:
        loaded_model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Archivo del modelo '{MODEL_LOAD_PATH}' no encontrado.")
        print("Asegúrate de haber entrenado y guardado el modelo primero.")
        exit()
    except Exception as e:
        print(f"Error cargando el estado del modelo: {e}")
        exit()

    loaded_model.to(device)
    loaded_model.eval()
    print("Modelo cargado y configurado para evaluación.")

    # --- Preprocesar la imagen de entrada ---
    input_tensor = preprocess_external_image(input_image, mean_px_loaded, std_px_loaded, invert=invert)

    if input_tensor is None:
        exit() # Error durante el preprocesamiento

    # --- Realizar la predicción ---
    predicted_class_index, confidence = predict_single_image(loaded_model, input_tensor, device)
    
    predicted_char = EMNIST_BYCLASS_MAPPING.get(predicted_class_index, "Índice Desconocido")

    print(f"\n--- Predicción ---")
    print(f"Índice de clase predicho: {predicted_class_index}")
    print(f"Carácter predicho: {predicted_char}")
    print(f"Confianza: {confidence*100:.2f}%")

    return predicted_char


"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predecir un carácter EMNIST desde una imagen.")
    parser.add_argument("image_path", type=str, help="Ruta al archivo de imagen del carácter.")
    parser.add_argument("--invert", action="store_true", help="Invertir colores de la imagen (usar si es negro sobre blanco).")
    args = parser.parse_args()

    # --- Configuración ---
    MODEL_LOAD_PATH = "lenet5_emnist_pytorch.pth"
    PREPROCESSING_PARAMS_PATH = "preprocessing_params.npz"
    NUM_CLASSES = 62 # EMNIST ByClass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo para predicción: {device}")

    # --- Cargar Parámetros de Preprocesamiento ---
    try:
        params = np.load(PREPROCESSING_PARAMS_PATH)
        mean_px_loaded = params['mean_px']
        std_px_loaded = params['std_px']
    except FileNotFoundError:
        print(f"Error: Archivo de parámetros de preprocesamiento '{PREPROCESSING_PARAMS_PATH}' no encontrado.")
        print("Asegúrate de haber entrenado el modelo (model2.py) y guardado los parámetros primero.")
        exit()
    
    print(f"Parámetros de preprocesamiento cargados: mean={mean_px_loaded:.4f}, std={std_px_loaded:.4f}")

    # --- Cargar el Modelo ---
    loaded_model = LeNet5PyTorch(num_classes=NUM_CLASSES)
    try:
        loaded_model.load_state_dict(torch.load(MODEL_LOAD_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Error: Archivo del modelo '{MODEL_LOAD_PATH}' no encontrado.")
        print("Asegúrate de haber entrenado y guardado el modelo primero.")
        exit()
    except Exception as e:
        print(f"Error cargando el estado del modelo: {e}")
        exit()

    loaded_model.to(device)
    loaded_model.eval()
    print("Modelo cargado y configurado para evaluación.")

    # --- Preprocesar la imagen de entrada ---
    input_tensor = preprocess_external_image(args.image_path, mean_px_loaded, std_px_loaded, invert=args.invert)

    if input_tensor is None:
        exit() # Error durante el preprocesamiento

    # --- Realizar la predicción ---
    predicted_class_index, confidence = predict_single_image(loaded_model, input_tensor, device)
    
    predicted_char = EMNIST_BYCLASS_MAPPING.get(predicted_class_index, "Índice Desconocido")

    print(f"\n--- Predicción ---")
    print(f"Ruta de la imagen: {args.image_path}")
    print(f"Índice de clase predicho: {predicted_class_index}")
    print(f"Carácter predicho: {predicted_char}")
    print(f"Confianza: {confidence*100:.2f}%")

    # Visualizar la imagen final que entra al modelo (después de toda la preparación)
    final_input_view = input_tensor.cpu().numpy().squeeze()
    plt.imshow(final_input_view, cmap='gray')
    plt.title(f"Entrada final al modelo\nPredicción: {predicted_char} ({confidence*100:.2f}%)")
    plt.show()
    

    """