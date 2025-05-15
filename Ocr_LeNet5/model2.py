import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split # Para dividir train en train/val si es necesario

# --- Carga de Datos ---
# Asumiendo que los CSV no tienen header. Si lo tienen, ajusta 'header=None'.
try:
    train_df = pd.read_csv('emnist-byclass-train_reducido.csv', header=None)
    test_df = pd.read_csv('emnist-byclass-test_reducido.csv', header=None)
except FileNotFoundError:
    print("Error: Asegúrate de que 'emnist-byclass-train.csv' y 'emnist-byclass-test.csv' están en el mismo directorio que el script.")
    exit()


Y_train_pd = train_df.iloc[:,0]
X_train_pd = train_df.iloc[:,1:]
Y_test_pd = test_df.iloc[:,0]
X_test_pd = test_df.iloc[:,1:]

# --- Visualización (opcional, pero útil para EMNIST) ---
sample_vis = np.array(X_train_pd.iloc[10, :])
sample_vis = sample_vis.reshape([28,28])
# Para rotar y voltear la imagen a la orientación correcta para EMNIST
sample_vis = np.fliplr(np.rot90(sample_vis, k=3))
plt.imshow(sample_vis, cmap='gray')
plt.title("Ejemplo de Imagen EMNIST (orientación corregida para visualización)")
plt.show()

# Convertir a NumPy arrays
X_train_np = X_train_pd.values.astype(np.float32)
X_test_np = X_test_pd.values.astype(np.float32)
Y_train_np = Y_train_pd.values.astype(np.int64) # CrossEntropyLoss espera LongTensor para las etiquetas
Y_test_np = Y_test_pd.values.astype(np.int64)

# --- Preprocesamiento y Reshape para PyTorch ---
# PyTorch espera (N, C, H, W)
# EMNIST las imágenes vienen como filas, hay que rotarlas y transponerlas para que el entrenamiento sea efectivo.
# La rotación es para que el carácter esté "de pie" como se espera.
def preprocess_emnist_images(images_np):
    num_images = images_np.shape[0]
    processed_images = np.empty((num_images, 1, 28, 28), dtype=np.float32)
    for i in range(num_images):
        img = images_np[i].reshape(28, 28)
        img = np.rot90(img, k=1) # Rotar 90 grados en sentido antihorario
        img = np.fliplr(img)     # Voltear horizontalmente
        processed_images[i, 0, :, :] = img
    return processed_images

X_train_processed_np = preprocess_emnist_images(X_train_np)
X_test_processed_np = preprocess_emnist_images(X_test_np)


# Normalización (basada en los datos de entrenamiento preprocesados)
mean_px = X_train_processed_np.mean()
std_px = X_train_processed_np.std()
X_train_norm_np = (X_train_processed_np - mean_px) / (std_px + 1e-6) # Añadir epsilon para evitar división por cero
X_test_norm_np = (X_test_processed_np - mean_px) / (std_px + 1e-6)

# Convertir a Tensores de PyTorch
X_train_tensor = torch.from_numpy(X_train_norm_np)
Y_train_tensor = torch.from_numpy(Y_train_np)
X_test_tensor = torch.from_numpy(X_test_norm_np)
Y_test_tensor = torch.from_numpy(Y_test_np)

# Crear DataLoaders
batch_size = 32 # Puedes ajustar esto
print("batch size")
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# --- Definición del Modelo LeNet-5 en PyTorch ---
class LeNet5PyTorch(nn.Module):
    def __init__(self, num_classes=62):
        super(LeNet5PyTorch, self).__init__()
        # Capa 1
        # Entrada: (N, 1, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        # Salida conv1: (N, 6, 24, 24) (28-5+1 = 24)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Salida pool1: (N, 6, 12, 12)

        # Capa 2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # Salida conv2: (N, 16, 8, 8) (12-5+1 = 8)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Salida pool2: (N, 16, 4, 4)

        # Flatten
        self.flatten = nn.Flatten()
        # Salida flatten: (N, 16*4*4) = (N, 256)

        # Capa 3: Totalmente conectada
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.relu3 = nn.ReLU()

        # Capa 4: Totalmente conectada
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.relu4 = nn.ReLU()

        # Capa 5: Salida
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)
        # Nota: Softmax no se aplica aquí si se usa CrossEntropyLoss, ya que lo incluye.

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x) # Salida de logits
        return x

# --- Instanciar el modelo, definir función de pérdida y optimizador ---
num_classes = len(np.unique(Y_train_np)) # Debería ser 62 para EMNIST-byclass
print(f"Número de clases detectadas: {num_classes}")

# AQUÍ ES DONDE SE CONFIGURA EL USO DE GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

model = LeNet5PyTorch(num_classes=num_classes).to(device) # Mover el modelo al dispositivo
criterion = nn.CrossEntropyLoss() # Incluye Softmax
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam es un buen optimizador por defecto

# --- Bucle de Entrenamiento ---
epochs = 30 # Keras usaba 40, puedes ajustarlo. Reducido para una prueba más rápida.
# El original de Keras usaba steps_per_epoch=10.
# Aquí, cada epoch procesará todo el dataset.

print(f"Iniciando entrenamiento por {epochs} epochs...")
for epoch in range(epochs):
    model.train() # Poner el modelo en modo entrenamiento
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, data in enumerate(train_loader): # train_loader ya está configurado con batch_size
        inputs, labels = data
        # Mover los datos del lote actual al dispositivo (GPU o CPU)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # Poner a cero los gradientes

        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Calcular la pérdida
        loss.backward() # Backward pass
        optimizer.step() # Actualizar pesos

        running_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        # Si quieres replicar el comportamiento de 'steps_per_epoch=10' de Keras:
        # if i >= 9: # (0-indexed, así que 10 iteraciones)
        #     break
            
    epoch_loss = running_loss / len(train_loader) # Promedio de pérdida por lote en la época
    epoch_acc_train = 100 * correct_train / total_train
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc_train:.2f}%")

print('Entrenamiento finalizado.')

# --- Evaluación del Modelo ---
model.eval() # Poner el modelo en modo evaluación
correct_test = 0
total_test = 0
all_predictions = []
all_labels_test = []

with torch.no_grad(): # No necesitamos calcular gradientes para la evaluación
    for data in test_loader:
        images, labels = data
        # Mover los datos del lote actual al dispositivo (GPU o CPU)
        images, labels_batch = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total_test += labels_batch.size(0)
        correct_test += (predicted == labels_batch).sum().item()
        all_predictions.extend(predicted.cpu().numpy()) # Guardar predicciones (mover a CPU primero)
        all_labels_test.extend(labels.cpu().numpy()) # Guardar etiquetas verdaderas para referencia

test_accuracy = 100 * correct_test / total_test
print(f'Accuracy en el conjunto de test: {test_accuracy:.2f}%')

# --- Predicción y guardado de resultados ---
y_pred_labels_np = np.array(all_predictions)

num_test_samples_actual = len(Y_test_pd)
if len(y_pred_labels_np) != num_test_samples_actual:
    print(f"Advertencia: El número de predicciones ({len(y_pred_labels_np)}) no coincide con el número de muestras de test ({num_test_samples_actual}).")
    # Esto podría ocurrir si batch_size no divide exactamente el tamaño del test set y drop_last=False (por defecto)
    # En este caso, DataLoader devuelve todas las muestras.
    # Aseguramos que y_pred_labels_np tenga la longitud correcta, aunque usualmente coincidirá.
    y_pred_labels_np = y_pred_labels_np[:num_test_samples_actual]


index = np.arange(1, num_test_samples_actual + 1)

y_pred_labels_reshaped = y_pred_labels_np.reshape(-1,1)
index_reshaped = index.reshape(-1, 1)

final_output = np.concatenate([index_reshaped, y_pred_labels_reshaped], axis = 1)

# Prediction csv file
output_filename = "emnist_pytorch_predictions.csv"
np.savetxt(output_filename, final_output, delimiter=",", fmt='%d', header='ImageId,Label', comments='')
print(f"Predicciones guardadas en {output_filename}")

MODEL_SAVE_PATH = "lenet5_emnist_pytorch.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Modelo entrenado guardado en {MODEL_SAVE_PATH}")

# Guardar también los parámetros de normalización, ya que son necesarios para preprocesar nuevas imágenes
PREPROCESSING_PARAMS_PATH = "preprocessing_params.npz"
np.savez(PREPROCESSING_PARAMS_PATH, mean_px=mean_px, std_px=std_px)
print(f"Parámetros de preprocesamiento guardados en {PREPROCESSING_PARAMS_PATH}")