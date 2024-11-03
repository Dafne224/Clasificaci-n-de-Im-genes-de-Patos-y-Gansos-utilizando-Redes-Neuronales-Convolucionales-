import torch
from torchvision import transforms
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
import warnings

# Ocultar warnings
warnings.filterwarnings("ignore")

# Definir el modelo CNN (igual que en el archivo de entrenamiento)
class DuckGooseClassifier(nn.Module):
    def __init__(self):
        super(DuckGooseClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # Cambiamos la última capa para 2 clases

    def forward(self, x):
        return self.model(x)

# Crear una instancia del modelo y cargar los pesos entrenados
model = DuckGooseClassifier()
model.load_state_dict(torch.load('duck_goose_model.pth'))
print("Modelo cargado exitosamente.")  # Confirmación de carga del modelo
model.eval()  # Poner el modelo en modo de evaluación

# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Transformación de la imagen (ajustada para mejorar claridad)
transform = transforms.Compose([
    transforms.Resize(224),  # Redimensionar a 224 directamente para evitar borrosidad
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Función para simular "pato, pato, ganso" con opción de continuar y mejorar la claridad
def duck_duck_goose_game(folder_path, model, num_samples=5):
    model.eval()  # Poner el modelo en modo de evaluación
    
    print(f"Starting 'duck, duck, goose' game in folder: {folder_path}")
    
    # Recorrer las subcarpetas de "duck" y "goose"
    duck_folder = os.path.join(folder_path, 'duck')
    goose_folder = os.path.join(folder_path, 'goose')
    
    # Obtener todas las imágenes en cada carpeta
    duck_images = [os.path.join(duck_folder, f) for f in os.listdir(duck_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    goose_images = [os.path.join(goose_folder, f) for f in os.listdir(goose_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Seleccionar un número aleatorio de imágenes de cada carpeta
    selected_duck_images = random.sample(duck_images, min(num_samples, len(duck_images)))
    selected_goose_images = random.sample(goose_images, min(num_samples, len(goose_images)))
    
    # Combinar las imágenes seleccionadas
    selected_images = selected_duck_images + selected_goose_images
    random.shuffle(selected_images)  # Mezclar las imágenes para que no estén en orden de "duck" y "goose"
    
    for image_path in selected_images:
        filename = os.path.basename(image_path)
        print(f"Processing file: {filename}")  # Mensaje de depuración

        # Cargar y transformar la imagen
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # Añadir dimensión para el lote y mover al dispositivo

        # Desactivar cálculo de gradientes para predicción
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        # Obtener la clase predicha
        class_names = ['duck', 'goose']
        predicted_class = class_names[predicted.item()]

        # Mostrar la imagen con su predicción
        plt.imshow(image)
        plt.title(f"Predicción: {predicted_class}")
        plt.axis('off')
        plt.show()

        # Simular "pato, pato, ganso" con opción de continuar
        if predicted_class == 'goose':
            print(f"{filename}: ¡Ganso encontrado!")
            user_input = input("¿Quieres seguir jugando? (sí/no): ").strip().lower()
            if user_input != 'sí':
                print("Juego terminado.")
                break
        else:
            print(f"{filename}: Pato")

# Path de la carpeta que contiene las imágenes
folder_path = 'C:\\Users\\dafne\\RetoBenji\\Data\\train\\images' 

# Ejecutar el juego con selección aleatoria de imágenes
duck_duck_goose_game(folder_path, model, num_samples=5)