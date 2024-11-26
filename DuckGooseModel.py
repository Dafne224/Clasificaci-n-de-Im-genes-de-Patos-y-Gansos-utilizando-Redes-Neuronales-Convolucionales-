import os
import random
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
import torch.nn as nn
import warnings

# Ocultar warnings
warnings.filterwarnings("ignore")

# Definir el modelo CNN
class DuckGooseClassifier(nn.Module):
    def __init__(self):
        super(DuckGooseClassifier, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.model(x)

# Cargar el modelo entrenado
model = DuckGooseClassifier()
model.load_state_dict(torch.load('duck_goose_model.pth'))
print("Modelo cargado exitosamente.")
model.eval()

# Usar GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Función para buscar imágenes en la carpeta y sus subcarpetas
def find_images_in_folder(folder_path):
    image_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, file))
    return image_files

# Función para el juego "Pato, Pato, Ganso" con búsqueda inicial de "duck"
def duck_duck_goose_game(folder_path, num_samples=5):
    print(f"Iniciando el juego 'Pato, Pato, Ganso' en: {folder_path}")

    # Buscar imágenes en la carpeta y subcarpetas
    image_files = find_images_in_folder(folder_path)
    if not image_files:
        print("No se encontraron imágenes en la carpeta o subcarpetas.")
        return

    print(f"Se encontraron {len(image_files)} imágenes disponibles.")

    # Seleccionar imágenes aleatoriamente
    random.shuffle(image_files)  # Mezclar imágenes para seleccionar al azar

    # Asegurar que el juego comience con un "duck"
    found_duck = False
    for image_path in image_files:
        # Cargar y transformar la imagen
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predicción
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        # Interpretar resultado
        class_names = ['duck', 'goose']
        predicted_class = class_names[predicted.item()]

        if predicted_class == 'duck':
            print(f"Inicio del juego con: {os.path.basename(image_path)} (Predicción: {predicted_class})")
            found_duck = True
            plt.imshow(image)
            plt.title(f"Predicción: {predicted_class}")
            plt.axis('off')
            plt.show()
            break

    if not found_duck:
        print("No se encontró ningún 'duck' para iniciar el juego.")
        return

    # Continuar el juego después de encontrar un "duck"
    selected_images = random.sample(image_files, min(num_samples, len(image_files)))
    for image_path in selected_images:
        filename = os.path.basename(image_path)
        print(f"Procesando: {filename}")

        # Cargar y transformar la imagen
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Predicción
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)

        # Interpretar resultado
        predicted_class = class_names[predicted.item()]
        print(f"Predicción: {predicted_class}")

        # Mostrar imagen
        plt.imshow(image)
        plt.title(f"Predicción: {predicted_class}")
        plt.axis('off')
        plt.show()

        # Si encuentra un "ganso", preguntar al usuario si desea continuar
        if predicted_class == 'goose':
            user_input = input("¡Ganso encontrado! ¿Quieres seguir jugando? (sí/no): ").strip().lower()
            if user_input != 'sí':
                print("Juego terminado. ¡Gracias por participar!")
                return
        else:
            print(f"{filename}: Pato")

# Función principal de interacción con el usuario
def user_interaction():
    while True:
        print("\nOpciones:")
        print("1. Jugar 'Pato, Pato, Ganso' en la carpeta original (train/images).")
        print("2. Jugar 'Pato, Pato, Ganso' en una nueva carpeta.")
        print("3. Salir.")
        
        choice = input("Elige una opción (1/2/3): ").strip()
        if choice == '1':
            folder_path = 'C:\\Users\\dafne\\RetoBenji\\Data\\train\\images'
            duck_duck_goose_game(folder_path, num_samples=5)
        elif choice == '2':
            folder_path = input("Introduce la ruta de la carpeta con imágenes nuevas: ").strip()
            duck_duck_goose_game(folder_path, num_samples=5)
        elif choice == '3':
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Inténtalo nuevamente.")

# Ejecutar interacción con el usuario
user_interaction()

