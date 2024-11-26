import os
from PIL import Image
import matplotlib.pyplot as plt

# Rutas de las carpetas
duck_folder = 'C:\\Users\\dafne\\RetoBenji\\Data\\train\\images\\duck'
goose_folder = 'C:\\Users\\dafne\\RetoBenji\\Data\\train\\images\\goose'

# Contar imágenes en cada clase
num_duck_images = len([f for f in os.listdir(duck_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])
num_goose_images = len([f for f in os.listdir(goose_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

print("Número de imágenes de patos:", num_duck_images)
print("Número de imágenes de gansos:", num_goose_images)

# Verificar la resolución de las imágenes
def check_image_quality(folder_path, num_samples=5):
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    sample_images = images[:num_samples]  # Seleccionar algunas muestras

    for img_name in sample_images:
        img_path = os.path.join(folder_path, img_name)
        with Image.open(img_path) as img:
            print(f"{img_name} - Tamaño: {img.size}")

            # Mostrar la imagen
            plt.imshow(img)
            plt.title(f"Tamaño: {img.size}")
            plt.axis('off')
            plt.show()

print("\nEjemplo de calidad de imágenes en la clase 'duck':")
check_image_quality(duck_folder)

print("\nEjemplo de calidad de imágenes en la clase 'goose':")
check_image_quality(goose_folder)
