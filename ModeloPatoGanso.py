import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
from PIL import Image
import torchvision.transforms as transforms

# Define transformations for training and validation sets
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the datasets with the transformations
train_dataset = datasets.ImageFolder(root='C:\\Users\\dafne\\RetoBenji\\Data\\train\\images', transform=train_transforms)
val_dataset = datasets.ImageFolder(root='C:\\Users\\dafne\\RetoBenji\\Data\\val\\images', transform=val_transforms)

# DataLoader for training and validation setss
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

train_classes = train_dataset.classes
print("Number of batches in train loader:", len(train_loader))
print("Classes:", train_classes)


# Definir el modelo CNN
class DuckGooseClassifier(nn.Module):
    def __init__(self):
        super(DuckGooseClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Usamos ResNet18 preentrenado
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # Cambiamos la última capa para 2 clases

    def forward(self, x):
        return self.model(x)

# Crear una instancia del modelo
model = DuckGooseClassifier()

# Mover el modelo a GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Parámetros de entrenamiento
epochs = 5

# Bucle de entrenamiento
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Adelante y atrás
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Imprimir el promedio de pérdida por cada epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluación en el conjunto de validación
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on validation set: {100 * correct / total:.2f}%")

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'duck_goose_model.pth')
print("Modelo guardado como 'duck_goose_model.pth'")

