import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------
# 1. Hyperparameters
# -----------------------------
# Hyperparameters that are used as part of the Classifier model
batch_size = 32
num_epochs = 15
learning_rate = 1e-4

# -----------------------------
# 2. Data transforms
# -----------------------------

# Transforms the images in the data set to the same format before training the model
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# 3. Datasets
# -----------------------------
# The data sets used to train, validate and test the model.
train_dataset = datasets.ImageFolder("./training_data", transform=train_transforms)
val_dataset = datasets.ImageFolder("./validate_data", transform=eval_transforms)
test_dataset = datasets.ImageFolder("./test_data", transform=eval_transforms)

# Load the data in bacthes until all images are used on the model
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# -----------------------------
# 4. Model
# -----------------------------
# Sets the model to efficientnet_b0 and adds the hyperparameter "Weights" to Default.
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
# Binary classification is used as there are currently only 2 outcomes "Real" or "AI-Generated"
model.classifier[1] = nn.Linear(1280, 2)  # binary classification

# Sets the CPU to be used for the model to be trained, validated, tested and run
device = torch.device("cpu")
model = model.to(device)

# -----------------------------
# 5. Loss + Optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# 6. Training loop
# -----------------------------
validation_accuracy = []
epoch_array = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    print(f"Epoch {epoch+1} Train Loss: {train_loss/len(train_loader):.4f}")

    # -----------------------------
    # 7. Validation
    # -----------------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    validation_accuracy.append(accuracy)
    epoch_array.append((epoch+1))
    print(f"Validation Accuracy: {accuracy:.4f}")

print(validation_accuracy)

plt.figure(figsize=(8, 6))
plt.plot(epoch_array, validation_accuracy, lw=2, label=f'Accuracy over each EPOCH')
plt.plot([0, 1], [0, 1], linestyle='--', color='grey')

plt.xlabel('EPOCH')
plt.ylabel('Accuracy')
plt.title('Accuracy over each EPOCH')
plt.legend(loc='lower right')

plt.savefig("epoch_accuracy.png", dpi=300, bbox_inches='tight')
# -----------------------------
# 8. Save model
# -----------------------------

torch.save(model.state_dict(), "efficientnet_ai_vs_real.pth")
print("Model saved.")

# -----------------------------
# 9. Test evaluation
# -----------------------------
num_classes = 2
metric =  MulticlassConfusionMatrix(num_classes=num_classes)
f1 = MulticlassF1Score(num_classes=num_classes)

precision = MulticlassPrecision(num_classes=num_classes, average="macro")
recall = MulticlassRecall(num_classes=num_classes, average="macro")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        metric.update(predicted, labels)
        f1.update(predicted, labels)
        precision.update(predicted, labels)
        recall.update(predicted, labels)

cm = metric.compute()
print(cm)
score = f1.compute()
print("F1:", score)
print("Precision:", precision.compute())
print("Recall:", recall.compute())
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")