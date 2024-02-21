# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
#pip install opencv-python (for cv2)

# %%
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    #Augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    ])



# %%

# Load datasets
train_dataset = datasets.ImageFolder(root='/users/edatkinson/LLL/split_classes/train/', transform=transforms)

val_dataset = datasets.ImageFolder(root='/users/edatkinson/LLL/split_classes/validation/', transform=transforms)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


#Set up the CNN model to classify the images 
# %%
import torch.nn as nn
import torch.nn.functional as F
# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 500)
        self.fc2 = nn.Linear(500, 6)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# %%
# Create the model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# %%

# Train the model
n_epochs = 2
for epoch in range(n_epochs):
    train_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{n_epochs}, Training Loss: {train_loss}")

# %%

#Evaluate the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Validation Accuracy: {100 * correct / total}%")
# %%
# Save the model
torch.save(model.state_dict(), 'model30.pth')


# %%
#model = CNN()
model.load_state_dict(torch.load('model30.pth'))
model.eval()


# Load the image
image_path = '/users/edatkinson/LLL/split_classes/test/Bullet/xray_08143_png.rf.e16c328f9e72c5aa1a529a916a11a23f.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Preprocess the image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from PIL import Image

# Assuming 'image' is your NumPy array
image_pil = Image.fromarray(image)

# Now apply the transform
input_tensor = transform(image_pil)

input_batch = input_tensor.unsqueeze(0)

# Make a prediction
#model.eval()
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class
_, predicted = torch.max(output, 1)
class_index = predicted.item()
print(class_index)
class_label = train_dataset.classes[class_index]

print(f"Predicted class: {class_label}")

#validation



# %%
# Test over the test set

test_dataset = datasets.ImageFolder(root='/users/edatkinson/LLL/split_classes/test/', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Test Accuracy: {100 * correct / total}%")


# %%
