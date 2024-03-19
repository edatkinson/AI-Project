#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:04:08 2024

@author: maxchesters
"""

# %%
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
#pip install opencv-python (for cv2)

# %%
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

# %%

######Stratified Sampling:########

train_dataset = datasets.ImageFolder(root='/Users/maxchesters/Desktop/AI_Project/MultiClassData/train', transform=transform)
val_dataset = datasets.ImageFolder(root='/Users/maxchesters/Desktop/AI_Project/MultiClassData/validation', transform=transform)

# Calculate weights for each class
train_targets = torch.tensor(train_dataset.targets)
class_sample_count = torch.tensor([(train_targets == t).sum() for t in torch.unique(train_targets, sorted=True)])
print(class_sample_count)
weight = 1. / class_sample_count.float()
print(weight)
samples_weight = torch.tensor([weight[t] for t in train_targets])
# WeightedRandomSampler for training set
train_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# For validation, you can repeat the process if you want stratified sampling there as well
# val_targets = torch.tensor(val_dataset.targets)
# val_class_sample_count = torch.tensor([(val_targets == t).sum() for t in torch.unique(val_targets, sorted=True)])
# val_weight = 1. / val_class_sample_count.float()
# val_samples_weight = torch.tensor([val_weight[t] for t in val_targets])

# val_sampler = WeightedRandomSampler(weights=val_samples_weight, num_samples=len(val_samples_weight), replacement=True)

# Data loaders with stratified sampling
train_loader = DataLoader(train_dataset, batch_size=10,sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=10)



#%%

# Load datasets

train_dataset = datasets.ImageFolder(root='/Users/maxchesters/Desktop/AI_Project/MultiClassData/train', transform=transform)

val_dataset = datasets.ImageFolder(root='/Users/maxchesters/Desktop/AI_Project/MultiClassData/validation', transform=transform)

test_dataset = datasets.ImageFolder(root='/Users/maxchesters/Desktop/AI_Project/MultiClassData/test', transform=transform)
# Data loaders


# Create random samplers to sample from the val_dataset and test_dataset


# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

#Set up the CNN model to classify the images 
# %%

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)  # Adding pooling layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)  # Adding pooling layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)  # Adding pooling layer
        
        # After 3 rounds of halving the dimensions, the size calculation needs adjustment
        # For an input of 224x224, after three poolings, the size is 224 / 2 / 2 / 2 = 28
        self.fc1_size = 64 * 28 * 28  # Adjusted based on the pooling layers
        self.fc1 = nn.Linear(self.fc1_size, 500)
        self.fc2 = nn.Linear(500, 8)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # Apply pooling
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # Apply pooling
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # Apply pooling
        x = x.view(-1, self.fc1_size)  # Flatten the output for the fully connected layer
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# %%

#custom cross entropy loss function

class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, input, target):
        # The F.cross_entropy function already applies log_softmax internally,
        # so we pass the raw scores directly.
        return F.cross_entropy(input, target, weight=self.class_weights)


class_counts = class_sample_count  # counts for classes
print(class_counts)
total_count = sum(class_counts)
classes_weights = torch.tensor([total_count / c for c in class_counts], dtype=torch.float32)
classes_weights /= classes_weights.min() #normalize the weights
print(f'Normalised Class Weights, Need to play around with these:{classes_weights}')
# If you're using CUDA, move the weights to the GPU
if torch.cuda.is_available():
    classes_weights = classes_weights.cuda()


# Create the model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
#criterion = WeightedCrossEntropyLoss(classes_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)


# %%

import pandas as pd
import torch

# Assuming `model`, `train_loader`, `test_loader`, `optimizer`, `criterion` are defined

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

if train_on_gpu:
    model.cuda()
    
n_epochs = 10
momentum_values = [0.0, 0.5, 0.9, 0.99]
weight_decay_val = [0, 0.01, 0.001, 0.0001]
learning_rates = [0.9, 0.1, 0.01, 0.001]
count = 0
print_every = 50
metrics = []
val_loss = np.zeros((48,10))
criterion = nn.CrossEntropyLoss()


for mom in range (len(momentum_values)):
    
    for nolr in range (len(learning_rates)):
        
        model = CNN()
        
        if not train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')
    
        if train_on_gpu:
            model.cuda()
            
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rates[nolr], momentum = momentum_values[mom])
        
        criterion = nn.CrossEntropyLoss()


        for epoch in range(n_epochs):
            train_loss = 0.0
            correct = 0
            total = 0
            running_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                
                
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
                # Calculate metrics after each batch
                batch_accuracy = 100 * correct / total
                batch_loss = train_loss / total
                
                
                # Optionally, evaluate test accuracy at a reduced frequency to save computation
                test_accuracy = None
                val_accuracy = None
               
                model.eval()  # Set model to evaluate mode
               
                #Also, optionally, evaluate validation accuracy
        
                
                val_correct = 0
                val_total = 0
        
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                       
                        val_outputs = model(val_images)
                        _, val_predicted = torch.max(val_outputs, 1)
                        val_total += val_labels.size(0)
                        val_correct += (val_predicted == val_labels).sum().item()
                        validation_loss = criterion(val_outputs, val_labels)
                        
                        running_loss = running_loss + validation_loss
                        
                val_accuracy = 100 * val_correct / val_total
                
                model.train()  # Set model back to train mode
        
                
                
                
               
        
            # Log epoch-level metrics
            epoch_loss = train_loss / len(train_loader.dataset)
            epoch_accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{n_epochs}, End of Epoch, Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy}%")
            avg_loss = running_loss/len(val_loader)
        
            val_loss[count, epoch] = avg_loss
            
        count = count + 1
        print(count)
        # # Convert metrics to DataFrame and save as CSV


for wd in range (len(weight_decay_val)):
    
    for nolr in range (len(learning_rates)):
        
        model = CNN()
        
        if not train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')
    
        if train_on_gpu:
            model.cuda()
            
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rates[nolr], weight_decay = weight_decay_val[wd])
        
        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            train_loss = 0.0
            correct = 0
            total = 0
            running_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
                # Calculate metrics after each batch
                batch_accuracy = 100 * correct / total
                batch_loss = train_loss / total
                
                
                # Optionally, evaluate test accuracy at a reduced frequency to save computation
                test_accuracy = None
                val_accuracy = None
               
                model.eval()  # Set model to evaluate mode
                
        
                
                val_correct = 0
                val_total = 0
        
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                       
                        val_outputs = model(val_images)
                        _, val_predicted = torch.max(val_outputs, 1)
                        val_total += val_labels.size(0)
                        val_correct += (val_predicted == val_labels).sum().item()
                        validation_loss = criterion(val_outputs, val_labels)
                        
                        running_loss = running_loss + validation_loss
                        
                val_accuracy = 100 * val_correct / val_total
                
                model.train()  # Set model back to train mode
        
                
                
                
               
        
            # Log epoch-level metrics
            epoch_loss = train_loss / len(train_loader.dataset)
            epoch_accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{n_epochs}, End of Epoch, Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy}%")
            avg_loss = running_loss/len(val_loader)
        
            val_loss[count, epoch] = avg_loss
            
        count = count + 1
        print(count)
        # Convert metrics to DataFrame and save as CSV
        


for wd in range (len(weight_decay_val)):
    
    for nolr in range (len(learning_rates)):
        
        model = CNN()
        
        if not train_on_gpu:
            print('CUDA is not available.  Training on CPU ...')
        else:
            print('CUDA is available!  Training on GPU ...')
    
        if train_on_gpu:
            model.cuda()
            
        optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rates[nolr], weight_decay = weight_decay_val[wd])
        
        criterion = nn.CrossEntropyLoss()
        for epoch in range(n_epochs):
            train_loss = 0.0
            correct = 0
            total = 0
            running_loss = 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                
                
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(output, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
                # Calculate metrics after each batch
                batch_accuracy = 100 * correct / total
                batch_loss = train_loss / total
                
                
                # Optionally, evaluate test accuracy at a reduced frequency to save computation
                test_accuracy = None
                val_accuracy = None
               
                model.eval()  # Set model to evaluate mode
                test_correct = 0
                test_total = 0
              
                #Also, optionally, evaluate validation accuracy
        
                
                val_correct = 0
                val_total = 0
        
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                        
                        val_outputs = model(val_images)
                        _, val_predicted = torch.max(val_outputs, 1)
                        val_total += val_labels.size(0)
                        val_correct += (val_predicted == val_labels).sum().item()
                        validation_loss = criterion(val_outputs, val_labels)
                        
                        running_loss = running_loss + validation_loss
                        
                val_accuracy = 100 * val_correct / val_total
                
                model.train()  # Set model back to train mode
        
                
                
                
               
        
            # Log epoch-level metrics
            epoch_loss = train_loss / len(train_loader.dataset)
            epoch_accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{n_epochs}, End of Epoch, Training Loss: {epoch_loss}, Training Accuracy: {epoch_accuracy}%")
            avg_loss = running_loss/len(val_loader)
            val_loss[count, epoch] = avg_loss
            
        count = count + 1
        print(count)
# Convert metrics to DataFrame and save as CSV
df = pd.DataFrame(val_loss)
# Write the DataFrame to a CSV file
df.to_csv('/Users/maxchesters/Desktop/AI_Project/HyperParamTuning/Hyperparam_tuning_multi_3_best.csv_final', index=False)



# %%

# #Evaluate the model
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in val_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f"Validation Accuracy: {100 * correct / total}%")

# # %%
# # Load test data

# test_dataset = datasets.ImageFolder(root='/users/edatkinson/LLL/split_classes/test/', transform=transform)
# #print(test_dataset)
# test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
# # %%
# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
# print(f"Test Accuracy: {100 * correct / total}%")

# # %%

# #This cell is for the confusion matrix 

# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

# model.eval()
# y_true = []
# y_pred = []

# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs, 1)
#         y_true += labels.tolist()
#         y_pred += predicted.tolist()

# # Define the correct class names in the same order as the labels
# classes = ['Baton', 'Bullet','Gun','Hammer','HandCuffs', 'Null', 'Scissors'] #needs to be in alphabetical order

# # Print the accuracies for each class 

# for i in range(len(classes)):
#     class_total = 0
#     class_correct = 0
#     for j in range(len(y_true)):
#         if y_true[j] == i:
#             class_total += 1
#             if y_pred[j] == y_true[j]:
#                 class_correct += 1
#     print(f'Accuracy of {classes[i]}: {100 * class_correct / class_total}%')

# cm = confusion_matrix(y_true, y_pred)
# cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# plt.figure(figsize=(10, 8))
# sns.heatmap(cm_df, annot=True, cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # %%