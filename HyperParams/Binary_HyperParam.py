#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:14:08 2024

@author: maxchesters
"""

"""
Arthur binary classificiation CNN

"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 14:06:15 2024

@author: arthu
"""
import numpy as np

import random
import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset, random_split, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Check whether we have a GPU.  Use it if we do.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Navigate to the folder
data_dir = '/Users/maxchesters/Desktop/AI_Project/all'

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Create dataset class
class X_ray_dataset(Dataset):
    def __init__(self, annotations_file = os.path.join(data_dir, 'binary_classes.csv'), img_dir = os.path.join(data_dir, 'XrayImages'), transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

'''    
# Test to see if the class is working
x_ray = X_ray_dataset(transform=transform)
print('len dataset:', len(x_ray)) # use the __len__ method to show the length of the dataset
example = x_ray[0] # use the __getitem__ method to get the first example
print(example[0].shape)
print('Class:', example[1])
'''

x_ray = X_ray_dataset(transform=transform)
# Define the size of the training set
sample_size = int(0.40 * len(x_ray))
 # 10% of the dataset size for the sample

# Randomly sample from the dataset
sample_indices = random.sample(range(len(x_ray)), sample_size)
sample_data = [x_ray[i] for i in sample_indices]

# Split the sample into training and testing sets
train_ds, test_ds= train_test_split(sample_data, test_size=0.2, random_state=42)  # 80% for training, 20% for testing

# Further split the training data into training and validation sets
train_ds, val_ds = train_test_split(train_ds, test_size=0.2, random_state=42)  # 80% for training, 20% for validation

# Define class labels
classes = ('Safe', 'Prohibited')

batch_size = 10

# Calculate the number of classes in train and val sets
class_0_count_train = 0
class_1_count_train = 0
class_0_count_val = 0
class_1_count_val = 0

for image, label in train_ds:  # count classes for train
    if label == 0:
        class_0_count_train += 1
    elif label == 1:
        class_1_count_train += 1

for image, label in val_ds:  # count classes for validation
    if label == 0:
        class_0_count_val += 1
    elif label == 1:
        class_1_count_val += 1
#Trying  to reformat this a bit

# Extract the class labels from the dataset
train_targets = torch.tensor([label for _, label in train_ds])
val_targets = torch.tensor([label for _, label in val_ds])

# Calculate weights for each class
class_sample_count = torch.tensor([class_0_count_train, class_1_count_train])
weight = 1. / class_sample_count.float()
samples_weight = torch.tensor([weight[t] for t in train_targets])

# WeightedRandomSampler for training set
train_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# For validation, you can repeat the process if you want stratified sampling there as well
val_class_sample_count = torch.tensor([class_0_count_val, class_1_count_val])
val_weight = 1. / val_class_sample_count.float()
val_samples_weight = torch.tensor([val_weight[t] for t in val_targets])

val_sampler = WeightedRandomSampler(weights=val_samples_weight, num_samples=len(val_samples_weight), replacement=True)

# Data loaders with stratified sampling
train_loader = DataLoader(train_ds,
                          batch_size=10,
                          sampler=train_sampler,
                          shuffle = False)  # don't shuffle with stratified sampling
test_loader = DataLoader(test_ds,
                         batch_size = batch_size*5,
                         shuffle = False)
val_loader = DataLoader(val_ds,
                        batch_size=10*5,
                        sampler=val_sampler,
                        shuffle = False)


# Create CNN
class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    # Creating layers
    self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 128, kernel_size = 3)
    self.batch1 = nn.BatchNorm2d(128)
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(in_channels = 128, out_channels = 10, kernel_size = 3)
    self.batch2 = nn.BatchNorm2d(10)
    self.relu2 = nn.ReLU()
    self.pool2 = nn.AdaptiveAvgPool2d(1)

  def forward(self, x):
    x = nn.functional.relu(self.conv1(x)) # apply non-linearity
    x = self.batch1(x)
    x = self.relu1(x)

    x = nn.functional.relu(self.conv2(x)) # apply non-linearity
    x = self.batch2(x)
    x = self.relu2(x)
    x = self.pool2(x)

    return x

# Instantiate Network


'''
# Checking the shape of an image before and after the network
for i, data in enumerate(train_loader):
  inputs, labels = data[0].to(device), data[1].to(device)
  print('Input shape:', inputs.shape)
  print('Output shape:', net(inputs).shape)
  break
'''

# Specifying hyperparameters

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        log_softmax_values = torch.log_softmax(input, dim=1)

        # Calculate class weights
        class_weights = torch.tensor([1.0 / count for count in torch.bincount(target)], device=input.device)

        # Weight the losses by class weights
        weighted_losses = -target * log_softmax_values * class_weights[target]

        return torch.mean(torch.sum(weighted_losses, dim=1))


#loss_func = CustomCrossEntropyLoss()

  
# Function for training one epoch
def train_epoch():
  net.train(True) # set to training mode

  # Metrics that will build up
  running_loss = 0
  running_accuracy = 0

  # Iterate over train data
  for batch_index, data in enumerate(train_loader):
    inputs, labels = data[0].to(device), data[1].to(device) # get the images and labels

    optimiser.zero_grad() # set all non-zero values for gradients to 0 (reset gradients)

    outputs = net(inputs).squeeze((-1, -2)) # shape: [batch size, 2]
    correct_prediction = torch.sum(labels == torch.argmax(outputs, dim = 1)).item() # check how many images the model predicted correctly
    running_accuracy += correct_prediction/batch_size # update the accuracy

    # Train model
    loss = loss_func(outputs, labels) # compare model predictions with labels
    running_loss += loss.item() # update the loss
    loss.backward() # calculate gradients
    optimiser.step()

    if batch_index % 50 == 49:      # print for every 500 batchs
      avg_loss = running_loss/50  # get the average loss across batches
      avg_acc = (running_accuracy/50) * 100 # get the average accuracy across batches
      print('Batch {0}, Loss: {1:.3f}, Accuracy: {2:.1f}%'.format(batch_index+1, avg_loss, avg_acc))
      
      running_loss = 0
      running_accuracy = 0

  print()

# Function for testing
test_accuracies = []

def test_epoch(epoch):
  with torch.no_grad():
    correct_prediction = 0
    total = 0
    for images, labels in test_loader:
      images = images.to(device)
      labels = labels.to(device)

      outputs = net(images).squeeze((-1,-2))

      predicted = torch.argmax(outputs, -1)
      correct_prediction += (predicted == labels).sum().item()
      total += labels.size(0)

    acc =(correct_prediction/total) * 100
    
    # add the accuracy to a list 
    test_accuracies.append(acc)

    print('Test accuracy after {0:.0f} epoch(s): {1:.1f}%'.format(epoch+1, acc))
    print()

# Function for validating one epoch
val_accuracies = []
val_losses = []

def validate_epoch():
  net.train(False) # set to evaluation mode
  running_loss = 0
  running_acc = 0

  # Iterate over validation data
  for batch_index, data in enumerate(val_loader):
    inputs, labels = data[0].to(device), data[1].to(device)

    with torch.no_grad(): # not worried about gradients here as not training
      outputs = net(inputs).squeeze((-1, -2)) # shape [batch size, 2]
      correct_prediction = torch.sum(labels == torch.argmax(outputs, dim = 1)).item() # check how many images the model predicted correctly
      running_acc += correct_prediction/(batch_size*5) # update the accuracy
      loss = loss_func(outputs, labels) # compare model predictions with labels
      running_loss += loss.item() # update the loss

  avg_loss = running_loss/len(val_loader)
  avg_acc = (running_acc/len(val_loader)) * 100
  
  # add the accuracy and loss to a list
  val_accuracies.append(avg_acc)
  val_losses.append(avg_loss)

  print('Val Loss: {0:.3f}, Val Accuracy: {1:.1f}%'.format(avg_loss, avg_acc))

  print('-----------------------------------------------------------')
  print()
  return avg_loss
    # Training loop
    
num_epochs = 10
all_val_loss = np.zeros((48,10))
learning_rates = [0.9, 0.1, 0.01, 0.001]
momentum_values = [0.0, 0.5, 0.9, 0.99]
weight_decay_val = [0, 0.01, 0.001, 0.0001]
count = 0
for mom in range (len(momentum_values)):
    for nolr in range (len(learning_rates)):
        
        
        net = CNN()
        net.to(device)
        loss_func = nn.CrossEntropyLoss()
        optimiser = optim.SGD(net.parameters(), lr = learning_rates[nolr], momentum = momentum_values[mom])
        
        for i in range(num_epochs):
          print('Epoch:', i+1, '\n')
          
          train_epoch()
          test_epoch(i)
          Val_Loss = validate_epoch()
          all_val_loss[(count), i] = (Val_Loss)
        
        print('Finished Training')
        count = count + 1
for wd in range (len(weight_decay_val)):
    for nolr in range (len(learning_rates)):
        
        
        net = CNN()
        net.to(device)
        loss_func = nn.CrossEntropyLoss()
        optimiser = optim.Adam(net.parameters(), lr = learning_rates[nolr], weight_decay = weight_decay_val[wd])
        
        for i in range(num_epochs):
          print('Epoch:', i+1, '\n')
          
          train_epoch()
          test_epoch(i)
          Val_Loss = validate_epoch()
          all_val_loss[(count), i] = (Val_Loss)
        
        print('Finished Training')    
        count = count + 1


for wd in range (len(weight_decay_val)):
    for nolr in range (len(learning_rates)):
        
        
        net = CNN()
        net.to(device)
        loss_func = nn.CrossEntropyLoss()
        optimiser = optim.RMSprop(net.parameters(), lr = learning_rates[nolr], weight_decay = weight_decay_val[wd])
        
        for i in range(num_epochs):
          print('Epoch:', i+1, '\n')
          
          train_epoch()
          test_epoch(i)
          Val_Loss = validate_epoch()
          all_val_loss[(count), i] = (Val_Loss)
        
        print('Finished Training')    
        count = count + 1
df = pd.DataFrame(all_val_loss)
# Write the DataFrame to a CSV file
df.to_csv('/Users/maxchesters/Desktop/AI_Project/HyperParamTuning/validation_loss_all_momentum_weightdecay', index=False)

# Create the confusion matrix
# net.eval()
# y_true = []
# y_pred = []

# with torch.no_grad():
#   for images, labels in test_loader:
#     images = images.to(device)
#     labels = labels.to(device)
#     outputs = net(images)
#     _, predicted = torch.max(outputs, 1)
#     y_true += labels.tolist()
#     y_pred += predicted.tolist()

# y_pred_flattened = [item for sublist in y_pred for subsublist in sublist for item in subsublist]

# cm = confusion_matrix(y_true, y_pred_flattened)
# cm_df = pd.DataFrame(cm, index=classes, columns=classes)

# plt.figure(figsize=(10, 8))
# sns.heatmap(cm_df, annot=True, cmap='Blues')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.show()

# # Plotting accuracy and loss graphs
# epochs = range(1, len(val_accuracies) + 1)
# plt.figure(2)
# plt.plot(epochs, val_accuracies, label = 'Validation Accuracy')
# plt.plot(epochs, test_accuracies, label = 'Test Accuracy')
# plt.title('Validation and Test Accuracies vs Number of Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(3)
# plt.plot(epochs, val_losses, label = 'Validation Loss')
# plt.title('Validation loss vs Number of Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.show()

# # Generate a classification report using the predicted and actual labels
# class_report = classification_report(y_true, y_pred_flattened, target_names=['Safe', 'Prohibited'])

# # Print the classification report
# print("Classification Report:\n", class_report)