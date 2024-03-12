
import pandas as pd
import glob
import json
import os
import shutil
from sklearn.model_selection import train_test_split
'''
Create a list of filenames which share the same class label for each class
'''


# csv_file_path = '/users/edatkinson/LLL/CSVfiles'
# dict_file_path = '/users/edatkinson/AI-Project/class_filenames.json'
# # df = pd.read_csv(csv_file_path + '/val_with_null_classes.csv')



# # Path to directory where all CSV files are stored
# path = '/Users/edatkinson/LLL/CSVfiles'
# all_files = glob.glob(path + "/*.csv")

# li = []

# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     li.append(df)

# frame = pd.concat(li, axis=0, ignore_index=True)
# frame.to_csv('combined_csv.csv', index=False)

# df = pd.read_csv('combined_csv.csv')

# # Create a list of class names including null
# class_names = df.columns[1:]

# # Create a dictionary to store the filenames for each class
# class_filenames = {class_name: [] for class_name in class_names}

# # Iterate over each row in the DataFrame
# for index, row in df.iterrows():
#     # Get the filename and class labels
#     image_filename = row['filename']  # Replace with the actual column name
#     class_labels = row.drop('filename').values
    
#     # Iterate over each class label
#     for class_name, label in zip(class_names, class_labels):
#         # If the label is 1, add the filename to the corresponding class list
#         if label == 1:
#             class_filenames[class_name].append(image_filename)


# #Save the dictionary to a file

# with open(dict_file_path, 'w') as file:
#     json.dump(class_filenames, file, indent=4)




## Move images into seperate folders depending on class label for the whole dataset

# Path to the JSON file and the directory containing all images
# json_file_path = '/users/edatkinson/AI-Project/class_filenames.json'

# train_images_directory = '/users/edatkinson/LLL/lll.v1i.multiclass/train'
# test_images_directory = '/users/edatkinson/LLL/lll.v1i.multiclass/test'
# valid_images_directory = '/users/edatkinson/LLL/lll.v1i.multiclass/valid'

# target_directory = '/users/edatkinson/LLL/classes'



# # Load the dictionary from the JSON file
# with open(json_file_path, 'r') as file:
#     class_filenames = json.load(file)

# # Iterate over each class and the corresponding filenames
# for class_name, filenames in class_filenames.items():
#     # # Create a new directory for the class if it doesn
#     class_directory = os.path.join(target_directory, class_name)
#     #os.makedirs(class_directory, exist_ok=True)

#     # # Copy each image to the corresponding class directory

#     for filename in filenames:
#         source_path = os.path.join(valid_images_directory, filename)
#         target_path = os.path.join(class_directory, filename)

#         # Check if the source file exists before copying
#         if os.path.exists(source_path):
#             try:
#                 shutil.copy(source_path, target_path)
#                 print(f"Copied {filename} to {class_name} folder.")
#             except Exception as e:
#                 print(f"Error copying {filename} to {class_name} folder: {e}")
#         else:
#             print(f"Source file {filename} does not exist.")



# Directory containing images of each class
source_dir = '/users/edatkinson/LLL/classes/Hammer/'

# Target directory for train, validation, and test datasets
target_base_dir = '/users/edatkinson/LLL/split_classes/'

# Ratios for splitting
train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

# Create target directories
for split in ['train', 'validation', 'test']:
    split_dir = os.path.join(target_base_dir, split, 'Hammer')
    os.makedirs(split_dir, exist_ok=True)

# List all files in the source directory
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Split files
train_files, test_files = train_test_split(files, test_size=1 - train_ratio, random_state=42)
val_files, test_files = train_test_split(test_files, test_size=test_ratio / (test_ratio + validation_ratio), random_state=42)

# Function to copy files to their respective directories
def copy_files(file_list, destination_folder):
    for file in file_list:
        shutil.copy(os.path.join(source_dir, file), destination_folder)

# Copy files to their new directories
copy_files(train_files, os.path.join(target_base_dir, 'train', 'Hammer'))
copy_files(val_files, os.path.join(target_base_dir, 'validation', 'Hammer'))
copy_files(test_files, os.path.join(target_base_dir, 'test', 'Hammer'))

print("Files have been successfully split into train, validation, and test datasets.")


