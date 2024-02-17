

import pandas as pd 


csv_file_path = '/users/edatkinson/LLL/CSVfiles'
dict_file_path = 'users/edatkinson/LLL/CSVfiles/class_filenames.json'
df = pd.read_csv(csv_file_path + '/train_with_null_classes.csv')

'''
Create a list of filenames which share the same class label for each class
'''

# Create a list of class names including null
class_names = df.columns[1:]

# Create a dictionary to store the filenames for each class
class_filenames = {class_name: [] for class_name in class_names}

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Get the filename and class labels
    image_filename = row['filename']  # Replace with the actual column name
    class_labels = row.drop('filename').values
    
    # Iterate over each class label
    for class_name, label in zip(class_names, class_labels):
        # If the label is 1, add the filename to the corresponding class list
        if label == 1:
            class_filenames[class_name].append(image_filename)


#Save the dictionary to a file
import json

with open(csv_file_path + '/class_filenames.json', 'w') as file:
    json.dump(class_filenames, file, indent=4)

#Load the dictionary from a file
with open(csv_file_path + '/class_filenames.json', 'r') as file:
    loaded_class_filenames = json.load(file)

# Print the first 5 filenames for each class
for class_name, filenames in loaded_class_filenames.items():
    print(f"Class: {class_name}")

    print(filenames[:5])
