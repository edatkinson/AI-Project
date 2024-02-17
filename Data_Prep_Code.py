#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:11:01 2024

@author: maxchesters
"""

#Data processing for AI

import os
import shutil
import pandas as pd

photo_dir = '/Users/maxchesters/Desktop/lll/train/X-Ray_image'

csv_file = '/Users/maxchesters/Desktop/lll/train/_classes.csv'

df = pd.read_csv(csv_file)

destination = '/Users/maxchesters/Desktop/lll/train'

no_class = df.iloc[:, 1:].eq(0).all(axis=1)

# Add a new column "null" with value 1 if all class columns are 0, else 0
df[' Null'] = no_class.astype(int)

class_names = df.columns[1:]

for index, row in df.iterrows():
    # Get the filename and class labels
    image_filename = row['filename']  # Replace with the actual column name
    class_labels = row.drop('filename').values
    
    # Iterate over each class label
    for class_name, label in zip(class_names, class_labels):
        # If the label is 1, move the image to the corresponding class folder
        if label == 1:
            source_path = os.path.join(photo_dir, image_filename)
            target_path = os.path.join(destination, class_name, image_filename)
            
            # Check if the source file exists before moving
            if os.path.exists(source_path):
                try:
                    shutil.move(source_path, target_path)
                    print(f"Moved {image_filename} to {class_name} folder.")
                except Exception as e:
                    print(f"Error moving {image_filename} to {class_name} folder: {e}")
            else:
                print(f"Source file {image_filename} does not exist.")
