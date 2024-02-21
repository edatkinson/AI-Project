#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:31:39 2024

@author: trekz1
"""

import pandas as pd

# Read the CSV file
df = pd.read_csv('/Users/edatkinson/lll.v1i.multiclass/valid/_classes.csv')  # Replace 'your_file.csv' with the path to your CSV file


# Function to check if all values in a row except the first are 0
def check_all_zeros(row):
    return all(value == 0 for value in row[1:])  # Skip the first column

# Apply the function to each row and create the 'Null' column
df['Null'] = df.apply(lambda row: 1 if check_all_zeros(row[:-1]) else 0, axis=1)

# Save the modified DataFrame back to a CSV, if needed
df.to_csv('/Users/edatkinson/lll.v1i.multiclass/CSVfiles/val_with_null_classes.csv', index=False)  # Replace 'modified_file.csv' with your desired output file name

