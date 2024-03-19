#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:53:09 2024

@author: maxchesters
"""


"""
hyper param plotter 2
"""



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('/Users/maxchesters/Desktop/AI_Project/HyperParamTuning/Hyperparam_tuning_multi.csv')

df_sgd = df.iloc[:16]
df_adam = df.iloc[16:32]
df_rmsprop = df.iloc[32:48]
learning_rates = [0.9, 0.1, 0.01, 0.001]
momentum_values = [0.0, 0.5, 0.9, 0.99]
weight_decay_val = [0, 0.01, 0.001, 0.0001]

"""

Find the lowest average row and the lowest at the last

"""
df_sgd = df_sgd.reset_index(drop = True)
df_rmsprop = df_rmsprop.reset_index(drop = True)
df_adam = df_adam.reset_index(drop = True)


#sgd
sgd_row_avg = df_sgd.mean(axis=1)
lowest_avg_row_index_sgd = sgd_row_avg.idxmin()
lowest_last_col_index_sgd = df_sgd.iloc[:, -1].idxmin()

#adam
adam_row_avg = df_adam.mean(axis=1)
lowest_avg_row_index_adam = adam_row_avg.idxmin()
lowest_last_col_index_adam = df_adam.iloc[:, -1].idxmin()

#rmsprop
rmsprop_row_avg = df_rmsprop.mean(axis=1)
lowest_avg_row_index_rmsprop = rmsprop_row_avg.idxmin()
lowest_last_col_index_rmsprop = df_rmsprop.iloc[:, -1].idxmin()
"""
Generate labels for hyperparameters
"""


labels = []

for a in range (len(momentum_values)):
    for b in range (len(learning_rates)):
        # Create a copy of the existing dataframe
        momentum_label = momentum_values[a]
        learning_label = learning_rates[b]
        
        # Create a label string
        label = f"Multi Class, SGD, momentum = {momentum_label}, lr = {learning_label}"
        # Add label as a new column
        
        labels.append(label)
        # Append the new dataframe to the list
        

sgd_labels = pd.DataFrame(labels, columns=['Label'])
# Concatenate all the new dataframes into a single dataframe

labels = []

for a in range (len(weight_decay_val)):
    for b in range (len(learning_rates)):
        # Create a copy of the existing dataframe
        weight_label = weight_decay_val[a]
        learning_label = learning_rates[b]
        
        # Create a label string
        label = f"Multi Class, Adam, weight_decay = {weight_label}, lr = {learning_label}"
        # Add label as a new column
        
        labels.append(label)
        # Append the new dataframe to the list
        
adam_labels = pd.DataFrame(labels, columns=['Label'])     
        
        
        
labels = []

for a in range (len(weight_decay_val)):
    for b in range (len(learning_rates)):
        # Create a copy of the existing dataframe
        weight_label = weight_decay_val[a]
        learning_label = learning_rates[b]
        
        # Create a label string
        label = f"Multi Class, RMSprop, weight_decay = {weight_label}, lr = {learning_label}"
        # Add label as a new column
        
        labels.append(label)
        # Append the new dataframe to the list
        
rms_labels = pd.DataFrame(labels, columns=['Label'])




"""
Plot For SGD
"""

column_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df_sgd.columns = column_names

df_sgd['Label'] = sgd_labels
melted_SGD_df = df_sgd.melt(id_vars='Label', var_name='Epoch', value_name='Loss')


plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_SGD_df, x='Epoch', y='Loss', hue='Label', marker='o')

# Optional: Set labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for Each Hyperparameter Across 10 Epochs')

plt.grid(True)  # Add grid for better visualization
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

plt.show()

"""
Plot For Adam

"""

column_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df_adam.columns = column_names
df_adam = df_adam.reset_index(drop = True)
df_adam['Label'] = adam_labels
melted_adam_df = df_adam.melt(id_vars='Label', var_name='Epoch', value_name='Loss')


plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_adam_df, x='Epoch', y='Loss', hue='Label', marker='o')

# Optional: Set labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for Each Hyperparameter Across 10 Epochs')

plt.grid(True)  # Add grid for better visualization
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

plt.show()

"""
Plot For RMSprop

"""

column_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df_rmsprop.columns = column_names
df_rmsprop = df_rmsprop.reset_index(drop = True)
#df_rmsprop = df_rmsprop[(df_rmsprop <= 2).all(axis=1)]
df_rmsprop = df_rmsprop.merge(rms_labels, left_index = True, right_index = True)

melted_rmsprop_df = df_rmsprop.melt(id_vars='Label', var_name='Epoch', value_name='Loss')


plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_rmsprop_df, x='Epoch', y='Loss', hue='Label', marker='o')

# Optional: Set labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for Each Hyperparameter Across 10 Epochs')

plt.grid(True)  # Add grid for better visualization
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

plt.show()

"""
Find lowest column and average 
"""

#lowest_col_sgd = df_sgd.iloc[[lowest_last_col_index_sgd]]
lowest_avg_sgd = df_sgd.iloc[[lowest_avg_row_index_sgd]]


#lowest_col_adam = df_adam.iloc[[lowest_last_col_index_adam]]
lowest_avg_adam = df_adam.iloc[[lowest_avg_row_index_adam]]

#lowest_col_rmsprop = df_rmsprop.iloc[[lowest_last_col_index_rmsprop]]
lowest_avg_rmsprop = df_rmsprop.iloc[[lowest_avg_row_index_rmsprop]]


lowest_df1 = pd.concat([ lowest_avg_sgd,
                        lowest_avg_adam,
                        lowest_avg_rmsprop
                        
                                 ])

"""
Plot Lowest Columns
"""

# melted_lowest_df = lowest_df.melt(id_vars='Label', var_name='Epoch', value_name='Loss')


# plt.figure(figsize=(10, 6))
# sns.lineplot(data=melted_lowest_df, x='Epoch', y='Loss', hue='Label', marker='o')

# # Optional: Set labels and title
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss for Each Hyperparameter Across 10 Epochs')
# plt.xlim(0, 10)  # Example limits, adjust according to your data
# plt.ylim(0, 3)
# plt.grid(True)  # Add grid for better visualization
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

# plt.show()


df = pd.read_csv('/Users/maxchesters/Desktop/AI_Project/validation_loss_all_momentum_weightdecay.csv')
df_sgd = df.iloc[:16]
df_adam = df.iloc[16:32]
df_rmsprop = df.iloc[32:48]
learning_rates = [0.9, 0.1, 0.01, 0.001]
momentum_values = [0.0, 0.5, 0.9, 0.99]
weight_decay_val = [0, 0.01, 0.001, 0.0001]

"""

Find the lowest average row and the lowest at the last

"""
df_sgd = df_sgd.reset_index(drop = True)
df_rmsprop = df_rmsprop.reset_index(drop = True)
df_adam = df_adam.reset_index(drop = True)


#sgd
sgd_row_avg = df_sgd.mean(axis=1)
lowest_avg_row_index_sgd = sgd_row_avg.idxmin()
lowest_last_col_index_sgd = df_sgd.iloc[:, -1].idxmin()

#adam
adam_row_avg = df_adam.mean(axis=1)
lowest_avg_row_index_adam = adam_row_avg.idxmin()
lowest_last_col_index_adam = df_adam.iloc[:, -1].idxmin()

#rmsprop
rmsprop_row_avg = df_rmsprop.mean(axis=1)
lowest_avg_row_index_rmsprop = rmsprop_row_avg.idxmin()
lowest_last_col_index_rmsprop = df_rmsprop.iloc[:, -1].idxmin()
"""
Generate labels for hyperparameters
"""


labels = []

for a in range (len(momentum_values)):
    for b in range (len(learning_rates)):
        # Create a copy of the existing dataframe
        momentum_label = momentum_values[a]
        learning_label = learning_rates[b]
        
        # Create a label string
        label = f"Binary, SGD, momentum = {momentum_label}, lr = {learning_label}"
        # Add label as a new column
        
        labels.append(label)
        # Append the new dataframe to the list
        

sgd_labels = pd.DataFrame(labels, columns=['Label'])
# Concatenate all the new dataframes into a single dataframe

labels = []

for a in range (len(weight_decay_val)):
    for b in range (len(learning_rates)):
        # Create a copy of the existing dataframe
        weight_label = weight_decay_val[a]
        learning_label = learning_rates[b]
        
        # Create a label string
        label = f"Binary, Adam, weight_decay = {weight_label}, lr = {learning_label}"
        # Add label as a new column
        
        labels.append(label)
        # Append the new dataframe to the list
        
adam_labels = pd.DataFrame(labels, columns=['Label'])     
        
        
        
labels = []

for a in range (len(weight_decay_val)):
    for b in range (len(learning_rates)):
        # Create a copy of the existing dataframe
        weight_label = weight_decay_val[a]
        learning_label = learning_rates[b]
        
        # Create a label string
        label = f"Binary, RMSprop, weight_decay = {weight_label}, lr = {learning_label}"
        # Add label as a new column
        
        labels.append(label)
        # Append the new dataframe to the list
        
rms_labels = pd.DataFrame(labels, columns=['Label'])




"""
Plot For SGD
"""
column_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df_sgd.columns = column_names

df_sgd['Label'] = sgd_labels
melted_SGD_df = df_sgd.melt(id_vars='Label', var_name='Epoch', value_name='Loss')


plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_SGD_df, x='Epoch', y='Loss', hue='Label', marker='o')

# Optional: Set labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for Each Hyperparameter Across 10 Epochs')

plt.grid(True)  # Add grid for better visualization
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

plt.show()

"""
Plot For Adam

"""

column_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df_adam.columns = column_names
df_adam = df_adam.reset_index(drop = True)
df_adam['Label'] = adam_labels
melted_adam_df = df_adam.melt(id_vars='Label', var_name='Epoch', value_name='Loss')


plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_adam_df, x='Epoch', y='Loss', hue='Label', marker='o')

# Optional: Set labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for Each Hyperparameter Across 10 Epochs')

plt.grid(True)  # Add grid for better visualization
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

plt.show()

"""
Plot For RMSprop

"""

column_names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
df_rmsprop.columns = column_names
df_rmsprop = df_rmsprop.reset_index(drop = True)
df_rmsprop = df_rmsprop[(df_rmsprop <= 2).all(axis=1)]
df_rmsprop = df_rmsprop.merge(rms_labels, left_index = True, right_index = True)

melted_rmsprop_df = df_rmsprop.melt(id_vars='Label', var_name='Epoch', value_name='Loss')


plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_rmsprop_df, x='Epoch', y='Loss', hue='Label', marker='o')

# Optional: Set labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss for Each Hyperparameter Across 10 Epochs')

plt.grid(True)  # Add grid for better visualization
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside the plot

plt.show()

"""
Find lowest column and average 
"""

lowest_col_sgd = df_sgd.iloc[[lowest_last_col_index_sgd]]
lowest_avg_sgd = df_sgd.iloc[[lowest_avg_row_index_sgd]]


lowest_col_adam = df_adam.iloc[[lowest_last_col_index_adam]]
lowest_avg_adam = df_adam.iloc[[lowest_avg_row_index_adam]]

lowest_col_rmsprop = df_rmsprop.iloc[[lowest_last_col_index_rmsprop - 6]]
lowest_avg_rmsprop = df_rmsprop.iloc[[lowest_avg_row_index_rmsprop]]


lowest_df2 = pd.concat([ lowest_avg_sgd, 
                        lowest_avg_adam, 
                        lowest_avg_rmsprop])

"""
Plot Lowest Columns

"""

lowest_df = pd.concat([lowest_df1, lowest_df2])

melted_lowest_df1 = lowest_df1.melt(id_vars='Label', var_name='Epoch', value_name='Loss')
melted_lowest_df2 = lowest_df2.melt(id_vars='Label', var_name='Epoch', value_name='Loss')


plt.figure(figsize=(10, 6))
sns.lineplot(data=melted_lowest_df1, x='Epoch', y='Loss', hue='Label', marker='o', linestyle='dashed', palette = 'deep')
sns.lineplot(data=melted_lowest_df2, x='Epoch', y='Loss', hue='Label', marker='o', palette = 'deep')
plt.axvline(x=3, color='black', linestyle=':', label='Epoch 3')
# Optional: Set labels and title
plt.xlabel('Epoch',fontsize=20)
plt.ylabel('Cross Entropy Loss',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylim(0, 3)
plt.grid(True)  # Add grid for better visualization
handles, labels = plt.gca().get_legend_handles_labels()
# Modify the legend handle for lowest_df1 to have a dashed linestyle
handles[0].set_linestyle('--')
handles[1].set_linestyle('--')
handles[2].set_linestyle('--')
plt.legend(handles, labels, loc='upper right', fontsize=12) # Move legend outside the plot

plt.show()














