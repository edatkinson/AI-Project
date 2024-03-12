import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('metrics.csv')
zf = pd.read_csv('metricsepoch2-4.csv')

batches = zf['batch']
batch_length = len(batches)

train_acc = zf['train_accuracy']
test_acc = zf['test_accuracy']

# batches = df['batch']
# batch_length = len(batches)

# train_acc = df['train_accuracy']
# test_acc = df['test_accuracy']
#fill in the Nan values with the previous value
test_acc = test_acc.fillna(method='ffill')
train_loss = df['train_loss']

plt.plot(range(batch_length), train_acc, label='Training Accuracy')
plt.plot(range(len(test_acc)), test_acc, label='Testing Accuracy')
plt.xlabel('Batch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.show()

plt.plot(range(batch_length), train_loss, label='Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()


