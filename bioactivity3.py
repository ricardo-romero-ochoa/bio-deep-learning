#!/usr/bin/env python
# coding: utf-8

# # Bioactivity classification model
# ## Dr. Ricardo Romero
# ### Natural Sciences Department, UAM-C

# In[1]:


# Load libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from rdkit import Chem
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


# In[2]:


# Load dataset (available from 10.34740/kaggle/dsv/8626441)
df = pd.read_csv('kinase_data_final.csv')
df


# In[3]:


# Convert the SMILES strings to RDKit molecules
molecules = [Chem.MolFromSmiles(smiles) for smiles
             in df['canonical_smiles']]

# Generate rdKit fingerprints for each molecule
fingerprints = [Chem.RDKFingerprint(molecule) for molecule in molecules]

# Convert the bioactivity data to binary classification labels
labels = [1 if bioactivity >= 6 else 0 for bioactivity in df['pIC50']]
#labels = [1 if bioactivity > 6 else 2 if bioactivity >= 5 else 0 for bioactivity in df['pIC50']]


# In[4]:


# Define features and target

X = np.concatenate((df.iloc[:, 4:8].values, fingerprints), axis=1)
y = np.array(labels)  # Convert labels to numpy array for compatibility with Keras

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for CNN (CNNs expect a 3D input: samples, timesteps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# In[5]:


# Define the CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv1D(16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# In[6]:


# Initialize the model
model = create_cnn_model()


# In[7]:


# Early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)


# In[8]:


# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                    callbacks=[early_stopping, reduce_lr])


# In[9]:


# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# In[10]:


# Predict probabilities
y_pred_prob = model.predict(X_test).ravel()

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# In[11]:


# Compute confusion matrix
y_pred = (y_pred_prob > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

# Plot confusion matrix
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


# In[12]:


# Print evaluation metrics
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')


# In[13]:


# Print the classification report
print(classification_report(y_test, y_pred))


# In[14]:


# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 Score:', f1)


# In[15]:


# Save the trained CNN model to a file
model.save('kinase_model.keras')
model.save('kinase_model.h5')


# In[16]:


# Cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_results = []

for train, val in kfold.split(X_scaled, y):
    model = create_cnn_model()
    history = model.fit(X_scaled[train], y[train], epochs=50, batch_size=32, verbose=0, validation_data=(X_scaled[val], y[val]), callbacks=[early_stopping])
    scores = model.evaluate(X_scaled[val], y[val], verbose=0)
    cross_val_results.append(scores[1])  # append accuracy

print(f'Cross-validated accuracy: {np.mean(cross_val_results)} ± {np.std(cross_val_results)}')

