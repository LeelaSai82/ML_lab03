#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel(r"C:\Users\saite\Downloads\embeddingsdata (1).xlsx")
df


# In[4]:


class_a_data = df[df['Label'] == 0]
class_b_data = df[df['Label'] == 1]
intra_class_var_a = np.var(class_a_data[['embed_0', 'embed_1']], ddof=1)
intra_class_var_b = np.var(class_b_data[['embed_0', 'embed_1']], ddof=1)
mean_class_a = np.mean(class_a_data[['embed_0', 'embed_1']], axis=0)
mean_class_b = np.mean(class_b_data[['embed_0', 'embed_1']], axis=0)
inter_class_distance = np.linalg.norm(mean_class_a - mean_class_b)
print(f"mean of class A: {mean_class_a}")
print(f"mean of class B: {mean_class_b}")
print(f'Intraclass spread (variance) for Class A: {intra_class_var_a}')
print(f'Intraclass spread (variance) for Class B: {intra_class_var_b}')
print(f'Interclass distance between Class A and Class B: {inter_class_distance}')


# In[8]:


#numeric_mean=df.mean()
num_bins=5
plt.hist(df, bins=num_bins, edgecolor='k')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Feature')

# Calculate the mean and variance
mean = np.mean(df)
variance = np.var(df)

print(f"Mean: {mean}")
print(f"Variance: {variance}")

plt.show()

import pandas as pd
vector1 = df.loc[0, ["embed_1", "embed_2"]].values
vector2 = df.loc[1, ["embed_1", "embed_2"]].values
print("Feature Vector 1:")
print(vector1)
print("\nFeature Vector 2:")
print(vector2)

# In[13]:


def minkowski_distance(x, y, r):
    return np.sum(np.abs(x - y) ** r) ** (1 / r)
values_r = range(1, 11)
distances = [minkowski_distance(vector1, vector2, r) for r in values_r]
print(distances)
plt.figure(figsize=(10, 7))
plt.plot(values_r, distances, marker='o', linestyle='-')
plt.title("Minkowski Distance vs Values of r")
plt.xlabel("r")
plt.ylabel("Minkowski Distance")
plt.grid(True)
plt.show()


# In[17]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Define your feature vectors (X) and class labels (y)
# Example:
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']  # Example class labels

# Split your data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a k-NN classifier with k = 3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training set
neigh.fit(X_train, y_train)

# Test the accuracy of the k-NN classifier on the test set
accuracy = neigh.score(X_test, y_test)

# Print the accuracy
print(f"Accuracy: {accuracy}")


# In[18]:

# Import the necessary libraries (if not already imported)
from sklearn.neighbors import KNeighborsClassifier

# Assuming you have already trained your 'neigh' classifier
# If not, make sure you fit it to your training data before making predictions
test_vector_index = 0
predicted_class = neigh.predict([X_test.iloc[test_vector_index]])  # Pass a list containing the test vector

print("Predicted class for test vector {}: {}".format(test_vector_index, predicted_class))


# In[17]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# Creating lists so as to store accuracy values for kNN and NN classifiers
accuracies_kNN = []
accuracies_NN = []

#k varies from 1 to 11
k_values = range(1, 11)

for k in k_values:
    # Train the kNN classifier with the current k value
    kNN_classifier = KNeighborsClassifier(n_neighbors=k)
    kNN_classifier.fit(X_train, y_train)
    y_pred_kNN = kNN_classifier.predict(X_test)
    
    # Calculating accuracy for kNN classifier
    accuracy_kNN = accuracy_score(y_test, y_pred_kNN)
    accuracies_kNN.append(accuracy_kNN)
    # Train the NN classifier for k=1
    NN_classifier = KNeighborsClassifier(n_neighbors=1)
    NN_classifier.fit(X_train, y_train)
    y_pred_NN = NN_classifier.predict(X_test)
        
    # Calculating accuracy for NN where k=1
    accuracy_NN = accuracy_score(y_test, y_pred_NN)
    accuracies_NN.append(accuracy_NN)



# Creating an accuracy plot
plt.figure(figsize=(15, 8))
plt.plot(k_values, accuracies_kNN, marker='o', label='kNN (k=3)')
plt.plot(k_values, accuracies_NN, marker='o', label='NN (k=1)')

plt.title('Accuracy vs. k Value')
plt.xlabel('k Value')
plt.ylabel('Accuracy')
plt.xticks(k_values)
plt.legend()
plt.grid(True)
plt.show()



# In[ ]:

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
# Assuming you have already trained your k-NN classifier (neigh) and have X_train and y_train for training data,
# and X_test and y_test for test data.
# Train the k-NN classifier
neigh.fit(X_train, y_train)
# Predictions for training and test data
y_train_pred = neigh.predict(X_train)
y_test_pred = neigh.predict(X_test)
# Confusion matrix for training data
confusion_matrix_train = confusion_matrix(y_train, y_train_pred)
# Confusion matrix for test data
confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
# Precision, recall, and F1-score for training data
precision_train = precision_score(y_train, y_train_pred, average='weighted')
recall_train = recall_score(y_train, y_train_pred, average='weighted')
f1_score_train = f1_score(y_train, y_train_pred, average='weighted')

# Precision, recall, and F1-score for test data
precision_test = precision_score(y_test, y_test_pred, average='weighted')
recall_test = recall_score(y_test, y_test_pred, average='weighted')
f1_score_test = f1_score(y_test, y_test_pred, average='weighted')

# Print confusion matrix and performance metrics
print("Confusion Matrix (Training Data):")
print(confusion_matrix_train)
print("\nConfusion Matrix (Test Data):")
print(confusion_matrix_test)

print("\nPerformance Metrics (Training Data):")
print("Precision:", precision_train)
print("Recall:", recall_train)
print("F1-Score:", f1_score_train)

print("\nPerformance Metrics (Test Data):")
print("Precision:", precision_test)
print("Recall:", recall_test)
print("F1-Score:", f1_score_test)


