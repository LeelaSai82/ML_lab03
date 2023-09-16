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
from sklearn.model_selection import train_test_split

# Assuming X contains your feature vectors and y contains your class labels
# X should be a 2D array where each row is a feature vector, and y should be a 1D array or list of class labels

# Split the data into training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# X_train: Training feature vectors
# X_test: Test feature vectors
# y_train: Training class labels
# y_test: Test class labels

# You can adjust the test_size parameter to set the size of your test set (e.g., 0.3 for a 70-30 split)

# The random_state parameter is set for reproducibility, you can change it or remove it if you don't need reproducibility


# In[18]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Assuming you have already split your data into a training set (X_train, y_train)

# Create a k-NN classifier with k = 3
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)


# In[20]:


pip install --upgrade scikit-learn numpy


# In[18]:


import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Assuming you have already split your data into a training set (X_train, y_train)

# Create a k-NN classifier with k = 3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training set
neigh.fit(X_train, y_train)

# Test the accuracy of the k-NN classifier on the test set
accuracy = neigh.score(X_test, y_test)

# Print the accuracy
print(f"Accuracy: {accuracy}")


# In[17]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Define your feature vectors (X) and class labels (y)
# Example:
vector1 = df['embed_0'].astype(int)
vector2 = df['embed_5'].astype(int)

# Create a 2D array by stacking the vectors vertically
X = np.vstack((vector1, vector2)).T  # Transpose for proper shape
y = np.array([0, 1])  # Example class labels

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


# In[ ]:




