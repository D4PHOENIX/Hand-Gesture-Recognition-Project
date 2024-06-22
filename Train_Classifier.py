import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# Load the data
data_dict = pickle.load(open('D:\\University\\Programming\\Python\\AI Projects\\Sign Language Translator\\data.pickle', 'rb'))

# Verify data consistency
data = data_dict['data']
labels = data_dict['labels']

# Ensure all entries in `data` have the same length
data = [d for d in data if len(d) == 42]

# Convert to numpy arrays
data = np.asarray(data)
labels = np.asarray(labels[:len(data)])  # Ensure labels match the filtered data

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print('Model saved successfully!')
