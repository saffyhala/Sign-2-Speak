import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from pickle file
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Ensure all elements in 'data' have the same shape
data = []
labels = []
for i, label in zip(data_dict['data'], data_dict['labels']):
    if len(i) == len(data_dict['data'][0]):
        data.append(np.array(i, dtype=object))
        labels.append(label)

# Convert lists to numpy arrays
data = np.array(data, dtype=object)
labels = np.array(labels)

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()