import pandas as pd
import numpy as np
from data_processing import train_test_spilt, standardization
import logistic_regression

heart_dataset = pd.read_csv(r'/Users/harshitsaini/Desktop/Project/heart.csv')

# print(heart_dataset.shape)

# print(heart_dataset['output'].value_counts())

features = heart_dataset.drop(columns='output', axis=1)
targets = heart_dataset['output']


standardized_data = (features-features.mean())/features.std()
features = standardized_data

features = standardization(features)

X_train, X_test, y_train, y_test = train_test_spilt(0.8, features, targets)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model = logistic_regression.Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
model.fit(X_train, y_train)

# accuracy score on the training data
X_train_prediction = model.predict(X_train)
accuracy = np.sum(np.equal(y_train, X_train_prediction)) / len(X_train_prediction)

print('Accuracy score of the training data : ', accuracy)
