import pandas as pd
import logistic_regression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

heart_dataset = pd.read_csv(r'D:\Project\heart.csv')

df1 = df
# define the columns to be encoded and scaled
cat_cols = ['sex', 'exng', 'caa', 'cp', 'fbs', 'restecg', 'slp', 'thall']
con_cols = ["age", "trtbps", "chol", "thalachh", "oldpeak"]

# # encoding the categorical columns
# df1 = pd.get_dummies(df1, columns=cat_cols, drop_first=True)

# defining the features and target
features = df1.drop(['output'], axis=1)
targets = df1[['output']]

scaler = StandardScaler()
scaler.fit(features)
standardized_data = scaler.transform(features)
print(standardized_data)

features = standardized_data
target = df1['output']

# # instantiating the scaler
# scaler = RobustScaler()
#
# # scaling the continuous featuree
# features[con_cols] = scaler.fit_transform(features[con_cols])
# print("The first 5 rows of X are")
# features.head()

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=2)
print("The shape of X_train is      ", X_train.shape)
print("The shape of X_test is       ", X_test.shape)
print("The shape of y_train is      ", y_train.shape)
print("The shape of y_test is       ", y_test.shape)


classifier = logistic_regression.Logistic_Regression(learning_rate=0.01, no_of_iterations=1000)
classifier.fit(X_train, y_train)



#
# # accuracy score on the training data
# X_train_prediction = classifier.predict(X_train)
# training_data_accuracy = accuracy_score(y_train, X_train_prediction)
#
# print('Accuracy score of the training data : ', training_data_accuracy)
