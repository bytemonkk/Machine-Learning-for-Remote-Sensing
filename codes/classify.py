# load The libraries and The dataset dobby..!
print('logistic_regresssion')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("E:/case study/merged_dataset.csv")

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# split The dataset dobby..!

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling..! dobby 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# let's Train The model using logistic_regression dobby

from sklearn.linear_model import LogisticRegression
logistic_classifier = LogisticRegression(random_state=42, max_iter=2000, solver='lbfgs')
logistic_classifier.fit(x_train, y_train)

y_pred = logistic_classifier.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# let's Evaluate The model dobby..!

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_s_log = accuracy_score(y_test, y_pred)

print(cm)
print(f"Logistic_classification: {acc_s_log:.4f}")

print('\n')
print('Random_forest_classification')

from sklearn.ensemble import RandomForestClassifier
random_forest_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
random_forest_classifier.fit(x_train, y_train)

y_pred = random_forest_classifier.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# let's Evaluate The model dobby..!

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_s_randomForest = accuracy_score(y_test, y_pred)

print(cm)
print(f"randomForest: {acc_s_randomForest:.4f}")

print('\n')
print("svm's my favourite dobby..!")
    
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear', random_state=0)
svm_classifier.fit(x_train, y_train)

y_pred = svm_classifier.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc_s_svm = accuracy_score(y_test, y_pred)

print(cm)
print(f"svm: {acc_s_svm:.4f}")
    
print('\n')
print('xgboost classification')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y) 
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


from xgboost import XGBClassifier
xgb_classifier = XGBClassifier(eval_metric='mlogloss', random_state=42)
xgb_classifier.fit(x_train, y_train)

y_pred = xgb_classifier.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
acc_s_xgboost = accuracy_score(y_test, y_pred)

print(cm)
print(f"xgboost: {acc_s_xgboost:.4f}")
    
print('\n')
print("Neural Networks damn favourite dobby..!")

from sklearn.neural_network import MLPClassifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,),  
                               max_iter=500,                
                               activation='relu',          
                               solver='adam',              
                               random_state=42)
mlp_classifier.fit(x_train, y_train)
y_pred = mlp_classifier.predict(x_test)

print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

cm = confusion_matrix(y_test, y_pred)
acc_s_neuralnetwork = accuracy_score(y_test, y_pred)

print(cm)
print(f"Neural_Networks: {acc_s_neuralnetwork:.4f}")