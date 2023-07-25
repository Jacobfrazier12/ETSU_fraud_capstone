import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

##This section obtains a sample of non-fraudulent transactions, so there will be an equal number of non-fraudulent and fraudulent transactions. This sampled data is split into a training and testing set and used to train each model. 
data2_path = os.path.join(os.getcwd(), "data2.csv")
data_path = os.path.join(os.getcwd(), "data.csv")
cols = ["distance_from_home", "distance_from_last_transaction", "ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number", "online_order", "fraud"]
data = pd.read_csv(data_path, sep="|")
data = data[cols]
data2 = pd.read_csv(data2_path)
data2 = data2[cols]
not_fraud_df = data[data["fraud"]==0]
fraud_df = data[data["fraud"]==1]
not_fraud_under_sample = resample(not_fraud_df, replace=True, n_samples=len(fraud_df), random_state=42)
data = pd.concat([not_fraud_under_sample, fraud_df])
y = data["fraud"].values
data = data.drop(labels = ["fraud"], axis = 1)
x = data.values
multinomial_nb = MultinomialNB()
knn = KNeighborsClassifier()
random_forest = RandomForestClassifier()
logistic_regression = LogisticRegression(random_state = 42)
logistic_regression_cv = LogisticRegressionCV(random_state = 42)
neural_network = MLPClassifier(activation="logistic", random_state = 42)
classifiers = ((multinomial_nb, "Multinomial Naive Bayes"), (knn, "K-Nearest Neighbor"), (random_forest, "Random Forest"), (logistic_regression, "Logistic Regression"),(logistic_regression_cv, "Logistic Regression CV"), (neural_network, "Neural Network"))
x_train, x_test, y_train, y_true = train_test_split(x, y, test_size = 0.20, random_state=42)
## 

## This section tests each model's ability to accurately label unseen data(i.e., the Kaggle dataset) and outputs each model's accuracy.
for classifier in classifiers:
    model = classifier[0].fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classifier[1], "\n")
    fraud_test_data = data2[data2["fraud"]==1]
    non_fraud_test_data = data2[data2["fraud"]==0]
    y_true = non_fraud_test_data["fraud"].values
    non_fraud_test_data = non_fraud_test_data.drop(["fraud"], axis=1)
    y_pred = model.predict(non_fraud_test_data.iloc[:, [0,1,2,3, 4,5, 6]].values)
    print("Testing with non-fraudulent transactions.")
    print(round(accuracy_score(y_true, y_pred),2)*100)  
    print("Testing with only fraudulent transactions.")
    y_true = fraud_test_data["fraud"].values
    fraud_test_data = fraud_test_data.drop(["fraud"], axis=1)
    y_pred = model.predict(fraud_test_data.iloc[:, [0,1,2,3, 4,5, 6]].values)
    print(round(accuracy_score(y_true, y_pred),2)*100, "\n")
            

   