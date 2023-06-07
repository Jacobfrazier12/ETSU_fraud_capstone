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

import sys
import csv
args = sys.argv
if args == None or len(args)<2:
    raise SystemExit("Must pass 2 (two) arguments: y/n to run the model training and y/n to show the analysis of the data.")
training = args[1]
get_info = args[2]
data2_path = os.path.join(os.getcwd(), "data2.csv")
data_path = os.path.join(os.getcwd(), "data.csv")
cols = ["distance_from_home","distance_from_last_transaction","ratio_to_median_purchase_price","repeat_retailer","used_chip","used_pin_number","online_order","fraud"]


if str.lower(training) == "y":
    data = pd.read_csv(data_path, sep="|")
    data = data[cols]
    data2 = pd.read_csv(data2_path)
    data2 = data2[cols]
    not_fraud_df = data[data["fraud"]==0]
    fraud_df = data[data["fraud"]==1]
    not_fraud_under_sample = resample(not_fraud_df, replace=False, n_samples=len(fraud_df), random_state=42)
    data = pd.concat([not_fraud_under_sample, fraud_df])
   
    
    
    y = data["fraud"].values
    data = data.drop(labels = ["fraud"], axis = 1)
    x = data.values
    sgd = SGDClassifier()
    gaussian_nb = GaussianNB()
    multinomial_nb = MultinomialNB()
    knn = KNeighborsClassifier()
    nearest_centroid = NearestCentroid()
    random_forest = RandomForestClassifier()
    logistic_regression = LogisticRegression()
    logistic_regression_cv = LogisticRegressionCV()
    decision_tree = DecisionTreeClassifier()
    neural_network =  classifier = MLPClassifier(hidden_layer_sizes = (500,500, 500), activation="tanh", random_state = 42, solver = "adam", max_iter=5000)
    classifiers = ((sgd, "Stochastic Gradient Decent"), (gaussian_nb, "Gaussian Naive Bayes"), (multinomial_nb, "Multinomial Naive Bayes"), (knn, "K-Nearest Neighbor"), (nearest_centroid, "Nearest Centroid"), (random_forest, "Random Forest"), (logistic_regression, "Logistic Regression"),(logistic_regression_cv, "Logistic Regression CV"), (decision_tree, "Decision Tree"))
    x_train, x_test, y_train, y_true = train_test_split(x, y, test_size = 0.20, random_state=42)
    
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
            
if str.lower(get_info) == "y":
    data = pd.read_csv(data_path, sep="|", low_memory=False)
    data2 = pd.read_csv(data2_path, low_memory=False)
    num_total_rows_synthetic = len(data) 
    num_total_rows_kaggle = len(data2) 
    num_fraud_synthetic = len(data[data["fraud"]==1])
    num_not_fraud_synthetic = len(data[data["fraud"]==0])
    num_fraud_kaggle = len(data2[data2["fraud"]==1])
    num_not_fraud_kaggle = len(data2[data2["fraud"]==0])
    fraud_df = data[data["fraud"]==1]
    not_fraud_df = data[data["fraud"]==0]
    data_after_under_sample_not_fraud = resample(not_fraud_df, replace=False, n_samples=len(fraud_df))
    data = pd.concat([fraud_df, data_after_under_sample_not_fraud])
    num_fraud_after_sample = len(data[data["fraud"]==0])
    num_not_fraud_after_sample = len(data[data["fraud"]==1])
    print("Total number of rows in synthetic data: ", f"{num_total_rows_synthetic:,}")
    print("Total number of fraud cases in synthetic data: ", f"{num_fraud_synthetic:,}")
    print("Total number of non-fraud cases in synthetic data: ", f"{num_not_fraud_synthetic:,}")
    print("Total number of rows in Kaggle data: ", f"{num_total_rows_kaggle:,}")
    print("Total number of fraud cases: ", f"{num_fraud_kaggle:,}")
    print("Total number of non-fraud cases: ", f"{num_not_fraud_kaggle:,}")
    print("Total number of fraud cases after under sampling: ", f"{num_fraud_after_sample:,}")
    print("Total number of non-fraud cases after under sample: ", f"{num_not_fraud_after_sample:,}")
  
    labels = ["Fraud", "Non-Fraud"]
    sizes = [num_fraud_synthetic, num_not_fraud_synthetic]
    plt.title("Fraud V. Non-Fraud")
    plt.pie(sizes, labels = labels, autopct = "%1.1f%%", pctdistance = 1.2, labeldistance = 1.4)
    plt.show()
    plt.title("Fraud V. Non-Fraud after under-sampling")
    sizes = [num_fraud_after_sample, num_not_fraud_after_sample]
    plt.pie(sizes, labels = labels, autopct = "%1.1f%%", pctdistance = 1.2, labeldistance = 1.4, startangle = 90)
    plt.show()


   