import os
import argparse
from shutil import rmtree
import subprocess
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample
import math
keep_original_files = False
cols = ["distance_from_home","distance_from_last_transaction","ratio_to_median_purchase_price","repeat_retailer","used_chip","used_pin_number","online_order","fraud"]
scaler = MinMaxScaler()
ignore_files = ["data.csv", "data2.csv", "demographic.csv", "card_transdata.csv", "original_data.csv"]
files_not_to_delete = ["data.csv", "data2.csv", "demographic.csv", "original_data.csv"]
def determine_if_chip(row):
    if "net" in str(row["category"]) and "gas" in str(row["category"]):
        return 0
    else: return 1
def determine_if_repeat(row):
    if row["merchant"] == 0:
        return 1
    else: 
        return 0

def determine_if_online(row):
    if "net" in str(row["category"]):
        return 1
    else:
        return 0



def calculate_distance_from_home(row):
    
    R = 6371
    #R = 3963 

    lat1, lon1 = (row["lat"], row["long"])
    lat2, lon2 = (row["merch_lat"], row["merch_long"])

   
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

   
    d_lon = lon2 - lon1 

   
    d_lat = lat2 - lat1 

   
    a = math.pow(math.sin(d_lat / 2), 2) + math.cos(lat1) * math.cos(lat2) * math.pow(math.sin(d_lon / 2), 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist = R * c 

    return dist
   





def combine_data(ignore_files: list, path: str):
    comm = "cd ./Sparkov_Data_Generation && python3 datagen.py -n 15000 -o ../credit_card_fraud 01-01-2020 08-05-2020"
    subprocess.run(comm, shell=True)
    list_of_files = os.listdir(os.path.join(os.getcwd(), "credit_card_fraud"))
    with open(path, "w+") as fp:
        pass
    first = True
  

    for root, dirs, files in os.walk(os.path.join(os.getcwd(), "credit_card_fraud")):
       
       for file in files:
            if first:
                temp_df = pd.read_csv(os.path.join(root, file), delimiter="|", low_memory=False)
                temp_df.to_csv(path, index=False, sep="|", mode="a", header=True)
                first = False
                
                    
                    
            if not first:
                temp_df = pd.read_csv(os.path.join(root, file), delimiter="|", low_memory=False)
                temp_df.to_csv(path, index=False, sep="|", mode="a", header=False)
    rmtree(os.path.join(os.getcwd(), "credit_card_fraud"))              
              
                
    
def clean_files(files_not_to_delete: list, data_path: str, demographic_path: str):

    
    
     if os.path.exists(data_path):
        data = pd.read_csv(data_path, sep="|", low_memory=False)
        data = data.iloc[:, [9, 10, 14,16,19, *range(20,26)]]
        data.insert(len(data.columns), "distance_from_home", None)
        print("Calculating distance from home. \n")
        data["distance_from_home"] = data.apply(lambda row: calculate_distance_from_home(row), axis=1)
        print("Done calculating distance from home. \n")
        data = data.sort_values(["acct_num","unix_time"], ascending=True)
        print("Calculating distance from last transaction. \n")
        data = data.iloc[:, [*range(2,9), 11]]
        data["distance_from_last_transaction"] = data.groupby("acct_num")["distance_from_home"].diff().fillna(0).abs()
        print("Done calculating distance from last transaction. \n")
        print("Calculating ratio to median purchase price. \n")
        data["ratio_to_median_purchase_price"] = data["amt"]/data.groupby("acct_num")["amt"].transform("median")
        print("Done calculating ratio to median purchase price. \n")
        print("Determining if card pin was used. \n")
        data["used_pin_number"] = 0
        print("Done determining if card pin was used. \n")
        data.insert(len(data.columns), "used_chip", None)
        print("Determining if card chip was used. \n")
        data["used_chip"] = data.apply(lambda row: determine_if_chip(row), axis=1)
        print("Done determining if card chip was used. \n")
        print("Determining if a purchase was an online-order. \n")
        data["online_order"] = data.apply(lambda row: determine_if_online(row), axis=1)
        print("Done determining if a purchase was an online-order. \n")
        data = data.sort_values(["acct_num","unix_time"], ascending=True)
        print("Determining if a purchase was a repeat-retailer. \n")
        data = data.sort_values(["acct_num","unix_time"], ascending=True)
        label_encoder = LabelEncoder()
        data["merchant"] = label_encoder.fit_transform(data["merchant"])
        data["repeat_retailer"] = data.groupby("acct_num")["merchant"].diff().fillna(0).abs()
        data["repeat_retailer"] =  data.apply(lambda row: determine_if_repeat(row), axis = 1 )
        print("Done determining if a purchase was a repeat-retailer. \n")
        print("Saving is_fraud to fraud. \n")
        data["fraud"] = data["is_fraud"]
        print("Done saving is_fraud to fraud. \n")
        print("Dropping category. \n")
        data = data.drop(["category", "acct_num", "amt", "trans_num", "unix_time", "is_fraud", "merchant"], axis=1)
        print("Done dropping category. \n")
        print("Dropping null rows. \n")
        data = data.dropna(axis=0)
        print("Done dropping null rows. \n")
        print(f"Saving and scaling data to {data_path}\n")
        data[["distance_from_home", "distance_from_last_transaction",]] = scaler.fit_transform(data[["distance_from_home", "distance_from_last_transaction"]])
        data.to_csv(data_path, header=True, index=False, mode="w+", sep="|")
        data2 = pd.read_csv(data2_path)
        data2 = data2[cols]
        data2[["distance_from_home", "distance_from_last_transaction"]] = scaler.fit_transform(data2[["distance_from_home", "distance_from_last_transaction"]])
        data2.to_csv(data2_path, header=True, index=False, mode="w+")
        print("Done saving. \n")



data_path = os.path.join(os.getcwd(),"data.csv")
data2_path = os.path.join(os.getcwd(),"data2.csv")
demographic_path = os.path.join(os.getcwd(), "demographic.csv")
cleaned_data_path = os.path.join(os.getcwd(), "cleaned_credit_card_data.csv")
combine_data(ignore_files, data_path)
clean_files(files_not_to_delete, data_path, demographic_path)



