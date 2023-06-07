# Overview  

- This is a credit card fraud detection system that is written in Python.

- Out of curiosity, I began to wonder if I could take two unrelated credit card datasets and use them to create a fraud detection system.
- The first dataset was synthetically generated by the [Sparkov Data Generation Tool](https://github.com/namebrandon/Sparkov_Data_Generation.git).

- The second dataset is [hosted on Kaggle](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud) 

- Because the Sparkov data is randomly generated, the results varied, sometimes being outstanding and at other times, less than ideal. 

# License

**You are free to distribute, copy, or modify this software, with or without credit towards the author; however, the author bares no responsibility for any damages sustained while using this software. Also, the software is offered "as-is."**

# Models used:
- Stochastic Gradient Decent
- Gaussian Naive Bayes
- Multi-nomial Naive Bayes
- K-Nearest Neighbor
- Nearest Centroid
- Random Forest
- Logistic Regression
- Logistic Regression CV
- Decision Tree

# Approach 

Since neither dataset contain columns similar to one-another, I implemented an Extract, Transform, and Load (ETL) process.

After running the Sparkov tool, I extracted all the records, which were spread-out across multiple files, into a single-unified CSV file.

I then began transforming the unified Sparkov data, so it would match the Kaggle dataset. Upon completion, the datasets conformed to the following schema:
- distance from home
- distance from last transaction
- ratio to median purchase price
- used chip 
- used pin number
- online order
- repeat retailer
- fraud

After examining the results, I realized that the machine learning models were failing because the data varied in size, so scaling was used to scale the data.

The models also suffered from biases towards non-fraudulent transactions due to the imbalance between non-fraudulent and fraudulent, so a subset of non-fraudulent transactions was extracted and merged with the fraudulent transactions.

After training each model, I tested each model with the Kaggle dataset, which was unseen data. The results weren't always favorable because because the models were trained on randomly generated data.

# Running the software
1. Ensure you have Python 3, Pip 3, and GNU Make installed.

2. Run `make init` && `make run`

- `make init` will install all required dependencies, via Pip; clone the Sparkov tool; and download the Kaggle dataset from Google Drive.



