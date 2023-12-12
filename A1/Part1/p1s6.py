import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def preprocess(train):
    y_train = train['t']    
    X_train = train.drop('t',axis = 1)

    #Adding ones for the bias term
    ones = np.ones(X_train.shape[0])
    X_train['x0'] = ones
    
    y_train = y_train.to_numpy()
    X_train = X_train.to_numpy()
    y_train = y_train.reshape([1,y_train.shape[0]])
    
    return X_train,y_train



# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-tr", "--train", help = "train data file path")

# Read arguments from command line
args = parser.parse_args()

train_file_path = args.train

train = pd.read_csv(train_file_path)

X_train,y_train = preprocess(train)

w_ml = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T,X_train)),X_train.T),y_train.T)

var = np.mean(np.square(y_train - np.matmul(w_ml.T,X_train.T)))

#Strategy in the Report

print('Variance in the noise :', var) #5.4548751265798506e-14