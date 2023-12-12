import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

class LinearRegression:
    def __init__(self):
            self.w = None

    def BGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,batch_size = 1,flag = 0):
        np.random.seed(42)
        self.w = np.random.randn(1,num_features+1) #randomly initialize weights. Row vector of size (1,num_features+1)
        N = X_train.shape[0]
        M = X_test.shape[0]
        history = dict()
        train_loss_after_each_epoch = []
        test_loss_after_each_epoch = []
        for i in range(epochs):
            for j in range(0,N,batch_size):
                if(j + batch_size > N):
                    nf = N-j+1
                else:
                    nf = batch_size
                target = y_train[0:1,j:min(j+batch_size,N)]
                prediction = np.matmul(self.w,X_train[j:min(j+batch_size,N),:].T)
                #Gradient of loss function
                grad = -1*np.matmul(target - prediction,X_train[j:min(j+batch_size,N),:])/nf
                #update step
                self.w = self.w - lr*grad
            train_loss = np.mean(np.square(y_train - np.matmul(self.w,X_train.T)))
            if(y_test is not None):
                test_loss = np.mean(np.square(y_test - np.matmul(self.w,X_test.T)))
            train_loss_after_each_epoch.append(train_loss)
            if(y_test is not None):
                test_loss_after_each_epoch.append(test_loss)
            if(flag):
                if(y_test is not None):
                    print('epoch: {} ----- train_loss: {}   test_loss: {}'.format(i+1,train_loss,test_loss))
                else:
                    print('epoch: {} ----- train_loss: {} '.format(i+1,train_loss))
        print('Done !')
        history['weights'] = self.w
        history['train_loss'] = train_loss_after_each_epoch
        history['test_loss'] = test_loss_after_each_epoch
        return history


                
            
    def MSE(self,X_test,y_test):
        loss = np.mean(np.square(y_test - np.matmul(self.w,X_test.T)))
        return loss

    def MAE(self,X_test,y_test):
        loss = np.mean(np.abs(y_test - np.matmul(self.w,X_test.T)))
        return loss
    
    def corr(self,X_test,y_test):
        return np.corrcoef(np.matmul(self.w,X_test.T),y_test)[0][1]
    
    def predict(self,X):
        return np.matmul(self.w,X.T)
    

def preprocess(train,test,sampling_rate = 1):
    train = train.sample(frac = sampling_rate,random_state = 42) #take random frac % samples from the data
    y_train = train['t']    
    X_train = train.drop('t',axis = 1)

    train_data_mean = X_train.mean()
    train_data_std = X_train.std()
    #Normalize features
    X_train = (X_train - train_data_mean)/train_data_std
    #Adding ones for the bias term
    ones = np.ones(X_train.shape[0])
    X_train['x0'] = ones
    
    y_train = y_train.to_numpy()
    X_train = X_train.to_numpy()
    y_train = y_train.reshape([1,y_train.shape[0]])

    y_test = None
    if('t' in test.columns):
        y_test = test['t']
    if(y_test is not None):
        X_test = test.drop('t',axis = 1,errors = 'ignore')
    else:
        X_test = test
    #Transfroming test data
    X_test = (X_test - train_data_mean)/train_data_std
    #Adding ones for the bias term
    ones = np.ones(X_test.shape[0])
    X_test['x0'] = ones
    if(y_test is not None):
        y_test = y_test.to_numpy()
    X_test = X_test.to_numpy()
    if(y_test is not None):
        y_test = y_test.reshape([1,y_test.shape[0]])
    
    return X_train,y_train,X_test,y_test




# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-tr", "--train", help = "train data file path")
parser.add_argument("-te", "--test", help = "test data file path")
parser.add_argument("-o", "--out", help = "output file path")




# Read arguments from command line
args = parser.parse_args()




train_file_path = args.train
test_file_path = args.test

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)

X_train,y_train,X_test,y_test = preprocess(train,test)

print("#### Running the Model that gives the most Optimal Weights ####")
model_BGD = LinearRegression()
model_BGD_history = model_BGD.BGD(X_train,y_train,X_test,y_test,10000,0.1,6,64)

predictions = model_BGD.predict(X_test)

out_file = args.out

with open(out_file, 'w') as my_file:
        for i in predictions:
            np.savetxt(my_file,i,fmt = '%.9f')

print('Optimal Weights(Note the last element is the Bias term) :',model_BGD.w)
if(y_test is not None):
    print('Mean Squared Loss on test data:',model_BGD.MSE(X_test,y_test))
    print('Mean Absolute Loss on test data:',model_BGD.MAE(X_test,y_test))
print('Mean Squared Loss on train data:',model_BGD.MSE(X_train,y_train))
print('Mean Absolute Loss on train data:',model_BGD.MAE(X_train,y_train))