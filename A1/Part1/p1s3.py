import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

class LinearRegression:
    def __init__(self):
            self.w = None

    def RidgeSGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,lambda_ = 1,flag = 0):
        np.random.seed(42)
        self.w = np.random.randn(1,num_features+1) #randomly initialize weights. Row vector of size (1,num_features+1)
        N = X_train.shape[0]
        M = X_test.shape[0]
        history = dict()
        train_loss_after_each_epoch = []
        test_loss_after_each_epoch = []
        for i in range(epochs):
            for j in range(0,N):
                target = y_train[0:1,j:j+1]
                prediction = np.matmul(self.w,X_train[j:j+1,:].T)
                #Gradient of loss function
                grad = (-1*np.matmul(target - prediction,X_train[j:j+1,:])) + lambda_*self.w
                #update step
                self.w = self.w - lr*grad
            train_loss = np.mean(np.square(y_train - np.matmul(self.w,X_train.T)))
            test_loss = np.mean(np.square(y_test - np.matmul(self.w,X_test.T)))
            train_loss_after_each_epoch.append(train_loss)
            test_loss_after_each_epoch.append(test_loss)
            if(flag):
                print('epoch: {} ----- train_loss: {}   test_loss: {}'.format(i+1,train_loss,test_loss))
        print('Done !')
        history['weights'] = self.w
        history['train_loss'] = train_loss_after_each_epoch
        history['test_loss'] = test_loss_after_each_epoch
        return history

    def LassoSGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,lambda_ = 1,flag = 0):
        np.random.seed(42)
        self.w = np.random.randn(1,num_features+1) #randomly initialize weights. Row vector of size (1,num_features+1)
        N = X_train.shape[0]
        M = X_test.shape[0]
        history = dict()
        train_loss_after_each_epoch = []
        test_loss_after_each_epoch = []
        for i in range(epochs):
            for j in range(0,N):
                target = y_train[0:1,j:j+1]
                prediction = np.matmul(self.w,X_train[j:j+1,:].T)
                #Gradient of loss function
                grad = (-1*np.matmul(target - prediction,X_train[j:j+1,:])) + lambda_*np.sign(self.w)
                #update step
                self.w = self.w - lr*grad
            train_loss = np.mean(np.square(y_train - np.matmul(self.w,X_train.T)))
            test_loss = np.mean(np.square(y_test - np.matmul(self.w,X_test.T)))
            train_loss_after_each_epoch.append(train_loss)
            test_loss_after_each_epoch.append(test_loss)
            if(flag):
                print('epoch: {} ----- train_loss: {}   test_loss: {}'.format(i+1,train_loss,test_loss))
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
    
    y_test = test['t']
    X_test = test.drop('t',axis = 1)
    #Transfroming test data
    X_test = (X_test - train_data_mean)/train_data_std
    #Adding ones for the bias term
    ones = np.ones(X_test.shape[0])
    X_test['x0'] = ones
    
    y_test = y_test.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.reshape([1,y_test.shape[0]])
    
    return X_train,y_train,X_test,y_test
    
# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-tr", "--train", help = "train data file path")
parser.add_argument("-te", "--test", help = "test data file path")




# Read arguments from command line
args = parser.parse_args()

train_file_path = args.train
test_file_path = args.test

train = pd.read_csv(train_file_path)
test = pd.read_csv(test_file_path)
X_train,y_train,X_test,y_test = preprocess(train,test)

#Ridge Regression
print("################ RIDGE REGRESSION(lambda = 0.00001) ##################")
model_RidgeSGD = LinearRegression()
model_RidgeSGD_history = model_RidgeSGD.RidgeSGD(X_train,y_train,X_test,y_test,2000,0.001,6,0.00001)

print("Mean Squared Error in Ridge :",model_RidgeSGD.MSE(X_test,y_test))
print("Mean Absolute Error in Ridge :",model_RidgeSGD.MAE(X_test,y_test))
print("Correlation between test predictions and test targets in Ridge :",model_RidgeSGD.corr(X_test,y_test))
print("Weights of RidgeSGD Model:",model_RidgeSGD.w)

#Lasso Regression
print("################ Lasso REGRESSION ##################")
print("For lambda = 0.00001,")
model_LassoSGD_1 = LinearRegression()
model_LassoSGD_history_1 = model_LassoSGD_1.LassoSGD(X_train,y_train,X_test,y_test,2000,0.001,6,0.00001)
print("For lambda = 0.001,")
model_LassoSGD_2 = LinearRegression()
model_LassoSGD_history_2 = model_LassoSGD_2.LassoSGD(X_train,y_train,X_test,y_test,2000,0.001,6,0.001)
print("For lambda = 0.1,")
model_LassoSGD_3 = LinearRegression()
model_LassoSGD_history_3 = model_LassoSGD_3.LassoSGD(X_train,y_train,X_test,y_test,2000,0.001,6,0.1)
print("For lambda = 1,")
model_LassoSGD_4 = LinearRegression()
model_LassoSGD_history_4 = model_LassoSGD_4.LassoSGD(X_train,y_train,X_test,y_test,2000,0.001,6,1)

metrics = pd.DataFrame(columns=['lambda','w0','w1','w2','w3','w4','w5','w6','MSE','MAE','corr'])
metrics['lambda'] = [0.00001,0.001,0.1,1]
metrics['w0'] = [model_LassoSGD_1.w[0][6],model_LassoSGD_2.w[0][6],model_LassoSGD_3.w[0][6],model_LassoSGD_4.w[0][6]]
metrics['w1'] = [model_LassoSGD_1.w[0][0],model_LassoSGD_2.w[0][0],model_LassoSGD_3.w[0][0],model_LassoSGD_4.w[0][0]]
metrics['w2'] = [model_LassoSGD_1.w[0][1],model_LassoSGD_2.w[0][1],model_LassoSGD_3.w[0][1],model_LassoSGD_4.w[0][1]]
metrics['w3'] = [model_LassoSGD_1.w[0][2],model_LassoSGD_2.w[0][2],model_LassoSGD_3.w[0][2],model_LassoSGD_4.w[0][2]]
metrics['w4'] = [model_LassoSGD_1.w[0][3],model_LassoSGD_2.w[0][3],model_LassoSGD_3.w[0][3],model_LassoSGD_4.w[0][3]]
metrics['w5'] = [model_LassoSGD_1.w[0][4],model_LassoSGD_2.w[0][4],model_LassoSGD_3.w[0][4],model_LassoSGD_4.w[0][4]]
metrics['w6'] = [model_LassoSGD_1.w[0][5],model_LassoSGD_2.w[0][5],model_LassoSGD_3.w[0][5],model_LassoSGD_4.w[0][5]]
metrics['MSE'] = [model_LassoSGD_1.MSE(X_test,y_test),model_LassoSGD_2.MSE(X_test,y_test),model_LassoSGD_3.MSE(X_test,y_test),model_LassoSGD_4.MSE(X_test,y_test)]
metrics['MAE'] = [model_LassoSGD_1.MAE(X_test,y_test),model_LassoSGD_2.MAE(X_test,y_test),model_LassoSGD_3.MAE(X_test,y_test),model_LassoSGD_4.MAE(X_test,y_test)]
metrics['corr'] = [model_LassoSGD_1.corr(X_test,y_test),model_LassoSGD_2.corr(X_test,y_test),model_LassoSGD_3.corr(X_test,y_test),model_LassoSGD_4.corr(X_test,y_test)]

print(metrics)