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


    def ImprovBGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,batch_size = 1,lambda_1 = 1,lambda_2 = 1,flag = 0):
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
                grad = (-1*np.matmul(target - prediction,X_train[j:min(j+batch_size,N),:]) + lambda_1*self.w + lambda_2*np.sign(self.w))/nf
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

print("################ Feature Selection ##################")
metrics = pd.DataFrame(columns = ['Weights','Test_MSE','Train_MAE','Corr'])
feat = ['x1','x2','x3','x6']
for i in feat:
    new_train = train
    new_test = test
    for j in feat:
        if(i != j):
            new_train = new_train.drop(j,axis = 1)
            new_test = new_test.drop(j,axis = 1)
    X_train,y_train,X_test,y_test = preprocess(new_train,new_test)
    model= LinearRegression()
    model_history = model.BGD(X_train,y_train,X_test,y_test,10000,0.1,X_train.shape[1] - 1,64)
    print('Weights when {}, x_4 and x_5 are selected(last element is the bias term):'.format(i),model.w)
    metrics.loc[len(metrics.index)] = [i+'x4x5', model.MSE(X_test,y_test), model.MAE(X_test,y_test), model.corr(X_test,y_test)]   
 
print(metrics)

print("################ Improvising Loss Function ##################")
X_train,y_train,X_test,y_test = preprocess(train,test)
model= LinearRegression()
model_history = model.ImprovBGD(X_train,y_train,X_test,y_test,10000,0.1,X_train.shape[1] - 1,64,0.00001,0.001)
print("Weights(last element is the bias term):",model.w)
print("Mean Squared Error(Test) :",model.MSE(X_test,y_test))
print("Mean Absolute Error(Test):",model.MAE(X_test,y_test))
print("Mean Squared Error (Train):",model.MSE(X_train,y_train))
print("Mean Absolute Error (Train):",model.MAE(X_train,y_train))