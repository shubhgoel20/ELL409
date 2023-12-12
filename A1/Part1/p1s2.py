import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

class LinearRegression:
    def __init__(self):
            self.w = None

    def SGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,flag = 0):
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
                grad = -1*np.matmul(target - prediction,X_train[j:j+1,:])
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

# (a) STOCHASTIC GRADIENT DESCENT
print("################ RUNNING STOCHASTIC GRADIENT DESCENT ##################")
model_SGD = LinearRegression()
model_SGD_history = model_SGD.SGD(X_train,y_train,X_test,y_test,3000,0.001,6)

print("Mean Squared Error :",model_SGD.MSE(X_test,y_test))
print("Mean Absolute Error:",model_SGD.MAE(X_test,y_test))
print("Correlation between test predictions and test targets:",model_SGD.corr(X_test,y_test))
print("Weights of SGD Model:",model_SGD.w)

# (b) BATCH GRADIENT DESCENT

print("################ RUNNING BATCH GRADIENT DESCENT ##################")

model_BGD_1 = LinearRegression()
model_BGD_16 = LinearRegression()
model_BGD_64 = LinearRegression()
model_BGD_256 = LinearRegression()
model_BGD_512 = LinearRegression()
model_BGD_1200 = LinearRegression()

hist1 = model_BGD_1.BGD(X_train,y_train,X_test,y_test,3000,0.001,6,1)
hist16 = model_BGD_16.BGD(X_train,y_train,X_test,y_test,3000,0.001,6,16)
hist64 = model_BGD_64.BGD(X_train,y_train,X_test,y_test,3000,0.001,6,64)
hist256 = model_BGD_256.BGD(X_train,y_train,X_test,y_test,3000,0.001,6,256)
hist512 = model_BGD_512.BGD(X_train,y_train,X_test,y_test,3000,0.001,6,512)
hist1200 = model_BGD_1200.BGD(X_train,y_train,X_test,y_test,3000,0.001,6,1200)

plt.plot(np.array(hist1['train_loss']))
plt.plot(np.array(hist16['train_loss']))
plt.plot(np.array(hist64['train_loss']))
plt.plot(np.array(hist256['train_loss']))
plt.plot(np.array(hist512['train_loss']))
plt.plot(np.array(hist1200['train_loss']))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['1','16', '64','256','512', '1200'], loc='upper right')
plt.title('Loss Curves for different Batch Sizes')


metrics = pd.DataFrame(columns=['Batch_size','w7','w1','w2','w3','w4','w5','w6','MSE','MAE','corr'])
metrics['Batch_size'] = [1,16,64,256,512,1200]
metrics['w7'] = [model_BGD_1.w[0][6],model_BGD_16.w[0][6],model_BGD_64.w[0][6],model_BGD_256.w[0][6],model_BGD_512.w[0][6],model_BGD_1200.w[0][6]]
metrics['w1'] = [model_BGD_1.w[0][0],model_BGD_16.w[0][0],model_BGD_64.w[0][0],model_BGD_256.w[0][0],model_BGD_512.w[0][0],model_BGD_1200.w[0][0]]
metrics['w2'] = [model_BGD_1.w[0][1],model_BGD_16.w[0][1],model_BGD_64.w[0][1],model_BGD_256.w[0][1],model_BGD_512.w[0][1],model_BGD_1200.w[0][1]]
metrics['w3'] = [model_BGD_1.w[0][2],model_BGD_16.w[0][2],model_BGD_64.w[0][2],model_BGD_256.w[0][2],model_BGD_512.w[0][2],model_BGD_1200.w[0][2]]
metrics['w4'] = [model_BGD_1.w[0][3],model_BGD_16.w[0][3],model_BGD_64.w[0][3],model_BGD_256.w[0][3],model_BGD_512.w[0][3],model_BGD_1200.w[0][3]]
metrics['w5'] = [model_BGD_1.w[0][4],model_BGD_16.w[0][4],model_BGD_64.w[0][4],model_BGD_256.w[0][4],model_BGD_512.w[0][4],model_BGD_1200.w[0][4]]
metrics['w6'] = [model_BGD_1.w[0][5],model_BGD_16.w[0][5],model_BGD_64.w[0][5],model_BGD_256.w[0][5],model_BGD_512.w[0][5],model_BGD_1200.w[0][5]]
metrics['MSE'] = [model_BGD_1.MSE(X_test,y_test),model_BGD_16.MSE(X_test,y_test),model_BGD_64.MSE(X_test,y_test),model_BGD_256.MSE(X_test,y_test),model_BGD_512.MSE(X_test,y_test),model_BGD_1200.MSE(X_test,y_test)]
metrics['MAE'] = [model_BGD_1.MAE(X_test,y_test),model_BGD_16.MAE(X_test,y_test),model_BGD_64.MAE(X_test,y_test),model_BGD_256.MAE(X_test,y_test),model_BGD_512.MAE(X_test,y_test),model_BGD_1200.MAE(X_test,y_test)]
metrics['corr'] = [model_BGD_1.corr(X_test,y_test),model_BGD_16.corr(X_test,y_test),model_BGD_64.corr(X_test,y_test),model_BGD_256.corr(X_test,y_test),model_BGD_512.corr(X_test,y_test),model_BGD_1200.corr(X_test,y_test)]


print(metrics)
plt.show()