import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-tx", "--trainX", help = "train images file path")
parser.add_argument("-ty", "--trainY", help = "image labels file path")


# Read arguments from command line
args = parser.parse_args()


class LogisticRegression:
    def __init__(self):
            self.w = None
            self.b = None
    
    def SGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,flag = 0):
        np.random.seed(2)
        self.w = np.zeros(num_features) #randomly initialize weights. Row vector of size (num_features)
        self.b = 0
        N = X_train.shape[0]
        M = X_test.shape[0]
        history = dict()
        train_loss_after_each_epoch = []
        test_loss_after_each_epoch = []

        for i in range(epochs):
            for j in range(0,N):
                prediction = np.dot(X_train[j:j+1,:],self.w) + self.b #prediction = NX1
                h_train = self.sigmoid(prediction)

                dw =  np.dot(X_train[j:j+1].T, h_train - y_train[j:j+1])
                db =  np.sum(h_train-y_train[j:j+1])
                self.w -= (lr * dw)
                self.b -= (lr * db)

            
            predtest = np.dot(X_test,self.w)+self.b
            h_test = self.sigmoid(predtest)
            predtrain = np.dot(X_train,self.w)+self.b
            h_train = self.sigmoid(predtrain)
            train_loss = np.mean(self.logloss(y_train , h_train))
            test_loss = np.mean(self.logloss(y_test , h_test))
            train_loss_after_each_epoch.append(train_loss)
            test_loss_after_each_epoch.append(test_loss)
            if(flag):
                print('epoch: {} ----- train_loss: {}   test_loss: {}'.format(i+1,train_loss,test_loss))
        print('Done !')
        history['weights'] = self.w
        history['train_loss'] = train_loss_after_each_epoch
        history['test_loss'] = test_loss_after_each_epoch

        outp = [1 if i >= 0 else 0 for i in predtest]
        acc = np.sum(outp == y_test) / len(y_test)
        print("Accuracy : " , np.array(acc))

        return history

    def BGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,batch_size = 1,flag = 0):
        np.random.seed(2)
        self.w = np.random.randn(num_features) #randomly initialize weights. Row vector of size (num_features+1)
        self.b = 0
        N = X_train.shape[0]
        M = X_test.shape[0]

        history = dict()
        train_loss_after_each_epoch = []
        test_loss_after_each_epoch = []
        for i in range(epochs):
            for j in range(0,N,batch_size):
                if(j + batch_size > N):
                    nf = N-j
                else:
                    nf = batch_size

                prediction = np.dot(X_train[j:j+nf,:],self.w) + self.b #prediction = NX1
                h_train = self.sigmoid(prediction)

                dw =  (np.dot(X_train[j:j+nf].T, h_train - y_train[j:j+nf]))/nf
                db =  (np.sum(h_train-y_train[j:j+nf]))/nf
                self.w -= (lr * dw)
                self.b -= (lr * db)
            predtest = np.dot(X_test,self.w)+self.b
            h_test = self.sigmoid(predtest)
            predtrain = np.dot(X_train,self.w)+self.b
            h_train = self.sigmoid(predtrain)
            train_loss = np.mean(self.logloss(y_train , h_train))
            test_loss = np.mean(self.logloss(y_test , h_test))
            train_loss_after_each_epoch.append(train_loss)
            test_loss_after_each_epoch.append(test_loss)
            if(flag):
                print('epoch: {} ----- train_loss: {}   test_loss: {}'.format(i+1,train_loss,test_loss))
        print('Done !')
        outp = [1 if i >= 0 else 0 for i in predtest]
        acc = np.sum(outp == y_test) / len(y_test)
        print("Accuracy : " , np.array(acc))
        history['weights'] = self.w
        history['train_loss'] = train_loss_after_each_epoch
        history['test_loss'] = test_loss_after_each_epoch
        return history

    
    def sigmoid(self,x):
        return (1 / (1 + np.exp(-x)))
    
    def logloss(self,x,xpred):
        return -1 * ( x * np.log( xpred + 1e-30 ) + (1-x) * np.log(( 1 - xpred + 1e-30 )) )
        
    def get_loss(self,X_test,y_test):
        predtest = np.dot(X_test,self.w)+self.b
        h_test = self.sigmoid(predtest)
        return np.mean(self.logloss(y_test , h_test))
    
    def predict(self,X):
        pred = np.dot(X,self.w)+self.b
        outp = [1 if i >= 0 else 0 for i in pred]
        return outp
    def accuracy(self,X_test,y_test):
        outp = self.predict(X_test)
        return np.sum(outp == y_test) / len(y_test)

def train_test_split(X_subset, y_subset, test_size = 0.2, random_state=1234):
    indices = np.array(range(0,X_subset.shape[0]))
    np.random.seed(random_state)
    np.random.shuffle(indices)
    N = indices[0:int((1-test_size)*X_subset.shape[0])+1]
    M = indices[int((1-test_size)*X_subset.shape[0])+1: X_subset.shape[0]]
    train_x = X_subset[N]
    test_x = X_subset[M]
    train_y = y_subset[N]
    test_y = y_subset[M]

    return train_x, train_y, test_x, test_y 

train_images_path = args.trainX
train_labels_path = args.trainY

X_subset = np.load(train_images_path)
y_subset = np.load(train_labels_path)

X_subset = X_subset/255
y_subset = np.array([1 if i == 2 else 0 for i in y_subset])


train_x, train_y, test_x, test_y  = train_test_split(X_subset, y_subset, test_size = 0.2, random_state=1234)

# (a) STOCHASTIC GRADIENT DESCENT

print("########### RUNNING STOCHASTIC GRADIENT DESCENT ###########")

model_SGD = LogisticRegression()
model_SGD_history = model_SGD.SGD(train_x,train_y,test_x,test_y,300,0.001,train_x.shape[1])

print("log-loss (Train):",model_SGD.get_loss(train_x,train_y))
print("log-loss (Test):",model_SGD.get_loss(test_x,test_y))
print("Weights of SGD Model:",model_SGD.w)
print("Bias:", model_SGD.b)

plt.scatter(range(1,len(model_SGD.w)+1),model_SGD.w)
plt.ylabel('Value')
plt.xlabel('Weights(w_1,w_2, ... ,w_784)')
plt.title('Stochastic Gradient Weights')
plt.show(block = False)
plt.pause(3)

print("########### RUNNING BATCH GRADIENT DESCENT ###########")

# (b) BATCH GRADIENT DESCENT

model_BGD_1 = LogisticRegression()
model_BGD_16 = LogisticRegression()
model_BGD_64 = LogisticRegression()
model_BGD_256 = LogisticRegression()
model_BGD_512 = LogisticRegression()
model_BGD_1024 = LogisticRegression()

hist1 = model_BGD_1.BGD(train_x,train_y,test_x,test_y,300,0.001,train_x.shape[1],1)
hist16 = model_BGD_16.BGD(train_x,train_y,test_x,test_y,300,0.001,train_x.shape[1],16)
hist64 = model_BGD_64.BGD(train_x,train_y,test_x,test_y,300,0.001,train_x.shape[1],64)
hist256 = model_BGD_256.BGD(train_x,train_y,test_x,test_y,300,0.001,train_x.shape[1],256)
hist512 = model_BGD_512.BGD(train_x,train_y,test_x,test_y,300,0.001,train_x.shape[1],512)
hist1024 = model_BGD_1024.BGD(train_x,train_y,test_x,test_y,300,0.001,train_x.shape[1],1024)

plt.plot(np.array(hist1['train_loss']))
plt.plot(np.array(hist16['train_loss']))
plt.plot(np.array(hist64['train_loss']))
plt.plot(np.array(hist256['train_loss']))
plt.plot(np.array(hist512['train_loss']))
plt.plot(np.array(hist1024['train_loss']))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Loss Curves for different Batch Sizes')
plt.legend(['1','16', '64','256','512', '1024'], loc='upper right')
plt.show(block = False)
plt.pause(3)

metrics = pd.DataFrame(columns=['Batch_size','Train_Loss','Test_Loss','Test Accuracy'])
metrics['Batch_size'] = [1,16,64,256,512,1024]
metrics['Test_Loss'] = [model_BGD_1.get_loss(test_x,test_y),model_BGD_16.get_loss(test_x,test_y),model_BGD_64.get_loss(test_x,test_y),model_BGD_256.get_loss(test_x,test_y),model_BGD_512.get_loss(test_x,test_y),model_BGD_1024.get_loss(test_x,test_y)]
metrics['Train_Loss'] = [model_BGD_1.get_loss(train_x,train_y),model_BGD_16.get_loss(train_x,train_y),model_BGD_64.get_loss(train_x,train_y),model_BGD_256.get_loss(train_x,train_y),model_BGD_512.get_loss(train_x,train_y),model_BGD_1024.get_loss(train_x,train_y)]
metrics['Test Accuracy'] = [model_BGD_1.accuracy(test_x,test_y),model_BGD_16.accuracy(test_x,test_y),model_BGD_64.accuracy(test_x,test_y),model_BGD_256.accuracy(test_x,test_y),model_BGD_512.accuracy(test_x,test_y),model_BGD_1024.accuracy(test_x,test_y)]

print(metrics)