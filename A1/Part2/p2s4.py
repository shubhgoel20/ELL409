import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-o", "--out", help = "output file path")
parser.add_argument("-tx", "--trainX", help = "train images file path")
parser.add_argument("-ty", "--trainY", help = "image labels file path")
parser.add_argument("-tex", "--testX", help = "test images file path")



# Read arguments from command line
args = parser.parse_args()


class LogisticRegression:
    def __init__(self):
            self.w = None
            self.b = None
    

    def RidgeSGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,lambda_ = 1,flag = 0):
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

                dw =  np.dot(X_train[j:j+1].T, h_train - y_train[j:j+1]) + lambda_*self.w
                db =  np.sum(h_train-y_train[j:j+1]) + lambda_*self.b
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

X_train, y_train, X_test, y_test  = train_test_split(X_subset, y_subset, test_size = 0.2, random_state=1234)

print("#### Running the Model that gives the most Optimal Weights(takes approx 210 secs) ####")
#Ridge Regression
model_RidgeSGD = LogisticRegression()
model_RidgeSGD_history = model_RidgeSGD.RidgeSGD(X_train,y_train,X_test,y_test,900,0.001,X_train.shape[1],0.0001)

print("log-loss (Test) in Ridge :",model_RidgeSGD.get_loss(X_test,y_test))
print("log-loss (Train) in Ridge :",model_RidgeSGD.get_loss(X_train,y_train))
print("Weights(w_1,w_2,.... w_784) of RidgeSGD Model:",model_RidgeSGD.w)
print("Bias of RidgeSGD Model:",model_RidgeSGD.b)

plt.scatter(range(1,len(model_RidgeSGD.w)+1),model_RidgeSGD.w)
plt.ylabel('Value')
plt.xlabel('Weights(w_1,w_2, ... ,w_784)')
plt.title('Ridge Weights')
plt.show(block = False)
plt.pause(3)

test_images_file = args.testX
test_images = np.load(test_images_file)
test_images = test_images/255

predictions = model_RidgeSGD.predict(test_images)

predictions = np.array([2 if i == 1 else 9 for i in predictions])

predictions = predictions.reshape([predictions.shape[0],1])

out_file = args.out

with open(out_file, 'w') as my_file:
        for i in predictions:
            np.savetxt(my_file,i,fmt = '%d')



