from json import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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



def one_hot(y,num_classes):
    y_one_hot = np.zeros((y.shape[0],num_classes))
    for i in range(y.shape[0]):
        y_one_hot[i][y[i]] = 1
    return y_one_hot

class LogisticRegression:
    def __init__(self):
            self.w = None
            # self.b = None
    
    
    def BGD(self,X_train,y_train,X_test,y_test,epochs,lr,num_features,num_class,batch_size = 1,flag = 0):
        #X_train = N X num_features
        #y_train = N X num_classes
        np.random.seed(2)
        self.w = np.random.randn(num_class,num_features)
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


                a_batch = np.dot(self.w,X_train[j:j+nf,:].T) #num_class X nf
                h_batch = self.softmax(a_batch) #num_class X nf

                grad = (np.dot(h_batch - y_train[j:j+nf,:].T,X_train[j:j+nf,:]))/nf

                self.w -= (lr*grad)
            a_test = np.dot(self.w,X_test.T)
            h_test = self.softmax(a_test)
            a_train= np.dot(self.w,X_train.T)
            h_train = self.softmax(a_train)
            train_loss = self.logloss(h_train , y_train)
            test_loss = self.logloss(h_test , y_test)
            train_loss_after_each_epoch.append(train_loss)
            test_loss_after_each_epoch.append(test_loss)
            if(flag):
                print('epoch: {} ----- train_loss: {}   test_loss: {}'.format(i+1,train_loss,test_loss))
        print('Done !')
        predictions = np.argmax(h_test,axis = 0)
        targets = np.argmax(y_test,axis = 1)
        count = 0
        for i in range(predictions.shape[0]):
            if(predictions[i] == targets[i]):
                count = count+1

        acc = count/len(y_test)
        # acc = np.sum(predictions == y_test) / len(y_test)
        print("Accuracy : " , np.array(acc))
        history['weights'] = self.w
        history['train_loss'] = train_loss_after_each_epoch
        history['test_loss'] = test_loss_after_each_epoch
        return history


    def softmax(self,x):
        #x is kXbatch_size
        return np.exp(x)/np.sum(np.exp(x),axis = 0)

    
    def logloss(self,y,t):
        #y num_class X N
        #t N X num_class
        sum = 0
        for i in range(t.shape[0]):
            sum+= -1*np.dot(t[i:i+1,:],np.log(y[:,i:i+1] + 1e-30))
        
        return sum/t.shape[0]
        
    def get_loss(self,X_test,y_test):
        a_pred = np.dot(self.w,X_test.T)
        y_pred = self.softmax(a_pred)
        return self.logloss(y_pred, y_test)
    
    def predict(self,X):
        a_pred = np.dot(self.w,X.T)
        y_pred = self.softmax(a_pred)
        return np.argmax(y_pred,axis = 0)

    def accuracy(self,X_test,y_test):
        predictions = self.predict(X_test)
        targets = np.argmax(y_test,axis = 1)
        count = 0
        for i in range(predictions.shape[0]):
            if(predictions[i] == targets[i]):
                count = count+1

        acc = count/len(y_test)
        
        return acc

    def confusion_matrix(self,X_test,y_test):
        predictions = self.predict(X_test)
        targets = np.argmax(y_test,axis = 1)
        matrix = np.zeros((y_test.shape[1],y_test.shape[1]))
        for i in range(predictions.shape[0]):
            matrix[targets[i]][predictions[i]] +=1
        
        df = pd.DataFrame(matrix, range(y_test.shape[1]),range(y_test.shape[1]))
        plt.figure(figsize = (8,8))
        sns.set(font_scale = 0.8)
        sns.heatmap(df, annot=True,cmap = 'summer', linewidths=1, linecolor='black', square=True)
        plt.show()


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

X = np.load(train_images_path)
y = np.load(train_labels_path)

X = X/255
y = one_hot(y,10)

ones = np.ones((X.shape[0],1))
X = np.append(X,ones,axis = 1)


X_train, y_train, X_test, y_test  = train_test_split(X, y, test_size = 0.2, random_state=1234)


print("#### RUNNING MULTI CLASS MULTIVARIATE LOGISTIC REGRESSION (takes approx 100 secs)")

model = LogisticRegression()
model_history = model.BGD(X_train,y_train,X_test,y_test,100,0.1,X_train.shape[1],10,64,1)
print("log-loss (Train):",model.get_loss(X_train,y_train))
print("log-loss (Test):",model.get_loss(X_test,y_test))

test_images_file = args.testX
test_images = np.load(test_images_file)
test_images = test_images/255
ones = np.ones((test_images.shape[0],1))
test_images = np.append(test_images,ones,axis = 1)

predictions = model.predict(test_images)

predictions = predictions.reshape([predictions.shape[0],1])

out_file = args.out

with open(out_file, 'w') as my_file:
        for i in predictions:
            np.savetxt(my_file,i,fmt = '%d')


model.confusion_matrix(X_test,y_test)

