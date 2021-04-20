# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:31:05 2020

@author: natalia
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def calc_softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 1).reshape(-1,1)

def create_minibatches(images, target, batch_size):
    new_idxs = np.random.permutation(len(images))
    batch_divide = np.arange(batch_size, len(images), batch_size)
    batches = np.split(images[new_idxs, :, :], batch_divide)
    y_batches = np.split(target[new_idxs, :], batch_divide)
    return batches, y_batches

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy*100 

def cross_entropy(pred, target):
    return -np.log((pred * target).sum(axis = 1))

class ConvNeuralNetwork:
    
    def __init__(self, num_filters, filter_size, stride, n_neurons):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.n_neurons = n_neurons
        self.mini_batch_size = None
        self.epochs = None
        self.lr = None
        self.h_out = None
        self.w_out = None
        self.filters = None
        self.w = None
        
    def __init_params(self, X, epochs, batch_size, lr):
        np.random.seed(10)
        self.mini_batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.filters = np.random.normal(0.0, 0.01, (self.filter_size, self.filter_size, 
                                                    self.num_filters))
        self.h_out, self.w_out = self.__calculate_output_dims(X[0])
        self.w = np.random.normal(0.0, 0.01, ( self.h_out * self.w_out * self.num_filters, 
                                              self.n_neurons))

    def __forward(self, mini_batch):
        out_conv = []
        [out_conv.append(self.__calculate_convolution(img, self.filters, 
                                                      self.h_out, self.w_out)) for img in mini_batch]
        feature_maps = np.array(out_conv)
        flatten = feature_maps.reshape((feature_maps.shape[0], -1))
        flatten = (flatten - np.min(flatten)) / (np.max(flatten) - np.min(flatten))   
        z = flatten @ self.w
        y_pred = calc_softmax(z)
        return feature_maps, flatten, z, y_pred
    
    def __backward(self, mini_batch, batch_y, params):
        feature_maps, flatten, z, y_pred = params
        delta = y_pred - batch_y
        self.w -= self.lr * (flatten.T @ delta)
        delta2 = (delta @ self.w.T).reshape(feature_maps.shape)
        filters_update = []
        [filters_update.append(self.__calculate_convolution(mini_batch[j], delta2[j], 
                                                            self.filter_size, self.filter_size)) \
         for j in range(len(mini_batch))]
        df = np.mean(np.array(filters_update), axis = 0)
        self.filters -= self.lr * df
        
    def fit(self, X, y_, epochs, batch_size, lr):
        self.__init_params(X, epochs, batch_size, lr)
        loss_list = []

        for epoch in range(self.epochs):
            input_list = []
            mini_batches, target_mini_batches = create_minibatches(X, y_, self.mini_batch_size)
            for mini_batch, batch_y in zip(mini_batches, target_mini_batches):
                params = self.__forward(mini_batch)
                self.__backward(mini_batch, batch_y, params)    
            if epoch % 10 == 0:
                loss = cross_entropy(self.__forward(X)[-1], y_).mean()
                loss_list.append(loss)
                print(f'epoch {epoch} loss {loss:.4f}')
                
        return loss_list
             
    def predict(self, X_new):
        return self.__forward(X_new)[-1].argmax(axis = 1)
    
    def __calculate_output_dims(self, img):
        '''Calculates output dimensions after performing convolution
        Parameters
        ----------
        img : array of floats, size: 8 x 8 according to load_digits dataset
        Returns
        -------
        h_out, w_out : int, heigth and width of feature map
        '''
        h_in, w_in = img.shape
        h_f, w_f = self.filters[:,:,0].shape
        h_out = (h_in - h_f) / self.stride + 1
        w_out = (w_in - w_f) / self.stride + 1
        return int(h_out), int(w_out)

    def __calculate_convolution(self, img, filter_n, h_out, w_out):
        '''Calculates convolution over single image
        Parameters
        ----------
        img : array of floats, size: 8 x 8 according to load_digits dataset
        filter_n : array of floats, size: filter_size x filter_size x num_filters
        h_out : int, heigth of feature map
        w_out : int. width of feature map
        Returns
        -------
        output : list with size of mini_batch, each element is a numpy array of floats
                 size: h_out x w_out x num_filters
            DESCRIPTION.

        '''
        h, w = filter_n[:,:,0].shape
        output = np.zeros((h_out, w_out, filter_n.shape[-1]))
        for i in range(h_out):
            for j in range(w_out):
                area = img[i : h, j : w]
                output[i,j] = np.sum(area * filter_n[:,:,i])
                w += 1
            w = filter_n[:,:,0].shape[1]
            h += 1
        return output

def plot_loss(loss1, loss2):
    '''Plots loss figure with train and validation losses during fit() method
    Parameters
    ----------
    loss1 : list, train set loss values
    loss2 : list, validation set loss values
    Returns
    -------
    None.
    '''
    plt.title('Loss')
    plt.plot(loss1, color = 'r', label = 'train')
    plt.plot(loss2, color = 'b', label = 'validation')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":

    digits = load_digits()
    X = digits.images
    y = pd.get_dummies(pd.Series(digits.target)).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 15)
    
    model = ConvNeuralNetwork(num_filters = 6, filter_size = 3, stride = 1, n_neurons = 10)        
    loss_ = model.fit(X_train, y_train, epochs = 100, batch_size = 32, lr = 0.01)
    
    prediction = model.predict(X_train)
    labels = np.argmax(y_train, axis = 1)
    print("Train set accuracy: %.2f" % accuracy(labels, prediction))
    
    pred_test = model.predict(X_test)
    labels_test = np.argmax(y_test, axis = 1)
    print("Test set accuracy: %.2f" % accuracy(labels_test, pred_test))
    
    plt.title('My Loss')
    plt.plot(loss_)
    plt.show()
    
    ###############    Keras implementation   #################################
    # from tensorflow.keras.layers import Conv2D, Dense, Flatten
    # from tensorflow.keras.models import Sequential  
    # from tensorflow.keras.utils import to_categorical  
    
    # train_img = np.expand_dims(X_train, axis = 3)
    # test_img = np.expand_dims(X_test, axis = 3)
    
    # model = Sequential()
    # model.add(Conv2D(6, (3,3), input_shape = (8, 8, 1), use_bias = False))
    # model.add(Flatten())
    # model.add(Dense(10, activation = 'softmax'))
    # model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    # model.summary()
    
    # history = model.fit(train_img, y_train, epochs=10, batch_size = 12, verbose=0, validation_split=0.2)
    # loss_train, accuracy_train = model.evaluate(train_img, y_train, batch_size = 12, verbose=0)
    # loss_test, accuracy_test = model.evaluate(test_img, y_test, batch_size = 12, verbose = 0)
    # print('Accuracy on train data: %.2f' % (accuracy_train*100))
    # print('Accuracy on test data: %.2f' % (accuracy_test*100))
    # loss, val_loss = history.history["loss"], history.history["val_loss"]
    # plot_loss(loss, val_loss)    
                         
    # predictions = model.predict_classes(test_img)
    # # y_keras = to_categorical(digits.target)
    
    
    
    
    
    
    
    
    












    