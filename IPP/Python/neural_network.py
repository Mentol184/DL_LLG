import pandas as pd
import numpy as np

import time
import datetime

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Nadam
import keras.backend as K
from keras.callbacks import EarlyStopping


#from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pickle import dump

import matplotlib.pyplot as plt

def main():
    

    ''' Prepare Data'''
    data = pd.read_csv("../Data/dataset.csv",sep=',')
    # data = data[ (data['Jz (A/m2)'] <= 80e9) & (data['Jz (A/m2)'] >= 70e9)]

    modelo = "modelo_4"
    # neuronas = str(neuron)
    # modelo = model + neuronas
    
     
    # Split into targets and inputs
    x = data.iloc[:,[0,6]]
    y = data.iloc[:,[1,2,3]]

    
    #Scale data between (0,1)
    x_training_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_training_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    scale_x = x_training_scaler.fit_transform(x)
    scale_y = y_training_scaler.fit_transform(y)
    
    
    
    # Split data into train and validation data
    # training_x, x_test, training_y, y_test = train_test_split(scale_x, scale_y, test_size = 0.00001)
    training_x = scale_x
    training_y = scale_y
    
    '''Neural Network'''
    # Perform the model
    model = Sequential()
    
    
    model.add(Dense(246, input_dim=2, activation='tanh'))
        
    
    # Hidden layers
    model.add(Dense(246, activation='tanh'))
    model.add(Dense(246, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    # model.add(Dense(64, activation='tanh'))
    # model.add(Dense(64, activation='tanh'))
    # model.add(Dense(32, activation='tanh'))
    # model.add(Dense(32, activation='tanh'))

       
    #output layer
    model.add(Dense(3, activation='tanh'))
    
    #Optimizer
    nadam = Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999)
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(loss=custom_loss, optimizer="sgd", metrics=['mse','accuracy'])
    
    
    '''Training Strategy'''
    
    # Train model
    # results = train_model(model, training_x, training_y, x_test, y_test)
    results = train_model(model, training_x, training_y)
    
    # Draw the loss history
    model_loss(results, modelo)
    
    
    '''Testing Analysis'''
    
    # Testing
    testing(model, y_training_scaler, x_training_scaler, modelo, data)
    
    
    '''Save Model'''
    #save model and scale
    save_model(model, x_training_scaler, y_training_scaler, modelo)
        
    
        
    return data



# This function fit the model
def train_model(model, training_x, training_y):
    
    # Count elapsed time
    start_time = time.time()
    
     # simple early stopping
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=500)
    
    # Fit model (train model)
    # results = model.fit(training_x, training_y, validation_data=(x_test,y_test), epochs=1000, batch_size=50000) 
    results = model.fit(training_x, training_y, epochs=10000, batch_size=2000, callbacks=[es], shuffle=True, verbose=1) 

    # Print elapsed time
    elapsed_time = time.time() - start_time  
    elapsed_time = str(datetime.timedelta(seconds=elapsed_time))
    print('Elapsed time: '+ elapsed_time)
    
    return results


# This function is a custom loss for this physics problem
def custom_loss(y_true, y_predict):
    
    
    l2 = K.mean(K.pow(K.abs(y_true-y_predict),2))
    l1 = K.mean(K.pow((K.sqrt(K.pow(y_predict,2) -1)),2))
    
    l = l1+l2
    
    return l

# This functions print the loss history
def model_loss(results, modelo):
    
    # Summarize train history for loss
    plt.figure()
    plt.plot(results.history['loss'])
    # plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('loss'+modelo+'.png')
    
    # Summarize train history for accuracy
    plt.figure()
    plt.plot(results.history['accuracy'])
    # plt.plot(results.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    plt.savefig('accuracy'+modelo+'.png')
         
    return    

# This function plot the predicted data.
def testing(model, y_training_scaler, x_training_scaler,modelo, data):
    
    current = 0.76e+09

    time = np.arange(0,10e-8,1e-11)
    
    points = time.size
    
    J = np.array([current for i in range(points)])
    
    Xnew = np.column_stack((time, J))
    
    ynew = model.predict(x_training_scaler.transform(Xnew))
    
    ynew = y_training_scaler.inverse_transform(ynew)
    
    # show the inputs and predicted outputs
    plt.figure()
    plt.plot(Xnew[:, 0], ynew[:, 0], label = 'mx')
    plt.plot(Xnew[:, 0], ynew[:, 1], label = 'my')
    plt.plot(Xnew[:, 0], ynew[:, 2], label = 'mz')
    plt.xlabel('time (s)')
    plt.title('J ='+str(current)+'A/m^2')
    plt.legend()
    plt.savefig('testing_'+modelo+'.png')
    
    
# This function save the model for posterior implementation    
def save_model(model, x_training_scaler, y_training_scaler, modelo):
    
    model.save("model_"+modelo+".h5")
    dump(x_training_scaler, open('x_scaler_'+modelo+'.pkl', 'wb'))
    dump(y_training_scaler, open('y_scaler_'+modelo+'.pkl', 'wb'))

# This function filter the data depending on the magnetic field
def filter_data(data, B_ext, target=''):
    
    data_filtered  = data[(data.B_ext_x == B_ext)]
    
    if target == "mx":
        real_values = data_filtered.iloc[:,[0,1]]
    elif target == "my":
        real_values = data_filtered.iloc[:,[0,2]]
    elif target == "mz":
        real_values = data_filtered.iloc[:,[0,3]]
    else:
        real_values = data_filtered
        print('Unknown target, try with mx, my or mz')
    
    return real_values


# Main define
if __name__ == '__main__':
    
    a = main()

