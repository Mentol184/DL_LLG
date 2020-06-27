import pandas as pd
import numpy as np

import time
import datetime

from keras.models import Sequential
from keras.layers import Dense
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
    data = pd.read_csv("data_set.csv",sep=',')
    
    # Get real values
    #B_ext = 0.1
    #real_values = filter_data(data, B_ext, 'my')
    
     
    # Split into targets and inputs
    x = data.iloc[:,[0,4,5]]
    y = data.iloc[:,1:4]
    
    #Scale data between (0,1)
    x_training_scaler = MinMaxScaler(feature_range=(0, 1))
    y_training_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scale_x = x_training_scaler.fit_transform(x)
    scale_y = y_training_scaler.fit_transform(y)
    
    # Split data into train and validation data
    training_x, x_test, training_y, y_test = train_test_split(scale_x, scale_y, test_size = 0.1)
#    
#    testing_data = pd.read_csv("data_set_testing.csv", sep=',')
#    
#    testing_x = testing_data.iloc[:,[0,4,5]]
#    testing_y = testing_data.iloc[:,1:4]
#    
#    x_test = x_training_scaler.fit_transform(testing_x)
#    y_test = y_training_scaler.fit_transform(testing_y)
#    
    '''Neural Network'''
    # Perform the model
    model = Sequential()
    
    #Input Layer
    model.add(Dense(246, input_dim=3, activation='elu'))
    
    #Hidden layers
    model.add(Dense(246, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(128, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(64, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(32, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(16, activation='elu'))
    model.add(Dense(8, activation='elu'))
    model.add(Dense(8, activation='elu'))       
            
    #output layer
    model.add(Dense(3, activation='tanh'))
    
    #Optimizer
    nadam = Nadam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999)

    #Compile model    
    model.compile(loss=custom_loss, optimizer=nadam, metrics=['mse','accuracy'])
    
    
    '''Training Strategy'''
    
    # Train model
    results = train_model(model, training_x, training_y, x_test, y_test)
    
    # Draw the loss history
    model_loss(results)
    
    
    '''Testing Analysis'''
    
    # Testing
    testing(model, y_training_scaler, x_training_scaler)
    
    
    '''Save Model'''
    #save model and scale
    save_model(model, x_training_scaler, y_training_scaler)
    
    return data



# This function fit the model
def train_model(model, training_x, training_y, x_test, y_test):
    
    # Count elapsed time
    start_time = time.time()
    
     # simple early stopping
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=20)
    
    # Fit model (train model)
    results = model.fit(training_x, training_y, validation_data=(x_test,y_test), epochs=4000, batch_size=600, callbacks=[es]) 
    
    # Print elapsed time
    elapsed_time = time.time() - start_time  
    elapsed_time = str(datetime.timedelta(seconds=elapsed_time))
    print('Elapsed time: '+ elapsed_time)
    
    return results


# This function is a custom loss for this physics problem
def custom_loss(y_true, y_predict):
    
#    mx = y_true[:, 0]
#    my = y_true[:, 1]
#    mz = y_true[:, 2]
#    
#    mx_p = y_predict[:, 0]
#    my_p = y_predict[:, 1]
#    mz_p = y_predict[:, 2]
#    
        
#    l2 = K.sum(K.abs(mx-mx_p)+K.abs(my-my_p)+K.abs(mz-mz_p))
    
    l2 = K.sum(K.abs(y_true-y_predict))

#    l1 = K.sum(K.sqrt(K.pow(mx_p,2)+K.pow(my_p,2)+K.pow(mz_p,2)) -1)
    l1 = K.sum(K.sqrt(K.pow(y_predict,2) -1))
    
    l = l1+l2
    
    return l

# This functions print the loss history
def model_loss(results):
    
    # Summarize train history for loss
    plt.figure()
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    # Summarize train history for accuracy
    plt.figure()
    plt.plot(results.history['accuracy'])
    plt.plot(results.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
         
    return    

# This function plot the predicted data.
def testing(model, y_training_scaler, x_training_scaler):
    
    magnetic_field = 0.1
    alpha = 0.01

    time = np.arange(0,1e-8,1.001e-11)
    
    B = np.array([magnetic_field for i in range(1000)])
    dumping_coefficient = np.array([alpha for i in range(1000)])
    
    Xnew = np.column_stack((time, B, dumping_coefficient))
    
    ynew = model.predict(x_training_scaler.transform(Xnew))
    
    ynew = y_training_scaler.inverse_transform(ynew)
    
    # show the inputs and predicted outputs
    plt.figure()
    plt.plot(Xnew[:, 0], ynew[:, 0], label = 'mx')
    plt.plot(Xnew[:, 0], ynew[:, 1], label = 'my')
    plt.plot(Xnew[:, 0], ynew[:, 2], label = 'mz')
    plt.xlabel('time (s)')
    plt.legend()
    
# This function save the model for posterior implementation    
def save_model(model, x_training_scaler, y_training_scaler):
    
    model.save("model.h5")
    dump(x_training_scaler, open('x_scaler.pkl', 'wb'))
    dump(y_training_scaler, open('y_scaler.pkl', 'wb'))

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

