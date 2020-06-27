import pandas as pd
import numpy as np
from scipy import stats
import os
from joblib import dump, load
from keras.models import load_model

import keras.losses
import keras.backend as K

from pickle import load

import matplotlib.pyplot as plt

from data_preparation import get_data, filter_data, select_field_values, split_data


def fft_testing(final_magnetic_field, initial_magnetic_field, alpha, model, y_trainig_scaler, x_training_scaler):
    
    item = int((initial_magnetic_field-final_magnetic_field)/(0.01))
        
    larmor_frequencies = [None]*(item)
    field_values = [None]*(item)
    frames = [None]*item
    
    B_ext = np.arange(final_magnetic_field, initial_magnetic_field, 0.01)
    
    i = 0
    
    for magnetic_field in B_ext:
                
        inputs, predictions = make_predictions(magnetic_field, alpha, model, y_trainig_scaler, x_training_scaler)

        theta = np.arccos(predictions[:,0])
        time = inputs[:,0]

        matrix = np.column_stack((theta,time))

        frames[i] = matrix
        
#        plot_fft(inputs, predictions, magnetic_field)
    
        frequency_max_value = fft_modulus_predicted(predictions)
          
        larmor_frequencies[i] = frequency_max_value
        field_values[i] = magnetic_field
    
        i +=1
    
    return larmor_frequencies, field_values, frames


def fft_real(final_magnetic_field, initial_magnetic_field, data):
    
    item = int((initial_magnetic_field-final_magnetic_field)/(0.01))
        
    larmor_frequencies = [None]*(item)
    field_values = [None]*(item)
    frames = [None]*(item)
    
    data, frames = select_field_values(data, initial_magnetic_field, final_magnetic_field)
        
    B_ext = np.arange(final_magnetic_field, initial_magnetic_field, 0.01)
    
    i = 0
    
    for magnetic_field in B_ext:
        
        inputs, targets = split_data(frames[i])
        
        theta = np.arccos(targets[:,0])
        time = inputs[:,0]
        
        matrix = np.column_stack((theta, time))
        
        frames[i] = matrix
        
#        plot_original_fft(inputs, targets, magnetic_field, 'original_data')
#        plot_predictions(str(magnetic_field), inputs, targets)
    
        frequency_max_value = fft_modulus(targets)
          
        larmor_frequencies[i] = frequency_max_value
        field_values[i] = magnetic_field
    
        i +=1
    
    return larmor_frequencies, field_values, frames


# This function calculates the fft of a given variable, and returns the high modulus value of fft
def fft_modulus(predictions, target='my'):
    
    # get data
    if target == "mx":
        m = predictions[:,0]
    elif target == "my":
        m = predictions[:,1]
    elif target == "mz":
        m = predictions[:,2]
                
    # Necessary parameters      
    n = m.size    
    sample_rate = 2.5e-11
      
    #calculate fft of magnetization (m(w))
    fourier_transform = np.fft.rfft(m)
    #calculates the frequency
    frequency = np.fft.rfftfreq(n, sample_rate)
    
    #calculate modulus of furier transform values (m(w))
    modulus = np.abs(fourier_transform)
    #calculate the max value 
    max_value = np.amax(modulus)

    max_value = np.where(modulus == max_value)
    
    frequency_maximum_value = frequency[max_value]
        
    return frequency_maximum_value

# This function calculates the fft of a given variable, and returns the high modulus value of fft
def fft_modulus_predicted(predictions, target='my'):
    
    # get data
    if target == "mx":
        m = predictions[:,0]
    elif target == "my":
        m = predictions[:,1]
    elif target == "mz":
        m = predictions[:,2]
        
    # Necessary parameters      
    n = m.size    
    sample_rate = 1.001e-11
  
    #calculate fft of magnetization (m(w))
    fourier_transform = np.fft.rfft(m)
    #calculates the frequency
    frequency = np.fft.rfftfreq(n, sample_rate)
    
    #calculate modulus of furier transform values (m(w))
    modulus = np.abs(fourier_transform)
    #calculate the max value 
    max_value = np.amax(modulus)

    max_value = np.where(modulus == max_value)
    
    frequency_maximum_value = frequency[max_value]
        
    return frequency_maximum_value


def plot_original_fft(inputs, targets, magnetic_field, label='prediction_data',target='my'):
    
    if target == "mx":
        m = targets[:,0]
    elif target == "my":
        m = targets[:,1]
    elif target == "mz":
        m = targets[:,2]

# Necessary parameters      
    n = m.size    
    sample_rate = 2.5e-11


# Calculates fast fourier transform    
    ft = np.fft.rfft(m)
    frequency = np.fft.rfftfreq(n, sample_rate)
    
#Plot fourier transform
    plt.figure()
    plt.ylabel("Amplitude m(w)")
    plt.xlabel("Frequency [Hz]")
    plt.title('Fourier Transform (B = '+str(magnetic_field)+' T)')
    plt.plot(frequency, ft.real**2 + ft.imag**2, label=label)
    plt.legend()
    plt.show()  


def plot_fft(inputs, targets, magnetic_field, label='prediction_data',target='my'):
    
    if target == "mx":
        m = targets[:,0]
    elif target == "my":
        m = targets[:,1]
    elif target == "mz":
        m = targets[:,2]

# Necessary parameters      
    n = m.size    
    sample_rate = 1.001e-11


# Calculates fast fourier transform    
    ft = np.fft.rfft(m)
    frequency = np.fft.rfftfreq(n, sample_rate)
    
#Plot fourier transform
    plt.figure()
    plt.ylabel("Amplitude m(w)")
    plt.xlabel("Frequency [Hz]")
    plt.title('Fourier Transform (B = '+str(magnetic_field)+' T)')
    plt.plot(frequency, ft.real**2 + ft.imag**2, label=label)
    plt.legend()
    plt.show() 
  
# Define the custom loss to load the model without porblems
def custom_loss(y_true, y_predict):
    
    mx = y_true[:, 0]
    my = y_true[:, 1]
    mz = y_true[:, 2]
    
    mx_p = y_predict[:, 0]
    my_p = y_predict[:, 1]
    mz_p = y_predict[:, 2]

    l2 = K.sum(K.abs(mx-mx_p)+K.abs(my-my_p)+ K.abs(mz-mz_p))
    
    l1 = K.sum(K.sqrt(K.pow(mx_p,2)+K.pow(my_p,2)+K.pow(mz_p,2)) -1)
    
    l = l1+l2
    
    return l
    
# Load model and scalers obtains during training
def update_model():
    #load model
    model= load_model("model.h5", custom_objects={'custom_loss': custom_loss})
    #load scaling method
    x_training_scaler = load(open('x_scaler.pkl', 'rb'))
    y_training_scaler = load(open('y_scaler.pkl', 'rb'))
    #summary model
    model.summary()
    
    return model, y_training_scaler, x_training_scaler

# Made a predictions with a given data, i.e. magnetic field
def make_predictions(magnetic_field, alpha, model, y_training_scaler, x_training_scaler, max_time=0.0):

    #Prepare the input data
    time = np.arange(0,1e-8,1.001e-11)
    B = np.array([magnetic_field for i in range(1000)])
    dumping_factor = np.array([alpha for i in range(1000)])
    magnetic_field = str(magnetic_field)

    inputs = np.column_stack((time, B, dumping_factor))
    
    # Make predictions with new data
    predictions = model.predict(x_training_scaler.transform(inputs))
    #unscale data and normalize to obtain the good results
    predictions = y_training_scaler.inverse_transform(predictions)  
    # Plot the results
    plot_predictions(magnetic_field, inputs, predictions)
    
    return inputs, predictions

# This function plot on screen the predictions
def plot_predictions(magnetic_field, inputs, predictions):
    
    #plot predictions
    plt.figure()
    plt.plot(inputs[:, 0], predictions[:, 0], label = 'mx_prediction')
    plt.plot(inputs[:, 0], predictions[:, 1], label = 'my_prediction')
    plt.plot(inputs[:, 0], predictions[:, 2],label = 'mz_prediction')
    plt.legend()
    plt.xlabel('time (s)')
    plt.title('B = '+magnetic_field+' T')    
    
