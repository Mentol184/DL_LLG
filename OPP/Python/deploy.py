import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

from mpl_toolkits.mplot3d import Axes3D

import os
import sys

from keras.models import load_model

import keras.backend as K

from pickle import load

import matplotlib.pyplot as plt

from fft import fft_testing, fft_simulation
from plot_methods import *



# Define the custom loss to load the model without porblems
def custom_loss(y_true, y_predict):
    
    l2 = K.sum(K.abs(y_true-y_predict))
    l1 = K.sum(K.sqrt(K.pow(y_predict,2) -1))
    
    l = l1+l2
    
    return l

def update_model(modelo):
    #load model
    model= load_model("model_"+modelo+".h5", custom_objects={'custom_loss': custom_loss})
    #load scaling method
    x_training_scaler = load(open('x_scaler_'+modelo+'.pkl', 'rb'))
    y_training_scaler = load(open('y_scaler_'+modelo+'.pkl', 'rb'))
    #summary model
    model.summary()
    
    return model, y_training_scaler, x_training_scaler

# Made a predictions with a given data, i.e. magnetic field
def make_predictions(current, model, y_training_scaler, x_training_scaler, max_time=0.0):

    #Prepare the input data
    time = np.arange(0,0.5e-7,1.001e-11)
    
    points = time.size
    
    J = np.array([current for i in range(points)])

    inputs = np.column_stack((time, J))
    
    # Make predictions with new data
    predictions = model.predict(x_training_scaler.transform(inputs))
    #unscale data and normalize to obtain the good results
    predictions = y_training_scaler.inverse_transform(predictions)
    
    # plot_predictions(current, inputs, predictions)

    
    return inputs, predictions

def compare_data(J, data, model, y_training_scaler, x_training_scaler):
    
    data = data[ (data['Jz (A/m2)'] == J)]
    
    inputs, predictions = make_predictions(J, model, y_training_scaler, x_training_scaler)
    
    x = data.iloc[:,[0]]
    mx = data.iloc[:,[1]]
    my = data.iloc[:,[2]]
    mz = data.iloc[:,[3]]
    
    #Plot mx
    plt.figure()
    plt.plot(inputs[:, 0], predictions[:, 0], label = 'mx_prediction')
    plt.plot(x, mx,'--', label = 'mx')
    plt.legend()
    plt.xlabel('time (s)')
    plt.title('J ='+str(J)+'A/m^2')
    plt.show()
    plt.savefig('images/mx'+str(J)+'.png')
    
    #Plot my
    plt.figure()
    plt.plot(inputs[:, 0], predictions[:, 1], label = 'my_prediction')
    plt.plot(x, my, '--',label = 'my')
    plt.legend()
    plt.xlabel('time (s)')
    plt.title('J ='+str(J)+'A/m^2')
    plt.show()
    plt.savefig('images/my'+str(J)+'.png')
    
    #Plot mz
    plt.figure()
    plt.plot(inputs[:, 0], predictions[:, 2], label = 'mz_prediction')
    plt.plot(x, mz, '--', label = 'mx')
    plt.legend()
    plt.xlabel('time (s)')
    plt.title('J ='+str(J)+'A/m^2')
    plt.show()
    plt.savefig('images/mz'+str(J)+'.png')
    
def compare_data_3D(modelo,J):
    
    data = pd.read_csv("dataset_enorme.csv",sep=',')
    
    data = data[ (data['Jz (A/m2)'] == J)]
    
    data.to_numpy()
    
    model, y_training_scaler, x_training_scaler = update_model(modelo)
    
    inputs, predictions = make_predictions(J, model, y_training_scaler, x_training_scaler)
    
    mx_p = predictions[:200,0]
    my_p = predictions[:200,1]
    mz_p = predictions[:200,2]
    
    mx = data.iloc[0:1000,1]
    my = data.iloc[0:1000,2]
    mz = data.iloc[0:1000,3]
    

    
    mx = np.array(mx)
    my = np.array(my)
    mz = np.array(mz)
    

    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot3D(mx, my, mz, label='parametric curve')
    ax.plot3D(mx_p, my_p, mz_p, label='prediction curve')
    ax.legend()
    plt.show()
        

def plot_predictions(magnetic_field, inputs, predictions):

    
    #plot predictions
    plt.figure()
    plt.plot(inputs[:, 0], predictions[:, 0], label = 'mx_prediction')
    plt.plot(inputs[:, 0], predictions[:, 1], label = 'my_prediction')
    plt.plot(inputs[:, 0], predictions[:, 2], label = 'mz_prediction')
    plt.legend()
    plt.xlabel('time (s)')
    
    
def plot_3D(modelo, current):
    
    model, y_training_scaler, x_training_scaler = update_model(modelo)
    
    inputs, predictions = make_predictions(current, model, y_training_scaler, x_training_scaler)
    
    mx = predictions[:200,0]
    my = predictions[:200,1]
    mz = predictions[:200,2]
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot3D(mx, my, mz, label='parametric curve')
    ax.legend()
    plt.show()


modelo = "modelo_entero_3"
J = 11.5e9
steps = 10e9
samples = 10



model, y_training_scaler, x_training_scaler = update_model(modelo)

frequencies, current_values = fft_testing(J, steps, samples, model, y_training_scaler, x_training_scaler)
simulation_frequencies, current_values = fft_simulation(10e9, steps, samples, model, y_training_scaler, x_training_scaler,"../dataset_enorme.csv")

    
frequencies = (np.array(frequencies)).reshape(samples)
simulation_frequencies = (np.array(simulation_frequencies)).reshape(samples)


plt.figure()
plt.plot(current_values, frequencies, 'o', label = 'Predicted')
plt.plot(current_values, simulation_frequencies, 'o', label = 'MuMax')
# plt.plot(field_values, new_y, label='Linear regression')
# plt.text(0.05, 0.5e9, r'$f_{larmor} = \frac{\gamma}{2\pi} B$', fontsize=15)
# plt.text(0.05, 0.8e9, r'$\gamma =$'+str(slope)+'', fontsize=15)
# plt.text(0.05, 0.6e9, r'$r^2 =$'+str(r_value**2)+'', fontsize=15)
plt.ylabel(r'$f (Hz)$')
plt.xlabel(r'$J (A/m^2)$')
plt.title('FFT')
plt.legend()
plt.show()



# data = pd.read_csv("dataset_enorme.csv",sep=',')
# compare_data(J,data,model,y_training_scaler, x_training_scaler)
# plot_3D(modelo, J)
# compare_data_3D(modelo, J)

# make_predictions(J, model, y_training_scaler, x_training_scaler)