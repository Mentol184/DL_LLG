import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

import os

from keras.models import load_model

import keras.backend as K

from pickle import load

import matplotlib.pyplot as plt

from fft import fft_testing, fft_real
from data_preparation import get_data, filter_data, split_data


def main():
    
    magnetic_field = 0.1
    final_magnetic_field = 0.01
    alpha = 0.006
        
    model, y_training_scaler, x_training_scaler = update_model()
    
    make_predictions(magnetic_field, alpha, model, y_training_scaler, x_training_scaler)
    
#     plot_simulation_data(magnetic_field, alpha)
    
    predicted_larmor_frequencies, field_values, frames = calculate_predicted_larmor_frequencies(final_magnetic_field,
                                                                                          magnetic_field,
                                                                                          alpha,
                                                                                          model, 
                                                                                          y_training_scaler, 
                                                                                          x_training_scaler)
# #    
    # larmor_frequencies, real_field_values, real_frames = calculate_larmor_frequencies('data_set_testing.csv',
    #                                                                       final_magnetic_field,
    #                                                                       magnetic_field, 
    #                                                                       alpha)
    
# #    
# #    ''' comparar tiempos característicos '''
    
    larmor_frequencies = []
    real_frames = []
    
    characteristic_time(predicted_larmor_frequencies, larmor_frequencies, field_values, alpha, frames, real_frames)
    
##    characteristic_time_simulation_data(larmor_frequencies, real_field_values, alpha, real_frames)
    

def plot_simulation_data(magnetic_field, alpha, filename='data_set_testing.csv'):
    
    raw_data = get_data(filename)
     
    data = filter_datas(raw_data,magnetic_field,alpha)
    
    print(data)
    
    inputs, targets = split_data(data)
    
    # plot_data(str(magnetic_field), inputs, targets)
    
    return inputs, targets
    
def filter_datas(data, magnetic_field, alpha, target=''):
    
    data_filtered  = data[(data.alpha == alpha)]
    
    data_filtered  = data_filtered[(data_filtered.B_ext_x == magnetic_field)]
    
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

def plot_data(magnetic_field, inputs, targets):
    
    #plot predictions
    plt.figure()
    plt.plot(inputs[:, 0], targets[:, 0], label = 'mx')
    plt.plot(inputs[:, 0], targets[:, 1], label = 'my')
    plt.plot(inputs[:, 0], targets[:, 2],label = 'mz')
    plt.legend()
    plt.xlabel('time (s)')
    plt.title('B = '+magnetic_field+' T')

def func(t,a,b):
    
    return a*np.exp(b*t)
def func2(t,a,b):
    
    return a*np.exp(b*t)
    
def characteristic_time(predicted_larmor_frequencies, larmor_frequencies,  field_values, alpha, frames, real_frames):
    
    items = predicted_larmor_frequencies.size
    
    coefficient = [None]*items
    
    for i in range(items):
    
        matrix = frames[i]

        theta = matrix[:,0]
        time = matrix[:,1]
    
        y = np.tan(theta/2)

        popt, pcov = curve_fit(func, time, y)
        
        coefficient[i] = popt[1]
    
        y_fit = func(time, popt[0], popt[1])
        
        
        
    # coefficients = characteristic_time_simulation_data(larmor_frequencies, field_values, alpha, real_frames)

    # df = pd.DataFrame([field_values[i], alpha, predicted_larmor_frequencies[i], larmor_frequencies[i], coefficient[i], coefficients[i]]
    df = pd.DataFrame([field_values[i], alpha, predicted_larmor_frequencies[i], coefficient[i]]
    for i in range(items))
    
    df.to_excel('alpha_'+str(alpha)+'.xlsx')
    
    
    
    
  
    

    
    tau = (-1.)*np.divide(1.0,coefficient)
    
    field_values = np.array(field_values)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(1./field_values, tau)
    
    print(slope)
    
    new_y = (np.array(1./field_values))*slope + intercept
    
    plt.figure()
    plt.plot(1./field_values, tau,'o')
    plt.plot(1./field_values, new_y, 'r-')
    plt.title('alpha ='+str(alpha)+'')
    plt.xlabel('$B^{-1} (T^{-1})$')
    plt.text(60, 2.8e-8, r'$\tau_B = \frac{1}{\alpha \gamma}\cdot \frac{1}{B}$', fontsize=15)
    plt.ylabel(r'$\tau_B $')
    plt.legend()
    plt.show()
    
    plt.savefig('alpha_'+str(alpha)+'_tau.png')
        
    
#     plt.figure()
    
#     plt.plot(time, y,'o')
#     plt.plot(time, y_fit, 'r-')
#     plt.title('Predicted data: alpha ='+str(alpha)+' B='+str(field_values[items-1])+'')
# #    plt.text(0, 0.3, r'$tan(\theta/2) = tan\theta_0 exp(\lambda w_k (t-t_0))$', fontsize=15)
#     plt.xlabel('time (s)')
#     plt.ylabel(r'$tan(\theta/2) $')
#     plt.legend()
#     plt.show()
    
#     plt.savefig('alpha_'+str(alpha)+'_predicted_regression.png')
    
def characteristic_time_simulation_data(larmor_frequencies, field_values, alpha,  frames):
    
    items = larmor_frequencies.size
    
    coefficients = [None]*items
    
    for i in range(items):
    
        matrix = frames[i]

        theta = matrix[:,0]
        time = matrix[:,1]
    
        y = np.tan(theta/2)

        popt, pcov = curve_fit(func, time, y)
        
        coefficients[i] = popt[1]
        
        y_fit = func(time, popt[0], popt[1])
        
    
    plt.figure()
    
    plt.plot(time, y,'o')
    plt.plot(time, y_fit, 'r-')
    plt.title('Simulation data: alpha ='+str(alpha)+'  B='+str(field_values[items-1])+'')
    plt.xlabel('time (s)')
    plt.ylabel(r'$tan(\theta/2) $')
    plt.legend()
    plt.show()
    
    plt.savefig('alpha_'+str(alpha)+'_simulation_regression.png')
    
    return  coefficients
    
    
#def characteristic_time(predicted_larmor_frequencies, alpha, frames):
#    
#    model, y_trainig_scaler, x_training_scaler = update_model()
#    
#    inputs, predictions = make_predictions(magnetic_field, alpha, model, y_trainig_scaler, x_training_scaler)
#       
#    df = pd.DataFrame([larmor_frequencies[i], predicted_larmor_frequencies[i],1/alpha*larmor_frequencies[i], 1/alpha*predicted_larmor_frequencies[i]]
#    for i in range(larmor_frequencies.size))
#    
#    df.to_excel('alpha_'+str(alpha)+'.xlsx')
#    df.to_csv('alpha_'+str(alpha)+'.csv')
#    
#    plt.figure()
#    plt.table(cellText=df.values,
##              colWidths = [0.25]*len(df.columns),
#              rowLabels=df.index,
#              colLabels=df.columns,
#              cellLoc = 'center', rowLoc = 'center',
#              loc='center')
##    fig = plt.gcf()
#
#    plt.show()
        
        

def calculate_predicted_larmor_frequencies(final_magnetic_field, magnetic_field, alpha, model, y_training_scaler, x_training_scaler):

    larmor_frequencies, field_values, frames = fft_testing(final_magnetic_field, magnetic_field, alpha, model, y_training_scaler, x_training_scaler)
    larmor_frequencies = (np.array(larmor_frequencies)).reshape(9)

    slope, intercept, r_value, p_value, std_err = stats.linregress(field_values, larmor_frequencies)
    
    new_y = (np.array(field_values))*slope + intercept
    
    plt.figure()
#    plt.plot(field_values, larmor_frequencies)
    plt.plot(field_values, larmor_frequencies, 'o', label = 'Values')
    plt.plot(field_values, new_y, label='Linear regression')
    plt.text(0.05, 1.0e9, r'$f_{larmor} = \frac{\gamma}{2\pi} B$', fontsize=15)
    plt.ylabel(r'$f_{Larmor} (Hz)$')
    plt.xlabel('B (T)')
    plt.title('FFT, alpha ='+str(alpha)+'')
    plt.legend()
    plt.show()
    
    plt.savefig('alpha_'+str(alpha)+'_predicted.png')
    
    print(slope)
    
    return larmor_frequencies, field_values, frames

def calculate_larmor_frequencies(filename, final_magnetic_field, magnetic_field, alpha):
    
    raw_data = get_data(filename)
    
    data = filter_data(raw_data, alpha)
    
    real_larmor_frequencies, real_field_values, frames = fft_real(final_magnetic_field, magnetic_field, data)
    real_larmor_frequencies = (np.array(real_larmor_frequencies)).reshape(9)
    
    real_slope, real_intercept, real_r_value, real_p_value, std_err = stats.linregress(real_field_values, real_larmor_frequencies)
    
    real_new_y = (np.array(real_field_values))*real_slope + real_intercept
    
    plt.figure()
    plt.plot(real_field_values, real_larmor_frequencies, 'o', label = 'Values')
    plt.plot(real_field_values, real_new_y, label='Linear regression')
    plt.text(0.05, 1.0e9, r'$w_{larmor} = \gamma B$', fontsize=15)
    plt.text(0.05, 0.8e9, r'$\gamma =$'+str(real_slope)+'', fontsize=15)
    plt.text(0.05, 0.6e9, r'$r^2 =$'+str(real_r_value**2)+'', fontsize=15)
    plt.ylabel(r'$w_{Larmor} (Hz)$')
    plt.xlabel('B (T)')
    plt.title('FFT (simulation data), alpha='+str(alpha)+'')
    plt.legend()
    plt.show()
    
    plt.savefig('alpha_'+str(alpha)+'_simulation.png')
    
    return real_larmor_frequencies, real_field_values, frames
    

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
    model= load_model("field_model.h5", custom_objects={'custom_loss': custom_loss})
    #load scaling method
    x_training_scaler = load(open('field_x_scaler.pkl', 'rb'))
    y_training_scaler = load(open('field_y_scaler.pkl', 'rb'))
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
    # plot_predictions(magnetic_field, inputs, predictions)
    
    return inputs, predictions

# This function plot on screen the predictions
def plot_predictions(magnetic_field, inputs, predictions):
    
    simulation_inputs, targets = plot_simulation_data(0.01, 0.01)
    
    
    #plot predictions
    plt.figure()
    plt.plot(inputs[:, 0], predictions[:, 0], label = 'mx_prediction')
    plt.plot(inputs[:, 0], predictions[:, 1], label = 'my_prediction')
    plt.plot(inputs[:, 0], predictions[:, 2],label = 'mz_prediction')
    plt.plot(simulation_inputs[:, 0], targets[:, 0], '--', label = 'mx_mumax')
    plt.plot(simulation_inputs[:, 0], targets[:, 1], '--', label = 'my_mumax')
    plt.plot(simulation_inputs[:, 0], targets[:, 2], '--', label = 'mz_mumax')
    plt.legend()
    plt.xlabel('time (s)')
    plt.title('B = '+magnetic_field+' T & alpha ='+str(0.01)+'')

    
# Main define
if __name__ == '__main__':
    
    a = main()




