import pandas as pd
import numpy as np
from scipy import stats



import matplotlib.pyplot as plt


def filter_data(data, alpha, target=''):
    
    data_filtered  = data[(data.alpha == alpha)]
    
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


def select_field_values(data, initial_magnetic_field, final_magnetic_field):
        
    one = data[(data.B_ext_x == final_magnetic_field)] 
    
    two = data[(data.B_ext_x == 0.02)]
    
    three = data[(data.B_ext_x == 0.03)]
    
    four = data[(data.B_ext_x == 0.04)]

    five = data[(data.B_ext_x == 0.05)] 
    
    six = data[(data.B_ext_x == 0.06)]
    
    seven = data[(data.B_ext_x == 0.07)]
    
    eight = data[(data.B_ext_x == 0.08)]
    
    nine = data[(data.B_ext_x == 0.09)]

#    ten = data[(data.B_ext_x == final_magnetic_field)] 
    
    frames =[one, two, three, four, five, six, seven, eight, nine]

    return pd.concat(frames), frames

def split_data(data):
    
    # split data
    inputs = data.iloc[:,[0,4,5]]
    targets = data.iloc[:,1:4]
    # transform DataFrame into numpy array
    inputs = inputs.to_numpy()
    targets = targets.to_numpy()
    
    return inputs, targets

# This function is valid to obtain real data storage in repository and split into inputs
# and targets, using for validate the predictions. It is necesary introduce the filename of file.
# returns numpy arrays
def get_real_data(filename, sep=','):
    
    # read data
    data = pd.read_csv("b_0.065.csv",sep=',')
    # split data
    inputs = data.iloc[:,[0,4]]
    targets = data.iloc[:,1:4]
    # transform DataFrame into numpy array
    inputs = inputs.to_numpy()
    targets = targets.to_numpy()
    
    return inputs, targets

# This functions return a raw data in DataFrame format
def get_data(filename, sep=''):
    
    data = pd.read_csv(filename,sep=',')
    
    return data