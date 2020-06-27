import pandas as pd
import numpy as np
from scipy import stats


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

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