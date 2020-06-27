import flask
from flask import Response

from flask_socketio import SocketIO
from pickle import load
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

import io


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def custom_loss(y_true, y_predict):
    
    mx = y_true[:, 0]
    my = y_true[:, 1]
    mz = y_true[:, 2]
    
    mx_p = y_predict[:, 0]
    my_p = y_predict[:, 1]
    mz_p = y_predict[:, 2]

    error_term = K.sum(K.abs(mx-mx_p)+K.abs(my-my_p)+ K.abs(mz-mz_p))
    
    modulus_m = K.sum(K.sqrt(K.pow(mx_p,2)+K.pow(my_p,2)+K.pow(mz_p,2)) -1)
    
    loss = error_term + modulus_m
    
    return loss



def anisotropy_predictions(magnetic_field, alpha):
    
    #Load model and scales
    model= load_model("model/model.h5", custom_objects={'custom_loss': custom_loss})
    
    x_training_scaler = load(open('model/x_scaler.pkl', 'rb'))
    y_training_scaler = load(open('model/y_scaler.pkl', 'rb'))

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

    
    return inputs, predictions


def prediction(magnetic_field, alpha):

    model= load_model("model/field_model.h5", custom_objects={'custom_loss': custom_loss})

    x_training_scaler = load(open('model/field_x_scaler.pkl', 'rb'))
    y_training_scaler = load(open('model/field_y_scaler.pkl', 'rb'))

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

    
    return inputs, predictions

    
app = flask.Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
# app.config['DEBUG'] = True
# app.config['THREAD'] = True
socketio = SocketIO(app)



@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('index.html'))
    if flask.request.method == 'POST':
        magnetic_field = flask.request.form['magnetic_field']
        dumping_factor = flask.request.form['dumping_factor']
        anisotropy = flask.request.form.get('anisotropy')

        
        
#        windspeed = flask.request.form['windspeed']
        
#        input_variables = pd.DataFrame([[magnetic_field, dumping_factor]],
#                                       columns=['temperature', 'humidity'],
#                                       dtype=float)
        global inputs
        global predictions
        
        
        if(anisotropy == '1'):
            inputs, predictions = anisotropy_predictions(magnetic_field, dumping_factor)
        else:
            inputs, predictions = prediction(magnetic_field, dumping_factor)
            
        table = np.column_stack((inputs[:,0],predictions))
        predictions_df=pd.DataFrame(data=table[:6,0:], columns = ["time(s)","mx","my","mz"], dtype = float)  
        
        if(anisotropy == '1'):
            return flask.render_template('index.html',
                                     original_input={'Magnetic Field(T)':magnetic_field,
                                                     'Dumping Factor':dumping_factor},
                                    result=[predictions_df.to_html(classes='data', header='false', table_id='anisotropy', index='false')],
                                     )
        else:
            return flask.render_template('index.html',
                                     original_input={'Magnetic Field':magnetic_field,
                                                     'Dumping Factor':dumping_factor},
                                    result1=[predictions_df.to_html(classes='data', header='false', table_id='anisotropy', index='false')],
                                     )
        
@app.route('/table/')
def table():
    table = np.column_stack((inputs[:,0],predictions))
    table_df = pd.DataFrame(data=table[0:,0:], columns = ["time(s)","mx", "my", "mz"]) 
    return flask.render_template('table.html', result=[table_df.to_html(classes='data', header='false', table_id='anisotropy', index='false')] )

@app.route('/exportTable')
def download_table():
    table = np.column_stack((inputs[:,0],predictions))
    table_df = pd.DataFrame(data=table[0:,0:], columns = ["time(s)","mx", "my", "mz"]) 
    response = flask.make_response(table_df.to_csv())
    response.headers["Content-Disposition"] = "attachment; filename=table.csv"
    response.headers["Content-Type"] = "text/csv"
    return response
    
@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    canvas = FigureCanvas(fig)
    output = io.BytesIO()
    canvas.print_png(output)
    response = flask.make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response


def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.plot(inputs[:, 0], predictions[:, 0], label = 'mx_prediction')
    axis.plot(inputs[:, 0], predictions[:, 1], label = 'my_prediction')
    axis.plot(inputs[:, 0], predictions[:, 2],label = 'mz_prediction')
    axis.legend()
    axis.set_xlabel('time (s)')
    axis.set_ylabel('m')
    return fig
	

if __name__ == '__main__':
    socketio.run(app)