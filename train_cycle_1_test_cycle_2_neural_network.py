
import numpy as np
import tensorflow as tf
from scipy.fft import fft, fftfreq
import os
import matplotlib.cm as cm
import pandas as pd
from utils import preprocess_data
from itertools import groupby
from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import stats
import scipy.optimize as opt
import math
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from scipy.optimize import curve_fit

tf.get_logger().setLevel('ERROR')

folder_cycle1 = r'D:\Downloads\Sensor_1_ESR_cycling-20240213T204739Z-001\Sensor_1_ESR_cycling\sensor1_esr_temp_cycle_1'
folder_cycle2 =  r'D:\Downloads\sensor_1_ESR_cycle_2'


# Get a list of all the files in the folders (excluding the PARAMS file)
cycle1_files = os.listdir(folder_cycle1)
cycle1_files = [f for f in cycle1_files if "PARAMS" not in f]
cycle2_files = os.listdir(folder_cycle2)
cycle2_files = [f for f in cycle2_files if "PARAMS" not in f]

# Defining Parameters
s1 = np.array([[0.0,1.0,0.0],
    [1.0,0.0,1.0],
    [0.0,1.0,0.0]])

s2 = np.array([[0.0,-1.0j,0.0],
    [1.0j,0.0,-1.0j],
    [0.0,1.0j,0.0]])

s3 = np.array([[1.0,0.0,0.0],
    [0.0,0.0,0.0],
    [0.0,0.0,-1.0]])

spin1 = (1.0/np.sqrt(2.0))*s1
spin2 = (1.0/np.sqrt(2.0))*s2
spin3=s3


spin1 = tf.constant(spin1, dtype = 'complex128')
spin2 = tf.constant(spin2, dtype = 'complex128')
spin3 = tf.constant(spin3, dtype = 'complex128')

# a=tf.constant(-7.86851953723355e-05,dtype='float64')# Linear Regression
# b= tf.constant(2.870665858002803,dtype='float64') # Linear Regression
# a=tf.constant(-7.772773696087131e-05,dtype='float64') # new regression
# b=tf.constant(2.8706500246147373,dtype='float64') # new regression
b= tf.constant( 2.87068615576284,dtype='float64') # Grad search cycle 2 
a=tf.constant(-7.723607188481802e-05, dtype='float64') # Grad search cycle 2 
c=tf.constant( -4.3478260869566193e-07,dtype='float64')
d=tf.constant(0.005185511627906974,dtype='float64')#Literature Value


v = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_v)
w = tf.constant(0.00, dtype = 'float64')    # ~N(0,sigma_w)

P_0 = tf.constant(1e-4, dtype = 'float64')
P = tf.constant(0.18, dtype = 'float64')
alpha= tf.constant(14.52e-3, dtype = 'float64')
I = tf.eye(3,dtype = 'complex128')
    
    
    
def getD(T, P):
    D = a * T + b + alpha * (P_0 - P_0)
    E = c * T + d + w
    return D, E

def H(D, E):
    Ham = tf.complex(D * (tf.math.real(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.real(spin1 @ spin1 - spin2 @ spin2)),
                    D * (tf.math.imag(spin3 @ spin3 - 2 / 3 * I)) + E * (tf.math.imag(spin1 @ spin1 - spin2 @ spin2)))
    return Ham


@tf.autograph.experimental.do_not_convert
@tf.function
def getP_k(T, P):
    D, E = getD(T, P)
    Ham = H(D, E)
    eigenvalues = tf.linalg.eigvals(Ham)
    return eigenvalues


@tf.function
def bilorentzian(x, T, P):
    eigenvalues = getP_k(T, P)
    x0 = tf.cast(eigenvalues[1] - eigenvalues[2], tf.float64)
    x01 = tf.cast(eigenvalues[0] - eigenvalues[2], tf.float64)
    x = tf.cast(x, tf.float64)
    a = tf.cast( 47.64938163757324, tf.float64)  # cycle2
    gamma = tf.cast(0.004283152054995298, tf.float64)  #cycle2
    
    return a * gamma**2 / ((x - x0)**2 + gamma**2) + a * gamma**2 / ((x - x01)**2 + gamma**2)

def _get_vals(T, P):
    timespace = np.linspace(start_frequency_cycle2, end_frequency_cycle2, num=N)
    timespace = tf.cast(timespace, 'float64')
    vals = bilorentzian(timespace, T, P)
    return tf.reshape(vals, [N, 1])



# Reading Data and taking everything that can be changed
delimiter = "\t"
variable_names = ["Frequency", "Intensity1", "Intensity2"]   


# Cycle 2 (Testing Data)
test_data=[]
temperatures_cycle2 = [-30.0, -20.0, -10.0,  0.0, 10.0,  20.0, 30.0,  40.0, 50.0,40.0, 30.0,  20.0, 10.0, 0.0, -10.0, -20.0, -30.0]
num_files_per_temp_cycle2 = 2
Frequency_cycle2 = None
# Process each group of 20 files
for i in range(0, len(cycle2_files), num_files_per_temp_cycle2):
    files_group_cycle2 = cycle2_files[i:i+num_files_per_temp_cycle2]
    temp_cycle2 = temperatures_cycle2[i//num_files_per_temp_cycle2]  # Get the corresponding temperature for this group
    T = tf.constant(temp_cycle2, dtype=tf.float64)
    ratios_cycle2 = np.array([])

   
    for file in files_group_cycle2:
        data_cycle2 = pd.read_csv(os.path.join(folder_cycle2, file), delimiter=delimiter, header=None, names=variable_names)

        ratio_cycle2 = np.divide(data_cycle2['Intensity2'], data_cycle2['Intensity1'])
        if ratios_cycle2.size == 0:
            ratios_cycle2 = np.array([ratio_cycle2])
        else:
            ratios_cycle2 = np.vstack((ratios_cycle2, [ratio_cycle2]))  # Add ratio to the numpy array

    avg_intensity_cycle2 = np.mean(ratios_cycle2, axis=0)
    if Frequency_cycle2 is None:
        Frequency_cycle2 = data_cycle2['Frequency'].values
        # Assuming Frequency is in Hz
        Frequency_GHz_cycle2 = Frequency_cycle2 / 1e9
        start_frequency_cycle2 = np.min(Frequency_cycle2)/1e9

    end_frequency_cycle2 = np.max(Frequency_cycle2)/1e9

    N = Frequency_cycle2.shape[0]
    dt = np.round((end_frequency_cycle2 - start_frequency_cycle2) / N, 4)

    timespace = np.linspace(start_frequency_cycle2, end_frequency_cycle2, num=N)
    sim_val = _get_vals(T, P)
    noise_sample_cycle2= avg_intensity_cycle2[np.where(np.abs(timespace)<2.85)[0]] 
    noise_mean_cycle2 = np.mean(noise_sample_cycle2)
    avg_intensity_cycle2 = avg_intensity_cycle2 - noise_mean_cycle2
    avg_intensity_cycle2 = np.max(sim_val)*( avg_intensity_cycle2)/(np.max(avg_intensity_cycle2))
    noise_sample_cycle2= avg_intensity_cycle2[np.where(np.abs(timespace)<2.85)[0]]
    std_noise_cycle2=np.std(noise_sample_cycle2)
    test_data.append(avg_intensity_cycle2)
    




all_data_test_2D = np.array(test_data)

# Cycle 1 (Training Data)



all_data = []
all_temperatures = []
all_roots = []
mt_list, mt_orig_list, valt_list, valt_orig_list = [[] for _ in range(4)]

# Reading Data and taking everything that can be changed
delimiter = "\t"
variable_names = ["Frequency", "Intensity1", "Intensity2"]   
Frequency = None 
num_files_per_temp = 20
temperatures = [25, 25, 30, 35, 40, 45, 50, 45, 40, 35, 30, 25, 20, 15, 10, 10]

train_data=[]

temperatures = temperatures[1:]

# Process each group of 20 files
for i in range(num_files_per_temp, len(cycle1_files), num_files_per_temp):
    files_group = cycle1_files[i:i+num_files_per_temp]
    temp = temperatures[(i//num_files_per_temp)-1]   # Get the corresponding temperature for this group
    T = tf.constant(temp, dtype=tf.float64)
# Process each group of 20 files
    ratios = np.array([])

   
    for file in files_group:
        data = pd.read_csv(os.path.join(folder_cycle1, file), delimiter=delimiter, header=None, names=variable_names)

        ratio = np.divide(data['Intensity2'], data['Intensity1'])
        if ratios.size == 0:
            ratios = np.array([ratio])
        else:
            ratios = np.vstack((ratios, [ratio]))  # Add ratio to the numpy array

    avg_intensity = np.mean(ratios, axis=0)
    if Frequency is None:
        Frequency = data['Frequency'].values
        # Assuming Frequency is in Hz
        Frequency_GHz = Frequency / 1e9
        start_frequency = np.min(Frequency)/1e9

    end_frequency = np.max(Frequency)/1e9

    N = Frequency.shape[0]
    dt = np.round((end_frequency - start_frequency) / N, 4)

    timespace = np.linspace(start_frequency, end_frequency, num=N)
    sim_val = _get_vals(T, P)
    noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]] 
    noise_mean = np.mean(noise_sample)
    avg_intensity = avg_intensity - noise_mean
    avg_intensity = np.max(sim_val)*( avg_intensity)/(np.max(avg_intensity))
    noise_sample= avg_intensity[np.where(np.abs(timespace)<2.85)[0]]
    std_noise=np.std(noise_sample)
    train_data.append(avg_intensity)


all_data_train_2D = np.array(train_data)


# Prepare your data
X_train = all_data_train_2D.reshape((all_data_train_2D.shape[0], all_data_train_2D.shape[1], 1))
X_test = all_data_test_2D.reshape((all_data_test_2D.shape[0], all_data_test_2D.shape[1], 1))

# Prepare your labels
y_train = np.array(temperatures)
y_test = np.array(temperatures_cycle2)



model = Sequential() # Linear stack of layers





#  #Step 3: Define your model
model = Sequential() # Linear stack of layers

model.add(Conv1D(filters=80, kernel_size=10, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2)) # reduce the spatial dimensions of the data, preventing overfitting
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2)) 
model.add(Flatten()) # convert the multi-dimensional output of the previous layer into a one-dimensional vector

model.add(Dense(48, activation='relu')) 
model.add(Dropout(0.0)) # For regularization
model.add(Dense(1)) # output layer of the model.



opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')

# Step 4: Train your model
model.fit(X_train, y_train,validation_split=0.2, batch_size=5, epochs=200, verbose=0)


# Evaluate your model
loss = model.evaluate(X_test, y_test, verbose=0)

# Predict temperatures for cycle 2
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(temperatures_cycle2, y_pred))  #28.04

y_train_pred = model.predict(X_train)
# y_train_pred = y_train_pred.flatten()

# Calculate training error (RMSE)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print(f'Training RMSE: {train_rmse}') # 5.5





## Testing
y_pred = model.predict(X_test) 
y_pred = y_pred.flatten()
coefficients = np.polyfit(temperatures_cycle2, y_pred, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures_cycle2), max(temperatures_cycle2), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures_cycle2)
# Your predicted values
predicted = y_pred

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
plt.figure(figsize=(6, 4))
# Plot the data points
plt.plot(temperatures_cycle2, predicted, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('Measured Temperatures (C)', fontsize=12)
plt.ylabel('Predicted Temperatures (C)', fontsize=12)
plt.text(min(temperatures_cycle2), max(predicted)-10 , 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=1.0))
plt.tight_layout()
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()





# Training

y_train_pred = model.predict(X_train) 
y_train_pred = y_train_pred.flatten()
coefficients = np.polyfit(y_train,y_train_pred , 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(y_train), max(y_train), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(y_train)
# Your predicted values
predicted = y_train_pred 
r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(y_train, y_train_pred , 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('Measured Temperatures (C)', fontsize=12)
plt.ylabel('Predicted Temperatures (C)', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
plt.text(min(y_train), max(predicted)-10 , 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.9))
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()




# # # QUADRATIC FIT for training
coefficients = np.polyfit(y_train,  y_train_pred, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(y_train), max(y_train), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(y_train)
# Your predicted values
predicted =y_train_pred

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(y_train, y_train_pred, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('Measured Temperatures (C)', fontsize=12)
plt.ylabel('Predicted Temperatures (C)', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
plt.text(min(y_train), max(y_train_pred) - 10, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()

# # # Quadratic Fit for testing




y_pred = model.predict(X_test) 
y_pred = y_pred.flatten()
coefficients = np.polyfit(y_test,y_pred, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(y_test), max(y_test), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(y_test)
# Your predicted values
predicted =y_pred

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
# Plot the data points
plt.plot(y_test, predicted, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('Measured Temperatures (C)', fontsize=12)
plt.ylabel('Predicted Temperatures (C)', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
plt.text(min(y_test), max(predicted) -10, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()






def build_model(hp):
    model = keras.Sequential()
    model.add(Conv1D(filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
                     kernel_size=hp.Choice('conv_1_kernel', values = [3,10]),
                     activation='relu',
                     input_shape=(X_train.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=hp.Int('conv_2_filter', min_value=16, max_value=64, step=16),
                     kernel_size=hp.Choice('conv_2_kernel', values = [3,10]),
                     activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
                    activation='relu'))
    model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))  # Added dropout rate to the search space
    model.add(Dense(hp.Int('dense_2_units', min_value=1, max_value=1, step=1)))  # Added number of units in the last Dense layer to the search space


    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
                  loss='mse')

    return model



# def build_model(hp):
#     model = keras.Sequential()
#     model.add(Conv1D(filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
#                      kernel_size=hp.Choice('conv_1_kernel',min_value=3, max_value=10, step=2),
#                      activation='relu',
#                      input_shape=(X_train.shape[1], 1)))
#     model.add(MaxPooling1D(pool_size=hp.Int('pool_1_size', min_value=2, max_value=4, step=1)))  # Added pool size to the search space
#     model.add(Conv1D(filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
#                      kernel_size=hp.Choice('conv_2_kernel',min_value=3, max_value=10, step=2),
#                      activation='relu'))
#     model.add(MaxPooling1D(pool_size=hp.Int('pool_2_size', min_value=2, max_value=4, step=1)))  # Added pool size to the search space
#     model.add(Flatten())
#     model.add(Dense(units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
#                     activation='relu'))
#     model.add(Dropout(hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
#     model.add(Dense(hp.Int('dense_2_units', min_value=1, max_value=1)))

#     model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-1, 1e-3])),
#                   loss='mse')

#     return model
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,# tuner will try 5 different hyperparameter configurations
    executions_per_trial=3,# combination of hyperparameters will be tried 3 times
    directory='new project',
    project_name='new Training')

tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
best_hps_2=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('dense_1_units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

print(f"The optimal number of units in the first densely-connected layer is {best_hps.get('dense_1_units')}")
print(f"The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}")
print(f"The optimal number of filters in the first convolution layer is {best_hps.get('conv_1_filter')}")
print(f"The optimal kernel size in the first convolution layer is {best_hps.get('conv_1_kernel')}")
print(f"The optimal number of filters in the second convolution layer is {best_hps.get('conv_2_filter')}")
print(f"The optimal dropout rate is {best_hps.get('dropout')}")
print(f"The optimal kernel size in the second convolution layer is {best_hps.get('conv_2_kernel')}")
print(f"The optimal number of units in the first densely-connected layer is {best_hps.get('dense_1_units')}")
print(f"The optimal number of units in the second densely-connected layer is {best_hps.get('dense_2_units')}")


# 'conv_1_filter':
# 48
# 'conv_1_kernel':
# 5
# 'conv_2_filter':
# 32
# 'conv_2_kernel':
# 5
# 'dense_1_units':
# 32
# 'learning_rate':
# 0.01
# 'dropout':
# 0.0
# 'dense_2_units':
# 2



# Resnet
#
# Step 1: Import necessary libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Add, Activation
from tensorflow.keras.optimizers import Adam

# Step 2: Prepare your data
# Assuming all_data_train_2D and all_data_test_2D are your training and testing data respectively
X_train = all_data_train_2D.reshape((all_data_train_2D.shape[0], all_data_train_2D.shape[1], 1))
X_test = all_data_test_2D.reshape((all_data_test_2D.shape[0], all_data_test_2D.shape[1], 1))

# Prepare your labels
# Assuming temperatures_train and temperatures_test are your training and testing labels respectively
y_train = np.array(temperatures)
y_test = np.array(temperatures_cycle2)

# Step 3: Define your model
inputs = Input(shape=(X_train.shape[1], 1))

# Define a function for a ResNet block
def resnet_block(inputs, filters, kernel_size, strides):
    shortcut = Conv1D(filters, kernel_size=1, strides=strides)(inputs)
    
    x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    x = Activation('relu')(x)
    
    x = Conv1D(filters, kernel_size=kernel_size, padding='same')(x)
    
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    
    return x



# Add ResNet blocks
# x = resnet_block(inputs, filters=128, kernel_size=4, strides=1)
# x = resnet_block(x, filters=64, kernel_size=2, strides=1)

x = Flatten()(inputs)
x = Dense(50, activation='relu')(x)
# x = Dropout(0.5)(x)
x = x + Dense(50, activation='relu')(x)
x = x + Dense(50, activation='relu')(x)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile your model with a custom learning rate
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='mse')

# Step 4: Train your model
model.fit(X_train, y_train, epochs=1000, verbose=1)

# Step 5: Evaluate your model
loss = model.evaluate(X_test, y_test, verbose=0)

# Step 6: Predict temperatures for cycle 2
y_pred = model.predict(X_test)