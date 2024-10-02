import os  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
import tensorflow as tf
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.cm as cm
import pandas as pd
from utils import preprocess_data
from itertools import groupby
from scipy.stats import poisson
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import stats
import scipy.optimize as opt
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error,root_mean_squared_error
from tensorflow.keras.models import Sequential
from keras_tuner.tuners import RandomSearch
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Add
from tensorflow.keras.optimizers import Adam

from scipy.optimize import curve_fit




tf.get_logger().setLevel('ERROR')

folder_cycle1 = r'D:\Downloads\sensor1_esr_temp_cycle_1'
# folder_cycle2 =  r'D:\Downloads\sensor_1_ESR_cycle_2'
folder_cycle2= r'D:\Downloads\sensor_1_ESR_cycle_2-20240811T005415Z-001\sensor_1_ESR_cycle_2'


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

# a=tf.constant(-7.772773696087131e-05,dtype='float64') # (cycle 2)
# b=tf.constant(2.8706500246147373,dtype='float64') # cycle 2
c=tf.constant( -4.3478260869566193e-07,dtype='float64')
d=tf.constant(0.005185511627906974,dtype='float64')#Literature Value

b= tf.constant( 2.87068615576284,dtype='float64') # Grad search cycle 2 
a=tf.constant(-7.723607188481802e-05, dtype='float64') # Grad search cycle 2 
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


# Cycle 2 (Training Data)
train_data=[]
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
    train_data.append(avg_intensity_cycle2)
    




all_data_train_2D = np.array(train_data)

# Cycle 1 (Testing Data)



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

test_data=[]

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
    test_data.append(avg_intensity)
all_data_test_2D = np.array(test_data)




# # Prepare your data
X_train = all_data_train_2D.reshape((all_data_train_2D.shape[0], all_data_train_2D.shape[1], 1))
X_test = all_data_test_2D.reshape((all_data_test_2D.shape[0], all_data_test_2D.shape[1], 1))

# # Prepare your labels
y_train = np.array(temperatures_cycle2)
y_test = np.array(temperatures)


# #  #Step 3: Define your model
model = Sequential() # Linear stack of layers

model.add(Flatten(input_shape=(101, 1)))
model.add(Dense(48, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1)) # output layer of the model.



# # #  #Step 3: Define your model
# # model = Sequential() # Linear stack of layers
# model.add(Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=(X_train.shape[1], 1)))
# #model.add(Conv1D(filters=48, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))
# model.add(Conv1D(filters=80, kernel_size=10, activation='relu', input_shape=(X_train.shape[1], 1)))
# model.add(MaxPooling1D(pool_size=2)) # reduce the spatial dimensions of the data, preventing overfitting
# #model.add(Conv1D(filters=16, kernel_size=10, activation='relu'))
# #model.add(Conv1D(filters=32, kernel_size=5, activation='relu'))
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
# model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
# model.add(MaxPooling1D(pool_size=2)) 
# model.add(Flatten()) # convert the multi-dimensional output of the previous layer into a one-dimensional vector
# #model.add(Dense(80, activation='relu')) 
# #model.add(Dense(32, activation='relu')) 

# model.add(Dense(48, activation='relu')) 
# model.add(Dropout(0.0)) # For regularization
# model.add(Dense(1)) # output layer of the model.


opt = Adam(learning_rate=0.001)


#opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='mse')

# Step 4: Train your model
#model.fit(X_train, y_train, epochs=200, verbose=0)
# model.fit(X_train, y_train,validation_split=0.2, batch_size=5, epochs=200, verbose=1)
model.fit(X_train, y_train,validation_split=0.0, batch_size=10, epochs=200, verbose=1)

 #trains the neural network on  training data and then evaluates its performance on the test data, returning the loss value.
# Evaluate your model
loss = model.evaluate(X_test, y_test, verbose=0)

# Predict temperatures for cycle 2
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(temperatures, y_pred)) # 3.4

y_train_pred = model.predict(X_train)
# y_train_pred = y_train_pred.flatten()

# Calculate training error (RMSE)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

print(f'Training RMSE: {train_rmse}') # 0.15




y_train = np.array(temperatures_cycle2) + 273.15
y_test = np.array(temperatures) + 273.15




## Testing
y_pred = model.predict(X_test) + 273.15
y_pred = y_pred.flatten()
coefficients = np.polyfit(y_test, y_pred, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(y_test), max(y_test), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(y_test)
# Your predicted values
predicted = y_pred

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = root_mean_squared_error(true_poly, predicted)
plt.figure(figsize=(6, 4))
# Plot the data points
plt.plot(y_test, predicted, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)
plt.text(min(y_test), max(y_test)-10 , 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=1.0))
plt.tight_layout()
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Linear_fit_NN_testing_cycle2_cycle1.png', dpi=300) 
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()


# Training

y_train_pred = model.predict(X_train) +273.15
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
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
plt.text(min(y_train), max(predicted)-10 , 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.9))
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Linear_fit_NN_training_cycle2_cycle1.png', dpi=300) 
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()


# # # QUADRATIC FIT for training


y_train_pred = model.predict(X_train) +273.15
y_train_pred = y_train_pred.flatten()
coefficients = np.polyfit(y_train,y_train_pred , 2)
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
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
plt.text(min(y_train), max(predicted)-10 , 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.9))
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Quad_fit_NN_training_cycle2_cycle1.png', dpi=300) 
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()



# # # Quadratic Fit for testing




y_pred = model.predict(X_test) + 273.15
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
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Quad_fit_NN_testing_cycle2_cycle1.png', dpi=300) 
plt.text(min(y_test), max(predicted) -10, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()



import matplotlib.pyplot as plt

# Create a figure
plt.figure(figsize=(6, 4))


y_pred = model.predict(X_test) + 273.15
y_pred = y_pred.flatten()
# Fit a linear polynomial
coefficients1 = np.polyfit(y_test, y_pred, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(y_test), max(y_test), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(y_test)
predicted = y_pred
r_squared_poly1 = r2_score(y_test, predicted)
rmse_poly1 = root_mean_squared_error(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(y_test, predicted, 'o', color='green')
plt.plot(x_values, y_values1, '-', color='red')


# # Fit a quadratic polynomial
# coefficients2 = np.polyfit(y_test, y_pred, 2)
# polynomial2 = np.poly1d(coefficients2)
# coefficient = coefficients2[0]
# y_values2 = polynomial2(x_values)
# true_poly2 = polynomial2(y_test)
# predicted =y_pred
# r_squared_poly2 = r2_score(true_poly2, predicted)
# rmse_poly2 = root_mean_squared_error(true_poly2, predicted)

# # Plot the quadratic fit in blue with dashed line
# plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit'])

# Add text boxes for the linear and quadratic fits
# plt.text(min(y_test), max(predicted) * 0.8, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly1, slope, rmse_poly1), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(max(y_test)*0.39, max(predicted)*0.7, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly2, coefficient, rmse_poly2), color='blue', bbox=dict(facecolor='white', alpha=0.7))
plt.text(min(y_test), max(predicted) -20, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(max(y_test)-15, max(predicted)-30, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\NN\\combined_fit_testing_cycle_1_Linear.png', dpi=300) 

#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_NN_testing.png', dpi=300) 

# Show the plot
plt.show()


# Linear Fit for testing


# Create a figure
plt.figure(figsize=(6, 4))


y_pred = model.predict(X_test) + 273.15
y_pred = y_pred.flatten()
# Fit a linear polynomial
coefficients1 = np.polyfit(y_test, y_pred, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(y_test), max(y_test), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(y_test)
predicted = y_pred
r_squared_poly1 = r2_score(y_test, predicted)
rmse_poly1 = root_mean_squared_error(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(y_test, predicted, 'o', color='green')
plt.plot(x_values, y_values1, '-', color='red')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit'])

# Add text boxes for the linear and quadratic fits
# plt.text(min(y_test), max(predicted) * 0.8, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly1, slope, rmse_poly1), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(max(y_test)*0.39, max(predicted)*0.7, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly2, coefficient, rmse_poly2), color='blue', bbox=dict(facecolor='white', alpha=0.7))
plt.text(min(y_test), max(predicted) -20, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_NN_linear_testing.png', dpi=300) 

# Show the plot
plt.show()



# Training






#  #
# Create a figure
plt.figure(figsize=(6, 4))
y_pred = model.predict(X_train) + 273.15
y_pred = y_pred.flatten()
# Fit a linear polynomial
coefficients1 = np.polyfit(y_train, y_pred, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(y_test), max(y_test), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(y_test)
predicted = y_pred
r_squared_poly1 = r2_score(y_test, predicted)
rmse_poly1 = root_mean_squared_error(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(y_test, predicted, 'o', color='green')
plt.plot(x_values, y_values1, '-', color='red')

# # Fit a quadratic polynomial
# coefficients2 = np.polyfit(y_test, y_pred, 2)
# polynomial2 = np.poly1d(coefficients2)
# coefficient = coefficients2[0]
# y_values2 = polynomial2(x_values)
# true_poly2 = polynomial2(y_test)
# predicted =y_pred
# r_squared_poly2 = r2_score(true_poly2, predicted)
# rmse_poly2 = root_mean_squared_error(true_poly2, predicted)

# # Plot the quadratic fit in blue with dashed line
# plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit'])

# Add text boxes for the linear and quadratic fits
# plt.text(min(y_test), max(predicted) * 0.8, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly1, slope, rmse_poly1), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(max(y_test)*0.39, max(predicted)*0.7, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly2, coefficient, rmse_poly2), color='blue', bbox=dict(facecolor='white', alpha=0.7))
plt.text(min(y_test), max(predicted) -20, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(max(y_test)-15, max(predicted)-30, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_NN_testing.png', dpi=300) 

# Show the plot
plt.show()

# Training For Linear Fit
plt.figure(figsize=(6, 4))
y_pred = model.predict(X_test) + 273.15
y_pred = y_pred.flatten()
# Fit a linear polynomial
coefficients1 = np.polyfit(y_test, y_pred, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(y_test), max(y_test), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(y_test)
predicted = y_pred
r_squared_poly1 = r2_score(y_test, predicted)
rmse_poly1 = root_mean_squared_error(true_poly1, predicted)

# Plot the linear fit in red
plt.plot(y_test, predicted, 'o', color='green')
plt.plot(x_values, y_values1, '-', color='red')

# Fit a quadratic polynomial
coefficients2 = np.polyfit(y_test, y_pred, 2)
polynomial2 = np.poly1d(coefficients2)
coefficient = coefficients2[0]
y_values2 = polynomial2(x_values)
true_poly2 = polynomial2(y_test)
predicted =y_pred
r_squared_poly2 = r2_score(true_poly2, predicted)
rmse_poly2 = root_mean_squared_error(true_poly2, predicted)

# Plot the quadratic fit in blue with dashed line
plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit'])

# Add text boxes for the linear and quadratic fits
# plt.text(min(y_test), max(predicted) * 0.8, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly1, slope, rmse_poly1), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(max(y_test)*0.39, max(predicted)*0.7, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly2, coefficient, rmse_poly2), color='blue', bbox=dict(facecolor='white', alpha=0.7))
plt.text(min(y_test), max(predicted) -20, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(max(y_test)-15, max(predicted)-30, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# Save the figure
#plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\NN\\combined_fit_training_cycle_2_MLE.png', dpi=300) 

# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\combined_fit_NN_testing.png', dpi=300) 

# Show the plot
plt.show()



# Create a figure
plt.figure(figsize=(6, 4))

y_train_pred = model.predict(X_train) +273.15
y_train_pred = y_train_pred.flatten()
# Fit a linear polynomial
coefficients1 = np.polyfit(y_train, y_train_pred, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(y_train), max(y_train), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(y_train)
predicted = y_train_pred
r_squared_poly1 = r2_score(y_train, predicted)
rmse_poly1 = mean_squared_error(true_poly1, predicted, squared=False)

# Plot the linear fit in red
plt.plot(y_train, predicted, 'o', color='green')
plt.plot(x_values, y_values1, '-', color='red')

# # Fit a quadratic polynomial
# coefficients2 = np.polyfit(y_train, y_train_pred, 2)
# polynomial2 = np.poly1d(coefficients2)
# coefficient = coefficients2[0]
# y_values2 = polynomial2(x_values)
# true_poly2 = polynomial2(y_train)
# predicted =y_train_pred 
# r_squared_poly2 = r2_score(true_poly2, predicted)
# rmse_poly2 = mean_squared_error(true_poly2, predicted, squared=False)

# # Plot the quadratic fit in blue with dashed line
# plt.plot(x_values, y_values2, '--', color='blue')

# Set the title and labels
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit'])


plt.text(min(y_train), max(predicted) -40, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(max(y_train)-30, max(predicted)-55, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}'.format(r_squared_poly2, coefficient), color='blue', bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# # Add text boxes for the linear and quadratic fits
# plt.text(min(y_train), max(y_train_pred)- 20, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly1, slope, rmse_poly1), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(max(y_train)-15, max(y_train_pred)-30, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly2, coefficient, rmse_poly2), color='blue', bbox=dict(facecolor='white', alpha=0.7))

# Adjust the layout to prevent overlap

# Save the figure
plt.savefig('D:\\Downloads\\ODMR Paper Figs\\Paper Figs\\cycle2\\NN\\combined_fit_training_cycle_2_linear.png', dpi=300) 


# Show the plot
plt.show()


# Training Linear Fit

# Create a figure
plt.figure(figsize=(6, 4))

y_train_pred = model.predict(X_train) +273.15
y_train_pred = y_train_pred.flatten()
# Fit a linear polynomial
coefficients1 = np.polyfit(y_train, y_train_pred, 1)
polynomial1 = np.poly1d(coefficients1)
slope = coefficients1[0]
x_values = np.linspace(min(y_train), max(y_train), 100)
y_values1 = polynomial1(x_values)
true_poly1 = polynomial1(y_train)
predicted = y_train_pred
r_squared_poly1 = r2_score(y_train, predicted)
rmse_poly1 = mean_squared_error(true_poly1, predicted, squared=False)

# Plot the linear fit in red
plt.plot(y_train, predicted, 'o', color='green')
plt.plot(x_values, y_values1, '-', color='red')

#
# Set the title and labels
plt.xticks(fontsize=12)  # Set the font size of x values
plt.yticks(fontsize=12)
plt.xlabel('Measured Temperatures (K)', fontsize=12)
plt.ylabel('Predicted Temperatures (K)', fontsize=12)

# Add a legend
plt.legend(['Data', 'Linear Fit'])


plt.text(min(y_train), max(predicted) -40, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}'.format(r_squared_poly1, slope), color='red', bbox=dict(facecolor='white', alpha=0.7))
# Adjust the layout to prevent overlap
plt.tight_layout()

# # Add text boxes for the linear and quadratic fits
# plt.text(min(y_train), max(y_train_pred)- 20, 'Linear Fit:\nR-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly1, slope, rmse_poly1), color='red', bbox=dict(facecolor='white', alpha=0.7))
# plt.text(max(y_train)-15, max(y_train_pred)-30, 'Quadratic Fit:\nR-squared = {:.3f}\nCoefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly2, coefficient, rmse_poly2), color='blue', bbox=dict(facecolor='white', alpha=0.7))

# Adjust the layout to prevent overlap

# Save the figure
# plt.savefig('D:\\Downloads\\ODMR Paper Figs\\NN_training_linear.png', dpi=300) 

# Show the plot
plt.show()




 #This process is essentially performing 10 runs of the model training and evaluation process and then reporting the average performance.

n_runs = 10


# Arrays to store results
train_rmses = np.zeros(n_runs)
test_rmses = np.zeros(n_runs)

for i in range(n_runs):
    # Define your model
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], 1)))
    #model.add(Conv1D(filters=48, kernel_size=5, activation='relu')) #constant
    model.add(Conv1D(filters=80, kernel_size=10, activation='relu')) 
   #model.add(Conv1D(filters=99, kernel_size=10, activation='relu'))
    #model.add(Conv1D(filters=96, kernel_size=10, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    #model.add(Conv1D(filters=32, kernel_size=5, activation='relu')) # constant
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    #model.add(Conv1D(filters=46, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Flatten())
    #model.add(Dense(32, activation='relu')) # constant
    model.add(Dense(48, activation='relu')) 
    #model.add(Dense(56, activation='relu'))
    #model.add(Dense(64, activation='relu'))  
    model.add(Dropout(0.0))# constant
    # model.add(Dropout(0.1))
    model.add(Dense(1))

    opt = Adam(learning_rate=0.001)
    #opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mse')

    # Train your model
    model.fit(X_train, y_train, validation_split=0.2, batch_size=5, epochs=200, verbose=0)

    # Predict temperatures for training and test data
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    # Calculate training and test error (RMSE)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Store results
    train_rmses[i] = train_rmse
    test_rmses[i] = test_rmse

# Calculate average training and test RMSE
avg_train_rmse = np.mean(train_rmses)
avg_test_rmse = np.mean(test_rmses)

print(f'Average training RMSE: {avg_train_rmse}') # 1.9696363746996635 
# with seocond model 1.56
print(f'Average test RMSE: {avg_test_rmse}') # 2.1337364085265564
# 1.82

# Calculate minimum training and test RMSE
min_train_rmse = np.min(train_rmses)
min_test_rmse = np.min(test_rmses)

print(f'Minimum training RMSE: {min_train_rmse}') 
print(f'Minimum test RMSE: {min_test_rmse}') 





from tensorflow import keras

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

tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=8,# tuner will try 5 different hyperparameter configurations
    executions_per_trial=2,# combination of hyperparameters will be tried 3 times
    directory='_ddddddirectory',
    project_name='_pppppproject_name' )

tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    
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




# Plotting the results
val_losses = []
conv_1_filters = []
conv_1_kernels = []
conv_2_filters = []
conv_2_kernels = []
dense_1_units = []
dropouts = []
learning_rates = []

for trial in tuner.oracle.get_best_trials(num_trials=8):
    val_losses.append(trial.metrics.get_best_value('val_loss'))
    conv_1_filters.append(trial.hyperparameters.get('conv_1_filter'))
    conv_1_kernels.append(trial.hyperparameters.get('conv_1_kernel'))
    conv_2_filters.append(trial.hyperparameters.get('conv_2_filter'))
    conv_2_kernels.append(trial.hyperparameters.get('conv_2_kernel'))
    dense_1_units.append(trial.hyperparameters.get('dense_1_units'))
    dropouts.append(trial.hyperparameters.get('dropout'))
    learning_rates.append(trial.hyperparameters.get('learning_rate'))

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(conv_1_filters, val_losses, 'o-')
plt.xlabel('conv_1_filters')
plt.ylabel('val_loss')

plt.subplot(2, 2, 2)
plt.plot(conv_1_kernels, val_losses, 'o-')
plt.xlabel('conv_1_kernels')
plt.ylabel('val_loss')

plt.subplot(2, 2, 3)
plt.plot(conv_2_filters, val_losses, 'o-')
plt.xlabel('conv_2_filters')
plt.ylabel('val_loss')

plt.subplot(2, 2, 4)
plt.plot(conv_2_kernels, val_losses, 'o-')
plt.xlabel('conv_2_kernels')
plt.ylabel('val_loss')

plt.tight_layout()
plt.show()


# conv_1_filter':
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


# # Get all trials
# all_trials = tuner.oracle.trials

# # Iterate over all trials
# for trial in all_trials.values():
#     # Get the hyperparameters for this trial
#     trial_hps = trial.hyperparameters.values

#     # Print the hyperparameters
#     print(trial_hps)


# # Testing
y_pred = y_pred.flatten()
coefficients = np.polyfit(temperatures, y_pred, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures), max(temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures)
# Your predicted values
predicted = y_pred

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = mean_squared_error(true_poly, predicted, squared=False)
# Plot the data points
plt.plot(temperatures, predicted, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')

plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()

# # # For Training
coefficients = np.polyfit(temperatures_cycle2,predicted_train_temperatures, 1)
polynomial = np.poly1d(coefficients)
slope = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures_cycle2), max(temperatures_cycle2), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures_cycle2)
# Your predicted values
predicted =predicted_train_temperatures
r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = mean_squared_error(true_poly, predicted, squared=False)
# Plot the data points
plt.plot(temperatures_cycle2, predicted_train_temperatures, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.title('True vs Estimated for Sensor I')
plt.text(min(temperatures_cycle2), max(predicted) * 0.9, 'R-squared = {:.3f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
#plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.2f}\nSlope = {:.2f}\nRMSE = {:.2f}'.format(r_squared_poly, slope, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()



# # # QUADRATIC FIT for training
coefficients = np.polyfit(temperatures_cycle2,  predicted_train_temperatures, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures_cycle2), max(temperatures_cycle2), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures_cycle2)
# Your predicted values
predicted =predicted_train_temperatures

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = mean_squared_error(true_poly, predicted, squared=False)
# Plot the data points
plt.plot(temperatures_cycle2, predicted_train_temperatures, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.title('True vs Estimated for Sensor I ')
plt.text(min(temperatures_cycle2), max(predicted) * 0.9, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()

# # # Quadratic Fit for testing
coefficients = np.polyfit(temperatures, predicted_temperatures, 2)
polynomial = np.poly1d(coefficients)
first_coefficient = coefficients[0]

# Generate x values
x_values = np.linspace(min(temperatures), max(temperatures), 100)

# Generate y values
y_values = polynomial(x_values)

true_poly= polynomial(temperatures)
# Your predicted values
predicted =predicted_temperatures

r_squared_poly = r2_score(true_poly, predicted)
# Calculate RMSE
rmse_poly = mean_squared_error(true_poly, predicted, squared=False)
# Plot the data points
plt.plot(temperatures, predicted_temperatures, 'o')
plt.plot(x_values, y_values, '-')
plt.xlabel('True Temperatures')
plt.ylabel('Estimated Temperatures')
plt.title('True vs Estimated for Sensor I ')
plt.text(min(temperatures), max(predicted) * 0.9, 'R-squared = {:.3f}\nFirst Coefficient = {:.2e}\nRMSE = {:.2f}'.format(r_squared_poly, first_coefficient, rmse_poly), color='red', bbox=dict(facecolor='white', alpha=0.7))
plt.show()


# Step 1: Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 2: Prepare your data
# Assuming all_data_train_2D and all_data_test_2D are your training and testing data respectively
X_train = all_data_train_2D.reshape((all_data_train_2D.shape[0], all_data_train_2D.shape[1], 1))
X_test = all_data_test_2D.reshape((all_data_test_2D.shape[0], all_data_test_2D.shape[1], 1))

# Conv1D expects the input shape to be (batch_size, steps, input_dim)
# Prepare your labels
# Assuming temperatures_train and temperatures_test are your training and testing labels respectively
y_train = np.array(temperatures_cycle2)
y_test = np.array(temperatures)

# Step 3: Define your model
model = Sequential() # Linear stack of layers
model.add(Conv1D(filters=128, kernel_size=5, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# The Conv1D layers are used to convolve the 1-D input data with a kernel, which helps to learn patterns in the data
# e MaxPooling1D layers are used to downsample the input along its temporal dimension (time or sequence), and the Dense layers are fully connected layers.
# Compile your model with a custom learning rate
opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='mse')

# Step 4: Train your model
model.fit(X_train, y_train, epochs=200, verbose=0)

# epochs: The number of times to iterate over the entire dataset
# verbose: 0 for no output, 1 for progress bar logging, 2 for one log line per epoch

# Step 5: Evaluate your model
loss = model.evaluate(X_test, y_test, verbose=0)
# Trained models are evaluated on the testing data

# Step 6: Predict temperatures for cycle 2
y_pred = model.predict(X_test)



from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, add
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

# First block
conv1 = Conv1D(filters=128, kernel_size=5, activation='relu')(inputs)
pool1 = MaxPooling1D(pool_size=2)(conv1)
conv2 = Conv1D(filters=32, kernel_size=5, activation='relu')(pool1)
pool2 = MaxPooling1D(pool_size=2)(conv2)

# Residual connection
# Residual connection
residual = MaxPooling1D(pool_size=4)(inputs)

shortcut = add([pool2, residual])

flatten = Flatten()(shortcut)
dense1 = Dense(100, activation='relu')(flatten)
dropout = Dropout(0.5)(dense1)
dense2 = Dense(50, activation='relu')(dropout)
outputs = Dense(1)(dense2)

model = Model(inputs=inputs, outputs=outputs)

# Compile your model with a custom learning rate
opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss='mse')

# Step 4: Train your model
model.fit(X_train, y_train, epochs=200, verbose=0)

# Step 5: Evaluate your model
loss = model.evaluate(X_test, y_test, verbose=0)

# Step 6: Predict temperatures for cycle 2
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error

# Predict temperatures for cycle 2
y_pred = model.predict(X_test)

# Calculate the RMSE
from sklearn.metrics import mean_squared_error


# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(temperatures, y_pred))

print(f'RMSE: {rmse}')



# he Sequential model is a type of model provided by Keras that is appropriate for a plain stack of layers
# where each layer has exactly one input tensor and one output tensor. 
# it allows you to build a model layer-by-layer in a sequence, where each layer is added on top of the previous one.
# Ours neural network is a linear stack of layers. 
# We start with a 1D convolutional layer, followed by a max pooling layer, 
# 
# another 1D convolutional layer, another max pooling layer, a flattening layer, and 
# then two fully connected (dense) layers. 
# Each of these layers feeds into the next one in a sequence



# filters: This is the number of output filters in the convolution. 
# In other words, it's the number of features the Conv1D layer will learn from the input data. 


#kernel_size: This is the length of the 1D convolution window. 
# Here first Conv1D layer will consider 4 data points at a time when learning features, and the
# second one will consider 2 data points at a time.


# The 'relu' activation function 
# helps to mitigate the vanishing gradients problem, which can occur when training deep neural networks.

# Dropout layer randomly sets a fraction of input units to 
# 0 at each update during training time, which helps prevent overfitting. 


#
# pool_size parameter in the MaxPooling1D layer 
# refers to the size of the window over which to take the maximum value.
#used to reduce the spatial dimensions of the data while retaining the most important information

# ool_size=2 for both MaxPooling1D layers, meaning that each of these layers will reduce the size of its input by half. This helps to reduce the computational complexity of the model and can also help to prevent overfitting 
# by providing a form of data abstraction.


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