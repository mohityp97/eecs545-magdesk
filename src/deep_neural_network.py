## Importing all libraries

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from sklearn.model_selection import LeaveOneOut,KFold
from matplotlib import pyplot as plt
import matplotlib


## Data Pre-Processing

## Read the Data

df1 = pd.read_csv(r'magdesk_dataset_SA_2022_04_12-09_03_PM_sector2.csv', header = None)
nf1 = pd.read_csv(r'magdesk_dataset_background_noise_SA_2022_04_12-09_03_PM_sector2.csv', header = None)
print(df1.shape)
print(nf1.shape)

df1 = df1[1:]

df1Float = df1.astype(float).to_numpy()


inputData = df1Float[:, 7:]

outputData = df1Float[:, 1:7]
noise = nf1.astype(float).to_numpy()[:, 1:]



## Calculate the Z-scores

colMean = np.reshape(np.mean(noise, axis = 0), (1, -1))
colStd = np.reshape(np.std(noise, axis = 0), (1, -1))
normalizedData = (inputData - colMean)/colStd
normalizedData[normalizedData<1] = 0


## Preprocessing Outputs - Degrees to Quaternions

outputXyz = outputData[:, :3]
outputRyp = outputData[:, 3:6]

yaw = np.reshape(outputData[:, 3], (-1,1))
pitch = np.reshape(outputData[:, 4], (-1,1))
roll = np.reshape(outputData[:, 5], (-1,1))
qx = np.multiply(np.multiply(np.sin(roll/2), np.cos(pitch/2)), np.cos(yaw/2)) - np.multiply(np.multiply(np.cos(roll/2), np.sin(pitch/2)), np.sin(yaw/2))
qy = np.multiply(np.multiply(np.cos(roll/2), np.sin(pitch/2)), np.cos(yaw/2)) + np.multiply(np.multiply(np.sin(roll/2), np.cos(pitch/2)), np.sin(yaw/2))
qz = np.multiply(np.multiply(np.cos(roll/2), np.cos(pitch/2)), np.sin(yaw/2)) - np.multiply(np.multiply(np.sin(roll/2), np.sin(pitch/2)), np.cos(yaw/2))
qw = np.multiply(np.multiply(np.cos(roll/2), np.cos(pitch/2)), np.cos(yaw/2))

outputData = np.hstack((outputXyz, qx, qy, qz, qw))


## Extracting Subarrays


line_distance = 10
pSensor_7_line_elevated_z_1cm_below_table = np.array([
    [75,( 6*line_distance) + 17.5,-2.5],
    [65,( 6*line_distance) + 17.5,-2.5],
    [55,( 6*line_distance) + 17.5,-2.5],
    [45,( 6*line_distance) + 17.5,-2.5],
    [35,( 6*line_distance) + 17.5,-2.5],
    [25,( 6*line_distance) + 17.5,-2.5],
    [15,( 6*line_distance) + 17.5,-2.5],
    [5,( 6*line_distance) + 17.5,-2.5],
    [-5,(  6*line_distance) + 17.5,-2.5],
    [-15,( 6*line_distance) + 17.5,-2.5],
    [-25,( 6*line_distance) + 17.5,-2.5],
    [-35,( 6*line_distance) + 17.5,-2.5],
    [-45,( 6*line_distance) + 17.5,-2.5],
    [-55,( 6*line_distance) + 17.5,-2.5],
    [-65,( 6*line_distance) + 17.5,-2.5],
    [-75,( 6*line_distance) + 17.5,-2.5],
    [75,( 5*line_distance) + 17.5,-3.5],
    [65,( 5*line_distance) + 17.5,-3.5],
    [55,( 5*line_distance) + 17.5,-3.5],
    [45,( 5*line_distance) + 17.5,-3.5],
    [35,( 5*line_distance) + 17.5,-3.5],
    [25,( 5*line_distance) + 17.5,-3.5],
    [15,( 5*line_distance) + 17.5,-3.5],
    [5,( 5*line_distance) + 17.5,-3.5],
    [-5,(  5*line_distance) + 17.5,-3.5],
    [-15,( 5*line_distance) + 17.5,-3.5],
    [-25,( 5*line_distance) + 17.5,-3.5],
    [-35,( 5*line_distance) + 17.5,-3.5],
    [-45,( 5*line_distance) + 17.5,-3.5],
    [-55,( 5*line_distance) + 17.5,-3.5],
    [-65,( 5*line_distance) + 17.5,-3.5],
    [-75,( 5*line_distance) + 17.5,-3.5],
    [75,( 4*line_distance) + 17.5,-2.5],
    [65,( 4*line_distance) + 17.5,-2.5],
    [55,( 4*line_distance) + 17.5,-2.5],
    [45,( 4*line_distance) + 17.5,-2.5],
    [35,( 4*line_distance) + 17.5,-2.5],
    [25,( 4*line_distance) + 17.5,-2.5],
    [15,( 4*line_distance) + 17.5,-2.5],
    [5,( 4*line_distance) + 17.5,-2.5],
    [-5,(  4*line_distance) + 17.5,-2.5],
    [-15,( 4*line_distance) + 17.5,-2.5],
    [-25,( 4*line_distance) + 17.5,-2.5],
    [-35,( 4*line_distance) + 17.5,-2.5],
    [-45,( 4*line_distance) + 17.5,-2.5],
    [-55,( 4*line_distance) + 17.5,-2.5],
    [-65,( 4*line_distance) + 17.5,-2.5],
    [-75,( 4*line_distance) + 17.5,-2.5],
    [75,( 3*line_distance) + 17.5,-3.5],
    [65,( 3*line_distance) + 17.5,-3.5],
    [55,( 3*line_distance) + 17.5,-3.5],
    [45,( 3*line_distance) + 17.5,-3.5],
    [35,( 3*line_distance) + 17.5,-3.5],
    [25,( 3*line_distance) + 17.5,-3.5],
    [15,( 3*line_distance) + 17.5,-3.5],
    [5,( 3*line_distance) + 17.5,-3.5],
    [-5,(  3*line_distance) + 17.5,-3.5],
    [-15,( 3*line_distance) + 17.5,-3.5],
    [-25,( 3*line_distance) + 17.5,-3.5],
    [-35,( 3*line_distance) + 17.5,-3.5],
    [-45,( 3*line_distance) + 17.5,-3.5],
    [-55,( 3*line_distance) + 17.5,-3.5],
    [-65,( 3*line_distance) + 17.5,-3.5],
    [-75,( 3*line_distance) + 17.5,-3.5],
    [75,( 2*line_distance) + 17.5,-2.5],
    [65,( 2*line_distance) + 17.5,-2.5],
    [55,( 2*line_distance) + 17.5,-2.5],
    [45,( 2*line_distance) + 17.5,-2.5],
    [35,( 2*line_distance) + 17.5,-2.5],
    [25,( 2*line_distance) + 17.5,-2.5],
    [15,( 2*line_distance) + 17.5,-2.5],
    [5,( 2*line_distance) + 17.5,-2.5],
    [-5,(  2*line_distance) + 17.5,-2.5],
    [-15,( 2*line_distance) + 17.5,-2.5],
    [-25,( 2*line_distance) + 17.5,-2.5],
    [-35,( 2*line_distance) + 17.5,-2.5],
    [-45,( 2*line_distance) + 17.5,-2.5],
    [-55,( 2*line_distance) + 17.5,-2.5],
    [-65,( 2*line_distance) + 17.5,-2.5],
    [-75,( 2*line_distance) + 17.5,-2.5],
    [75,( line_distance) + 17.5,-3.5],
    [65,( line_distance) + 17.5,-3.5],
    [55,( line_distance) + 17.5,-3.5],
    [45,( line_distance) + 17.5,-3.5],
    [35,( line_distance) + 17.5,-3.5],
    [25,( line_distance) + 17.5,-3.5],
    [15,( line_distance) + 17.5,-3.5],
    [5,( line_distance) + 17.5,-3.5],
    [-5,(  line_distance) + 17.5,-3.5],
    [-15,( line_distance) + 17.5,-3.5],
    [-25,( line_distance) + 17.5,-3.5],
    [-35,( line_distance) + 17.5,-3.5],
    [-45,( line_distance) + 17.5,-3.5],
    [-55,( line_distance) + 17.5,-3.5],
    [-65,( line_distance) + 17.5,-3.5],
    [-75,( line_distance) + 17.5,-3.5],
    [75, 17.5, -2.5],
    [65, 17.5, -2.5],
    [55, 17.5, -2.5],
    [45, 17.5, -2.5],
    [35, 17.5, -2.5],
    [25, 17.5, -2.5],
    [15, 17.5, -2.5],
    [5, 17.5, -2.5],
    [-5, 17.5, -2.5],
    [-15, 17.5, -2.5],
    [-25, 17.5, -2.5],
    [-35, 17.5, -2.5],
    [-45, 17.5, -2.5],
    [-55, 17.5, -2.5],
    [-65, 17.5, -2.5],
    [-75, 17.5, -2.5]
])

X_combined = np.empty((0,75))
y_combined = np.empty((0,7))
timestamps = ['2022_02_16-07_17_PM']
for timestamp_of_file in timestamps:  
    #filename = './magdesk_dataset_background_noise_SA_' + timestamp_of_file + '.csv'
    #csv = pd.read_csv(filename, delimiter=',', header=1)
    #background_data = csv.values[1:,:]
    #filename = './magdesk_dataset_SA_' + timestamp_of_file + '.csv'
    #csv = pd.read_csv(filename, delimiter=',', header=1)
    #data = csv.values[1:,:]
    
    #y = data[:,1:7]
    #X = data[:, 7:]
    #background_X = background_data[:,1:]
    #mean_background = np.mean(background_X, axis=0)
    #std_background = np.std(background_X, axis=0)
    # print(mean_background)
    # print(std_background)
    #X = X - mean_background
    #X = X / std_background
   
    #X[X < 1] = 0
    X = normalizedData
    y = outputData
    print(np.shape(X))
    print(np.shape(y))

    size_of_feature_matrix = 5

    X_reduced_features = np.zeros((len(X), ((size_of_feature_matrix**2) *3) ))
   
    sensor_pos = pSensor_7_line_elevated_z_1cm_below_table.reshape(7,16,3)

    for sample_iter in range(len(X)):
        X_physical_layout = np.reshape(X[sample_iter], (-1,16,3))
        summed_array = np.zeros((7-(size_of_feature_matrix - 1), 16-(size_of_feature_matrix - 1)))
        for row_iter in range(7-(size_of_feature_matrix - 1)):
            for col_iter in range(16-(size_of_feature_matrix - 1)):
                    summed_array[row_iter, col_iter] = np.sum(X_physical_layout[row_iter:row_iter+(size_of_feature_matrix + 1), col_iter:col_iter+(size_of_feature_matrix + 1), :], axis=None)
        max_index = np.unravel_index(np.argmax(np.abs(summed_array), axis=None), summed_array.shape)
        max_index = np.array(max_index)
        subarray = X_physical_layout[max_index[0] : max_index[0] + (size_of_feature_matrix), max_index[1] : max_index[1] + (size_of_feature_matrix), :]
        coordinates = max_index + ((size_of_feature_matrix - 1)/2)
        reduced_features = np.reshape(subarray,-1)
#         reduced_features = np.append(reduced_features, coordinates)
        X_reduced_features[sample_iter] = reduced_features
       
       
#         centre_of_mass = ndimage.measurements.center_of_mass(subarray)
#         print(subarray)
# #         print(centre_of_mass)
#         coordinates = np.array([round(centre_of_mass[0]), round(centre_of_mass[1])])
        coordinates = coordinates.astype(int)
#         print(coordinates)
        y[sample_iter, 0] = sensor_pos[coordinates[0], coordinates[1], 0] - y[sample_iter, 0]
        y[sample_iter, 1] = sensor_pos[coordinates[0], coordinates[1], 1] - y[sample_iter, 1]
       
#         print(X_physical_layout)
#         print(summed_array)
#         print(max_index)
#         print(y[sample_iter + 10000])

    print(np.shape(X_reduced_features))
#     print(X_reduced_features)

    X = X_reduced_features

    X = X[(y[:,2] < 35)]
    y = y[(y[:,2] < 35)]

    X_combined = np.append(X_combined, X, axis=0)
    y_combined = np.append(y_combined, y, axis=0)
    normalizedData = X_combined
    outputData = y_combined
    print(np.shape(normalizedData))
    print(np.shape(outputData))



## Output Data Analysis


plt.hist(outputData[:,0], bins = 100)
plt.show()
plt.hist(outputData[:,1], bins = 100)
plt.show()
plt.hist(outputData[:,2], bins = 100)
plt.show()
plt.hist(outputData[:, 3], bins = 100)
plt.show()
plt.hist(outputData[:, 4], bins = 100)
plt.show()
plt.hist(outputData[:, 5], bins = 100)
plt.show()
plt.hist(outputData[:, 6], bins = 100)
plt.show()


## Building the Tensorflow model

n_split = 5


testingrms = 0
trainingrms = 0
testingmin = 0
testingmax = 0
testingmean = 0
testingstd = 0
testingmedian = 0
for train_index, test_index in KFold(n_split, shuffle = True).split(normalizedData):
    print("****************************************************************************************************")
    x_train, x_test = normalizedData[train_index], normalizedData[test_index]
    y_train, y_test = (outputData)[train_index], (outputData)[test_index]
    
    ## x_train = np.swapaxes(np.reshape(x_train, (np.shape(x_train)[0], 3, 16, 7)), 1, 3)
    
    ## x_test = np.swapaxes(np.reshape(x_test, (np.shape(x_test)[0], 3, 16, 7)), 1, 3)
    
    x_train = np.swapaxes(np.reshape(x_train, (np.shape(x_train)[0], 3, 5, 5)), 1, 3)
    
    x_test = np.swapaxes(np.reshape(x_test, (np.shape(x_test)[0], 3, 5, 5)), 1, 3)
    
    ## Clear the previous Tensorflow session
    #print(np.shape(x_train))
    ## Start building the Tensorflow model
    #for act in ['sigmoid', 'tanh', 'relu', 'linear']:
    #    clear_session()
    #    model = Sequential()
    
    ## Input Layer
    ##model.add(layers.InputLayer(input_shape = np.shape(x_train)[1:]))
    #    print("*********************************** Activation is "+act+"*********************************")
    ## First Convolutional Layer
    #    model.add(layers.Conv2D(5, (2,2), padding = 'same', activation = 'tanh', input_shape = np.shape(x_train)[1:]))
    
    ## Max Pooling Layer
    #    model.add(layers.MaxPooling2D(2, 2))

    ## Second Convolutional Layer
    #    model.add(layers.Conv2D(10, (3, 3), padding = 'same', activation = act))
    
    ## Max Pooling Layer
    #    model.add(layers.MaxPooling2D(2, 2))
    
    #    model.add(layers.Flatten())
    
    
    #    model.add(layers.Dense(20, activation = 'relu'))
    
    #    model.add(layers.Dropout(0.1))
    
    #    model.add(layers.Dense(3, activation = 'linear'))
    
    #    model.summary()
    
    #    model.compile(Adam(learning_rate = 0.001), loss= 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
    #    model.fit(x_train, y_train, epochs = 10)
    
    #    print(model.evaluate(x_test, y_test))
    #    print("Testing RMS Error:")
    #    print(np.sqrt(np.sum(np.square(np.subtract(model.predict(x_test), y_test[:, :3])), axis = 0)/np.shape(y_test)[0]))
    #    print("Training RMS Error:")
    #    print(np.sqrt(np.sum(np.square(np.subtract(model.predict(x_train), y_train[:, :3])), axis = 0)/np.shape(y_train)[0]))
    #    print("Testing Min Error:")
    #    print(np.amin(np.abs(np.subtract(model.predict(x_test), y_test[:, :3])), axis = 0))
    #    print("Testing Max Error:")
    #    print(np.amax(np.abs(np.subtract(model.predict(x_test), y_test[:, :3])), axis = 0))
    #    print("Testing Mean Error:")
    #    print(np.mean(np.abs(np.subtract(model.predict(x_test), y_test[:, :3])), axis = 0))
    #    print("Testing Standard Deviation Error:")
    #    print(np.std(np.abs(np.subtract(model.predict(x_test), y_test[:, :3])), axis = 0))
    #    print("Testing Median Error:")
    #    print(np.median(np.abs(np.subtract(model.predict(x_test), y_test[:, :3])), axis = 0))
        

        
    print(np.shape(x_train))
    ## Start building the Tensorflow model
    for i in range(5):
        clear_session()
        model = Sequential()
    
    ## Input Layer
    ##model.add(layers.InputLayer(input_shape = np.shape(x_train)[1:]))
        print("*********************************** Number of filters: "+str(i)+"*********************************")
    ## First Convolutional Layer
        model.add(layers.Conv2D(25, (2,2), padding = 'same', activation = 'tanh', input_shape = np.shape(x_train)[1:]))
    
    ## Max Pooling Layer
        model.add(layers.MaxPooling2D(2, 2))

    ## Second Convolutional Layer
        model.add(layers.Conv2D(50, (2, 2), padding = 'same', activation = 'relu'))
    
    ## Max Pooling Layer
        model.add(layers.MaxPooling2D(2, 2))
        
        
    
        ##model.add(layers.Reshape((10, 5)))
    
        ##model.add(layers.LSTM(5, dropout = 0.1, recurrent_dropout = 0.1))
        
        #model.add(layers.Reshape((1, 3)))
    
        #model.add(layers.LSTM(3, dropout = 0.1, recurrent_dropout = 0.1))
        
        model.add(layers.Flatten())
        
        model.add(layers.Dense(20, activation = 'relu'))
    
        model.add(layers.Dropout(0.1))
        
        model.add(layers.Dense(10, activation = 'relu'))
    
        model.add(layers.Dropout(0.1))
    
        model.add(layers.Dense(7, activation = 'linear'))
    
        model.summary()
    
        model.compile(Adam(learning_rate = 0.001), loss= 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
    
        model.fit(x_train, y_train, epochs = 10)
    
        testingrms = testingrms + np.sqrt(np.sum(np.square(np.subtract(model.predict(x_test), y_test)), axis = 0)/np.shape(y_test)[0])
        print(testingrms)
        trainingrms = trainingrms + np.sqrt(np.sum(np.square(np.subtract(model.predict(x_train), y_train)), axis = 0)/np.shape(y_train)[0])
        print(trainingrms)
        testingmin = testingmin + np.amin(np.abs(np.subtract(model.predict(x_test), y_test)), axis = 0)
        print(testingmin)
        testingmax = testingmax + np.amax(np.abs(np.subtract(model.predict(x_test), y_test)), axis = 0)
        print(testingmax)
        testingmean = testingmean + np.mean(np.abs(np.subtract(model.predict(x_test), y_test)), axis = 0)
        print(testingmean)
        testingstd = testingstd + np.std(np.abs(np.subtract(model.predict(x_test), y_test)), axis = 0)
        print(testingstd)
        testingmedian = testingmedian + np.median(np.abs(np.subtract(model.predict(x_test), y_test)), axis = 0)
        print(testingmedian)
        break

        
print("Testing RMS Error:")
print(testingrms)
print("Training RMS Error:")
print(trainingrms)
print("Testing Min Error:")
print(testingmin)
print("Testing Max Error:")
print(testingmax)
print("Testing Mean Error:")
print(testingmean)
print("Testing Standard Deviation Error:")
print(testingstd)
print("Testing Median Error:")
print(testingmedian)


print("Testing RMS Error:")
print(testingrms/n_split)
print("Training RMS Error:")
print(trainingrms/n_split)
print("Testing Min Error:")
print(testingmin/n_split)
print("Testing Max Error:")
print(testingmax/n_split)
print("Testing Mean Error:")
print(testingmean/n_split)
print("Testing Standard Deviation Error:")
print(testingstd/n_split)
print("Testing Median Error:")
print(testingmedian/n_split)


import time

test_x = np.reshape(x_test[0], (1, 5, 5, 3))
begin = time.time()
print(model.predict(test_x))
end = time.time()

diff = end-begin

print(diff)

model.save("eecs545cnn")



