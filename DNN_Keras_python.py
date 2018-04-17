# -*- coding: utf-8 -*-
"""
Image Classification: Deep Nural Network 
@author: zua3
"""

# Use scikit-learn to grid search the batch size and epochs
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.regularizers import l2
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn import metrics

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
os.chdir('F:/My_GitHub/DNN_Keras_Python/Data') 
filename = 'point_data.csv'
dataset = pd.read_csv(filename)
list(dataset)

# dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
Class_ID = dataset[['Class_ID']]
X = dataset[['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']]

# Specify the target labels and flatten the array 
Y= np.ravel(Class_ID)

# Split the data up in train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Scale the train & test data set
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)

# Initialize the constructor

# define vars
input_num_units = 10
hidden1_num_units = 200
hidden2_num_units = 200
hidden3_num_units = 200
hidden4_num_units = 200
output_num_units = 5

# Define model
model = Sequential([
    Dense(output_dim=hidden1_num_units, input_dim=input_num_units, kernel_regularizer=l2(0.0001), activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden2_num_units, input_dim=hidden1_num_units, kernel_regularizer=l2(0.0001), activation='relu'),
    Dropout(0.2),
    Dense(output_dim=hidden3_num_units, input_dim=hidden2_num_units,  kernel_regularizer=l2(0.0001), activation='relu'),
    Dropout(0.1),
    Dense(output_dim=hidden4_num_units, input_dim=hidden3_num_units,  kernel_regularizer=l2(0.0001), activation='relu'),
    Dropout(0.1),
    Dense(output_dim=output_num_units, input_dim=hidden4_num_units, activation='softmax'),
 ])

# Model summary
model.summary()

## Define optimizer: Stochastic gradient descent 
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Compile model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# Fit model                
history=model.fit(X_train, 
          Y_train,
          epochs=100, 
          batch_size=100, 
          validation_split = 0.2,
          verbose=1,
          )

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

# Prediction at test data set
Y_pred = model.predict(X_test)
score = model.evaluate(X_test, Y_test,batch_size=100, verbose=1)
print(score)
print("Baseline Error: %.2f%%" % (100-score[1]*100))

# Class Predictions
# A class prediction is given the finalized model and one or 
# more data instances, predict the class for the data instances
test_class = model.predict_classes(X_test)

## Confusion matrix
print("Confussion matrix:\n%s" %
      metrics.confusion_matrix(Y_test, test_class))

# precission and accuracy:
print("Classification report:\n%s" %
      metrics.classification_report(Y_test, test_class))
print("Classification accuracy: %f" %
      metrics.accuracy_score(Y_test, test_class))

# Predicton at Grid location
# Import grid point data
grid = 'grid_data.csv'
grid_point = pd.read_csv(grid)
# Create grid data frame with ten bands
X_grid = grid_point[['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']]
# Create xy-coordinated data frame
xy_grid=grid_point[['ID','x', 'y']]

# Scale the grid set
X_grid = preprocessing.scale(X_grid)

# Prediction
grid_class = pd.DataFrame(model.predict_classes(X_grid))

# Join xy-coordinates with predicted grid_class data frames
grid_class_xy = pd.concat([xy_grid, grid_class], axis=1, join_axes=[xy_grid.index])

# Rename predicted class column to Class_ID
grid_class_xy.columns.values[3] = 'Class_ID'

## Load landuse ID file
id = 'Landuse_ID.csv'
LU_ID = pd.read_csv(id)
print(LU_ID)

## Join  Landuse class 
grid_class_final=pd.merge(grid_class_xy, LU_ID, left_on='Class_ID', right_on='Class_ID', how='left')

## Write CSV files 
grid_class_final.to_csv('predcted_landuse_class.csv', index=False)

### Create a spatial data frame
from osgeo import ogr, gdal
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj, transform

# Define geometry
geometry = [Point(xy) for xy in zip(grid_class_final.x, grid_class_final.y)]
# Define projection (UTM zone 17N)
crs = {'init': 'epsg:26917'}
## Create Geodata frame
gdf = gpd.GeoDataFrame(grid_class_final, crs=crs, geometry = geometry)
list(gdf)
# Save as ESRI shape file
gdf.to_file("predicted_landuse.shp")

# A list of "random" colors (for a nicer output)

gdf.plot(column='Description',  legend=True,figsize=(6, 6))





