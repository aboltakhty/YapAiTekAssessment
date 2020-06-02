
# import packages
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Define variables
num_features=	6				# number of features
num_outputs =	num_features	# number of outputs
maxIter =		20				# maximum iterations in training process
T =				1				# next time

# load the dataset
dataset = pandas.read_csv( '../Data/psi_df_2016_2019.csv' ,usecols=numpy.arange(num_features),engine='python', skipfooter = 3)

DataHeader =	dataset.columns.tolist()
dataset =		dataset.values.astype('float32')

# normalize the dataset
scaler =		MinMaxScaler(feature_range=(0, 1))
dataset =		scaler.fit_transform(dataset)

# split into train and test sets
train_size =	int(len(dataset) * 0.70)
test_size =		len(dataset) - train_size
train, test =	dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Define a function to split Data
def create_dataset(dataset, look_back=T, num_features= num_features, num_outputs=num_outputs):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a =	dataset[i:(i+look_back), 0:num_features]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0:num_outputs])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t and Y=t+1
look_back =			T
trainX, trainY =	create_dataset(train, look_back, num_features, num_outputs)
testX, testY =		create_dataset(test, look_back, num_features, num_outputs)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, num_features))
testX = numpy.reshape(testX, (testX.shape[0], 1, num_features))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=num_features))
model.add(Dense(num_outputs))

model.compile(loss= 'mean_squared_error' , optimizer= 'adam' )
model.fit(trainX, trainY, epochs= maxIter, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict =	scaler.inverse_transform(trainPredict)
trainY =		scaler.inverse_transform(trainY)
testPredict =	scaler.inverse_transform(testPredict)
testY =			scaler.inverse_transform(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print( 'Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print( 'Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
dataset = scaler.inverse_transform(dataset)

for i in range(num_features):
	plt.figure(i)
	plt.plot(dataset[:,i],label="Dataset")
	plt.plot(trainPredictPlot[:,i],label="Train")
	plt.plot(testPredictPlot[:,i],label="Test")
	plt.title(DataHeader[i])
	plt.legend()

plt.show()

