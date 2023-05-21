import pandas as pd
import os
import csv
import requests
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import math

# get the file and save it
#url = 'https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot/download?datasetVersionNumber=5'
#response = requests.get(url)

#if (not os.path.exists('./data')):
#    os.makedirs('./data')

#with open('out.csv', 'w') as f:
#    writer = csv.writer(f)
#    for line in response.iter_lines():
#        writer.writerow(line.decode('utf-8').split(','))

melbourne_file_path = './data/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data.describe()
melbourne_data.columns

#drop rows that contain any NA's
melbourne_data = melbourne_data.dropna(axis=0)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
melbourne_data_sub = melbourne_data[melbourne_features]
y = melbourne_data.Price

####
# Ceate test and train data sets
x_train, x_test, y_train, y_test = train_test_split(melbourne_data_sub, y, random_state = 0)

#now subset them out
x_test = pd.DataFrame.reset_index(x_test)
x_test.describe()
x_test.head()
x_test.columns

x_train = pd.DataFrame.reset_index(x_train)
x_train.describe()
x_train.head()
x_train.columns

# build model
#prices are what we want to predict

melbourne_model = DecisionTreeRegressor(random_state=1)


#fit the model
melbourne_model.fit(x_train, y_train)
melbourne_model.feature_importances_


# results
print("Making predictions for the following 5 houses:")
print(x_test.head())
print("The predictions are")
#print(melbourne_model.predict(x.head()))

predictions = melbourne_model.predict(x_test)

#inserts NAN
x_test = x_test.join(pd.DataFrame({'predictions': list(predictions)})) 
x_test = x_test.join(pd.DataFrame({'actual': list(y_test)}))

#works with errors 
#x_test.loc['predictions'] =  list(predictions)
#x_test.loc['actual'] = list(y_test)

print(x_test.head())


## now validate the model using rmse
MAE = round(mean_absolute_error(x_test.actual, x_test.predictions), ndigits=2)
RMSE = round(math.sqrt(mean_absolute_error(x_test.actual, x_test.predictions)), ndigits=2)

print(f'The MAE is: {MAE}')
print(f'The RMSE is: {RMSE}')