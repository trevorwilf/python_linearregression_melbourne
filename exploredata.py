import pandas as pd
import os
import csv
import requests
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import math

## 
# functions
##
def get_mae(max_leaf_nodes, train_x, val_x, train_y, val_y):
    """this is desingned to be used in conjuction with a for loop to 
        test for the optimal number of leaf_nodes to avoid over and under fitting.

    Args:
        max_leaf_nodes (_type_): current leaf nodes for testing
        train_x (_type_): training data set x
        val_x (_type_): testing data set x
        train_y (_type_): training data set y
        val_y (_type_): testing data set y
    """    
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_x, train_y)
    preds_val = model.predict(val_x)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


##

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

# find max leaf nodes to prevent under and over fitting
best_max_leag_nodes = 0
best_mae = 1000000
for max_leaf_node in list(range(5, 10000, 50)):
    cur_mae = get_mae(max_leaf_node, x_train, x_test, y_train, y_test)
    if cur_mae < best_mae:
        best_max_leag_nodes = max_leaf_node
        best_mae = cur_mae
        
    
    print(f'current max leaf node: {max_leaf_node} \t\t mae: {cur_mae} \t\t best mae: {best_mae} \t\t best max leaf node: {best_max_leag_nodes}')
        

# build model
#prices are what we want to predict

melbourne_model = DecisionTreeRegressor(max_leaf_nodes=best_max_leag_nodes, random_state=0)


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