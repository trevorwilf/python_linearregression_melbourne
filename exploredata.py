import pandas as pd
import os
import csv
import requests
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

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

####
# Ceate test and train data sets
train, test = train_test_split(melbourne_data, test_size=0.2)

#select a features/columns
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
#now subset them out
y_test = train.Price
x_test = train[melbourne_features]
x_test = pd.DataFrame.reset_index(x_test)
x_test.describe()
x_test.head()
x_test.columns

y_train = train.Price
x_train = train[melbourne_features]
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



fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#ROC curve for a specific class here for the class 2
roc_auc[2]

