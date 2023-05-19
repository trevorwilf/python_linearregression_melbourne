import pandas as pd
import os
import csv
import requests

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

