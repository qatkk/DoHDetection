from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import ipaddress
import os

pd.set_option('display.max_columns', None)

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, '../Models/RandomForest_fold.joblib')
dataset_path = os.path.join(current_dir, '../Datasets/doh_decrypted_traffic_2.csv')

rf_model = load(model_path)

test_dataset = pd.read_csv(dataset_path, sep=",")

test_dataset['SourceIP'] = test_dataset['SourceIP'].apply(lambda ip: int(ipaddress.ip_address(ip)))
test_dataset['DestinationIP'] = test_dataset['DestinationIP'].apply(lambda ip: int(ipaddress.ip_address(ip)))

labels = test_dataset['Classification']

test_dataset = test_dataset.drop(columns=['Classification', 'TimeStamp'], axis=1)
test_dataset = test_dataset.rename(columns={'ResponseTimeCoefficientofVariation': 'ResponseTimeTimeCoefficientofVariation'})
test_dataset = test_dataset.rename(columns={'ResponseTimeMean': 'ResponseTimeTimeMean'})
test_dataset = test_dataset.rename(columns={'ResponseTimeMedian': 'ResponseTimeTimeMedian'})
test_dataset = test_dataset.rename(columns={'ResponseTimeMode': 'ResponseTimeTimeMode'})
test_dataset = test_dataset.rename(columns={'ResponseTimeSkewFromMedian': 'ResponseTimeTimeSkewFromMedian'})
test_dataset = test_dataset.rename(columns={'ResponseTimeSkewFromMode': 'ResponseTimeTimeSkewFromMode'})
test_dataset = test_dataset.rename(columns={'ResponseTimeStandardDeviation': 'ResponseTimeTimeStandardDeviation'})
test_dataset = test_dataset.rename(columns={'ResponseTimeVariance': 'ResponseTimeTimeVariance'})

predictions = rf_model.predict(test_dataset)
print(f"Random forest predictions:{predictions}")
print(pd.Series(predictions).value_counts())
print(labels.value_counts())

accuracy = accuracy_score(labels, predictions)
precision = precision_score(labels, predictions)  
recall = recall_score(labels, predictions)
f1 = f1_score(labels, predictions)

todos_son_false = all(not x for x in predictions)
print(todos_son_false)

# Resultados
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


