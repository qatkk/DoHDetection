# -*- coding: utf-8 -*-
"""ClassificationDNSoverHTTP

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dJ8TLdzqO2KubIUOiL_OI8YUpRUT5uuX
"""

print("Hello Miurelcita")

from google.colab import drive
drive.mount('/content/drive/')

!pip install pandas scikit-learn

"""Descomprimir y revisar datos"""

import zipfile
import os
import pandas as pd

zip_path = '/content/drive/My Drive/Total_CSVs.zip'
csv_path = './l2-malicious.csv'

if not os.path.exists(csv_path):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./')
    print("We")
else:
    print("Listo")


doh_pd = pd.read_csv("l1-doh.csv")
nondoh_pd = pd.read_csv("l1-nondoh.csv")

combined_df = pd.concat([doh_pd, nondoh_pd], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)


print(combined_df.head)

"""Random Forest Clasfication"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

X = combined_df.drop(columns=['Label', 'SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort','TimeStamp'])
y = combined_df['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


rf_model = RandomForestClassifier(random_state=42)


rf_model.fit(X_train, y_train)


y_pred = rf_model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))