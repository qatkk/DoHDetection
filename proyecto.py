import zipfile
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # Normalizar datos
#Validación cruzada
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
##########################Modelo2############################
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
##########################Modelo2############################

# Descomprimir archivo si no existe el CSV
zip_path = './Total_CSVs.zip'
csv_path = './l2-malicious.csv'

if not os.path.exists(csv_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('./')
    print("Archivos descomprimidos.")
else:
    print("Archivos listos.")

# Leer los datos
doh_pd = pd.read_csv("l1-dohio.csv")
nondoh_pd = pd.read_csv("l1-nondohc.csv")

# Combinar, mezclar y resetear índices
combined_df = pd.concat([doh_pd, nondoh_pd], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Manejar valores NaN e infinitos
combined_df = combined_df.replace([float('inf'), -float('inf')], None)
combined_df = combined_df.dropna()

# Separar características y etiquetas
#Dataset1
#X = combined_df.drop(columns=['Label', 'SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort','TimeStamp'])
#Dataset2
X = combined_df.drop(columns=['DoH', 'SourceIP', 'DestinationIP', 'SourcePort', 'DestinationPort','TimeStamp'])

y = combined_df['DoH']

#Imprimir los datos originales antes de la normalizacion
#print("Datos originales:")
#print(X.head())

# Normalizar características
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


#Como el modelo no acepta etiquetas en formato de txt
# La etiqueta DoH -> 0       La etiqueta NonDoH -> 1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#y = label_encoder.fit_transform(combined_df['Label'])


# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



##########################Modelo1############################

# Entrenar modelo
#rf_model = RandomForestClassifier(random_state=42)
#rf_model.fit(X_train, y_train)

# Evaluar modelo
#y_pred = rf_model.predict(X_test)

##########################Modelo1############################


##########################Modelo2############################
#xgb_model = XGBClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)

##########################Modelo2############################

# Validación cruzada (5 folds)
scores = cross_val_score(xgb_model, X, y, cv=5, scoring='accuracy')


# Mostrar resultados de la validación cruzada
#print("Scores por fold:", scores)
#print("Precisión promedio:", scores.mean())
#print("Desviación estándar:", scores.std())



print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


#Imprimir los datos normalizados
#print("Datos normalizados:")
#print(X.head())
