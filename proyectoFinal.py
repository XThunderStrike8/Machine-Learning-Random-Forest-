from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos
DatosBD = load_breast_cancer()

# Convertir los datos en un DataFrame de pandas
dfDB = pd.DataFrame(data=DatosBD.data, columns=DatosBD.feature_names)
dfDB['target'] = DatosBD.target

# Ver las primeras filas de los datos
print(dfDB.head())

#cantidad de valores de la columna target
#0 - B Y 1 - M

# Contar los valores de la columna 'target'
count_values = dfDB['target'].value_counts()

# Imprimir los resultados
print(count_values)

#Variables de entrenamiento (Drop se usa para retirar a target y la columna del mismo)
#Axis 1  -> Columnas, Axis 0 -> Filas.

X = dfDB.drop(['target'], axis = 1)
Y = dfDB['target']

#Variables de prueba y entrenamiento a usar con su respectivo tamaño de datos (25%)
x_train, x_final, y_train, y_final = train_test_split(X,Y, test_size=.25)

# Entrenar el modelo RandomForestClassifier
Tree = RandomForestClassifier(n_estimators=100, criterion='entropy')
Tree.fit(x_train, y_train)

# Obtener las predicciones en los datos de prueba y entrenamiento
y_hat_test = Tree.predict(x_final)
y_hat_train = Tree.predict(x_train)

# Obtener la importancia de las características (feature importance)
feature_importance = Tree.feature_importances_

# Imprimir las predicciones
print("Predicciones en los datos de prueba:")
print(y_hat_test)
print()

print("Predicciones en los datos de entrenamiento:")
print(y_hat_train)
print()

# Imprimir la importancia de las características
print("Importancia de las características:")
print(feature_importance)

#Variables de datos con las predicciones 
y_hat_test = Tree.predict(x_final)
y_hat_train = Tree.predict(x_train)

#Accuracy para saber la precision de los datos
print(accuracy_score(y_final,y_hat_test))

#Se muestran las metricas busacadas que son Precision, Recall, f1-score y support
print(classification_report(y_final, y_hat_test))

#Se utiliza para hacer visualizaciones de datos
plt.figure(figsize=(20, 5)) 
sns.boxplot(data=dfDB)
plt.xticks(rotation=90)
plt.show()

#Se asigna a una variable para la escala o normalizacion
escala = MinMaxScaler() 

#Fit_transform es un metodo para calcular los parametros requeridos, para que después se le asigné los parametros calculados 
#a los datos estandarizados

df_norm = pd.DataFrame(escala.fit_transform(dfDB), columns=dfDB.columns)
print(df_norm.head())

#Se utiliza para hacer visualizaciones de datos
plt.figure(figsize=(20, 5)) 
sns.boxplot(data=df_norm)
plt.xticks(rotation=90)
plt.show()

#Para visualizar la matriz de confusión
cm = confusion_matrix(y_final, y_hat_test)
print("Matriz de confusión:")
print(cm)

""" 
⠄⠄⠄⠄⢠⣿⣿⣿⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣯⢻⣿⣿⣿⣿⣆⠄⠄⠄
⠄⠄⣼⢀⣿⣿⣿⣿⣏⡏⠄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣧⢻⣿⣿⣿⣿⡆⠄⠄
⠄⠄⡟⣼⣿⣿⣿⣿⣿⠄⠄⠄⠈⠻⣿⣿⣿⣿⣿⣿⣿⣇⢻⣿⣿⣿⣿⠄⠄
⠄⢰⠃⣿⣿⠿⣿⣿⣿⠄⠄⠄⠄⠄⠄⠙⠿⣿⣿⣿⣿⣿⠄⢿⣿⣿⣿⡄⠄
⠄⢸⢠⣿⣿⣧⡙⣿⣿⡆⠄⠄⠄⠄⠄⠄⠄⠈⠛⢿⣿⣿⡇⠸⣿⡿⣸⡇⠄
⠄⠈⡆⣿⣿⣿⣿⣦⡙⠳⠄⠄⠄⠄⠄⠄⢀⣠⣤⣀⣈⠙⠃⠄⠿⢇⣿⡇⠄
⠄⠄⡇⢿⣿⣿⣿⣿⡇⠄⠄⠄⠄⠄⣠⣶⣿⣿⣿⣿⣿⣿⣷⣆⡀⣼⣿⡇⠄
⠄⠄⢹⡘⣿⣿⣿⢿⣷⡀⠄⢀⣴⣾⣟⠉⠉⠉⠉⣽⣿⣿⣿⣿⠇⢹⣿⠃⠄
⠄⠄⠄⢷⡘⢿⣿⣎⢻⣷⠰⣿⣿⣿⣿⣦⣀⣀⣴⣿⣿⣿⠟⢫⡾⢸⡟⠄.
⠄⠄⠄⠄⠻⣦⡙⠿⣧⠙⢷⠙⠻⠿⢿⡿⠿⠿⠛⠋⠉⠄⠂⠘⠁⠞⠄⠄⠄
⠄⠄⠄⠄⠄⠈⠙⠑⣠⣤⣴⡖⠄⠿⣋⣉⣉⡁⠄⢾⣦⠄⠄⠄⠄⠄⠄⠄⠄
#Firmado por ThunderStrike8 y mis ganas de morir
 """

