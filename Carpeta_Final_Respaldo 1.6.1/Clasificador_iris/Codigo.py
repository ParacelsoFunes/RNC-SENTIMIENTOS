import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC                         # libreria para maquinas de vectores de soporte clasificación
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


iris = pd.read_csv("C:/Users/billy/Desktop/Perceptron/Perceptron codigo/iris.csv")
iris = iris.drop("Id",axis=1)
print("archivo_de_entrenamiento_iris:",iris.head()) #print("archivo_de_entrenamiento_iris:",iris.head())
print("*********")
print("INFORMACION DATA SET")
print("*********")
print(iris.info())                                  # da informacion estadistica

print("*********")
print("DESCRIPCION DATA SET")
print("*********")
print(iris.describe())                              # datos a obtener con el describe mean desviación estandar std min mediana y cuatriles.

print("*********")
print("DESCRIPCION de las especies de iris")
print("*********")
print(iris.groupby("Species").size())              # función para agrupar alumnos con mayor probabilidad de abandono escolar. buscar una propiedad para ordenar alumnos con menor posibilidad de abandono escolar                

# Grafica de Sepalo/ longitud versus ancho
graficaDeSepalo = iris[iris.Species =="Iris-setosa"].plot(kind = "scatter", x = "PetalLengthCm", y = "PetalWidthCm", color = "red", label = "Setosa")
iris[iris.Species =="Iris-versicolor"].plot(kind = "scatter", x = "PetalLengthCm", y = "PetalWidthCm", color = "green", label = "Versicolor", ax = graficaDeSepalo)
iris[iris.Species =="Iris-virginica"].plot(kind = "scatter", x = "PetalLengthCm", y = "PetalWidthCm", color = "blue", label = "Virginica", ax = graficaDeSepalo)

graficaDeSepalo.set_xlabel("Petalo-Longitud")
graficaDeSepalo.set_ylabel("Petalo-Ancho")
graficaDeSepalo.set_title("Petalo-Longitud versus Ancho")
plt.show()

'''Trabajar con tres modelos 1 modelo sepalo y datos del petalo
Segundo modelo datos del sepalo
Tercero Modelo del datos del petalo'''

'''Separar todos los datos con las caracteristicas excepto la columna de especies y se le va a llamar X
Poner los datos con las etiquetas o respuestas que son la columna de las especies Se le llama Y'''

X = np.array(iris.drop(["Species"],1))
Y = np.array(iris["Species"])


X_entrenamiento, X_prueba, Y_entrenamiento, Y_prueba = train_test_split(X, Y, test_size = 0.2)                    #Separar datos de entrenamiento y datos de prueba
print("Son {} datos de entrenamiento Y {} datos para prueba".format(X_entrenamiento.shape[0], X_prueba.shape[0])) #Shape da el numero de dimensiones de la matriz

#Algoritmo de regresión logistica
algoritmo = LogisticRegression()                                                                                  #Se crea el modelo
algoritmo.fit(X_entrenamiento, Y_entrenamiento)                                                                   #Entrenamiento del modelo
Y_prediccion = algoritmo.predict(X_prueba) 
print("*********")
print("Regresion logistica {}".format(algoritmo.score(X_entrenamiento,Y_entrenamiento)))
print("*********")

#Algoritmo maquinas de vectores de soporte
algoritmo = SVC()
algoritmo.fit(X_entrenamiento, Y_entrenamiento)
Y_prediccion =algoritmo.predict(X_prueba)
print("*********")
print("Maquinas de vectores  de soporte{}".format(algoritmo.score(X_entrenamiento,Y_entrenamiento)))
print("*********")

#Algoritmo vecinos mas cercanos
algoritmo = KNeighborsClassifier(n_neighbors = 5) 
algoritmo.fit(X_entrenamiento, Y_entrenamiento)
Y_prediccion =algoritmo.predict(X_prueba)
print("*********")
print("Vecinos as cercanos{}".format(algoritmo.score(X_entrenamiento,Y_entrenamiento)))
print("*********")

#Arboles de decisiones
algoritmo = DecisionTreeClassifier() 
algoritmo.fit(X_entrenamiento, Y_entrenamiento)
Y_prediccion =algoritmo.predict(X_prueba)
print("*********")
print("Arbol de decisiones{}".format(algoritmo.score(X_entrenamiento,Y_entrenamiento)))
print("*********")









'''import csv

def leer_archivo_de_entrenamiento(archivo):
    with open(archivo,'r') as archivo_de_entrenamiento:
        leer_archivo = csv.read(archivo_de_entrenamiento)
        informacion_archivo = []
        for columna in leer_archivo:
            informacion_archivo.append(columna)
            print(informacion_archivo)
            
        for datos in informacion_archivo:
            return información_archivo
        
dato = leer_archivo_de_entrenamiento('C:/Users/billy/Desktop/Perceptron/Perceptron codigo/iris.csv')
iris_setosa = datos[0:70]
iris_versicolor = datos[70:140]
iris_virginica = datos[140:210]'''