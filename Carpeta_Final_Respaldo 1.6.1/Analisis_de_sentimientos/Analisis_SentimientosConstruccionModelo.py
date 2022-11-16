from pickletools import optimize
from tabnanny import verbose
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer #Version 2.5.0 de tensorflow

tipoRellenado="post"
longitudMaxima=100
tipoTruncado="post"

dbAnalisisSentimientos = pd.read_excel("basededatosAnalisisSentimientosEntrenamiento.xlsx") 
#print(dbAnalisisSentimientos)

dbAnalisisSentimientos.head()
dbAnalisisSentimientos=pd.DataFrame(dbAnalisisSentimientos["Texto_Emocion"].str.split(";", 1).tolist(),columns=["Texto","Emocion"]) #DIVIDIR COLUMNAS DE TEXTO ## Esta parte se refiere a el fotrmato de la base de datos.
#print(dbAnalisisSentimientos.head)

dbAnalisisSentimientos["Emocion"].unique()   #Se etiquetan las emociones unicas
print("***************")
print("SENTIMIENTOS UNICOS: ",dbAnalisisSentimientos["Emocion"].unique())

emocionesEtiquetadas={"NoAplica":2, "Triste":3, "Indiferente":4, "Preocupado":5, "NadaFeliz":6, "PocoFeliz":7, "MedianamenteFeliz":8, "Feliz":9, "MuyFeliz":10} #Creacion de diccionario para etiquetar emociones.
dbAnalisisSentimientos.replace(emocionesEtiquetadas, inplace=True) #Remplazo de emociones por numeros.
print(dbAnalisisSentimientos)

frasesEntrenamiento=[]
etiquetasEntrenamiento=[]

for i in range(len(dbAnalisisSentimientos)):
    respuestasFrasesPP=dbAnalisisSentimientos.loc[i,"Texto"]
    frasesEntrenamiento.append(respuestasFrasesPP)

    etiqueta=dbAnalisisSentimientos.loc[i,"Emocion"]
    etiquetasEntrenamiento.append(etiqueta)
#print("***************")
#print("validacion frase entrenamiento: ",frasesEntrenamiento[7])   # print para validar el for
#print("validacion etiqueta de entrenamiento: ",etiquetasEntrenamiento[7])   #print para validar el for


tamanoVocablo=1000 #Definicion de los parametros para tokenizar.
dimensionesIncrustradas=16
oovTok="<OOV>"
tamanoEntrenamiento=2000

tokenizador=Tokenizer(num_words=tamanoVocablo, oov_token=oovTok)
tokenizador.fit_on_texts(frasesEntrenamiento) #Entrenar la lista de frases de entrenamiento.

word_index=tokenizador.word_index
for i in range(len(word_index)): #Validar el contenido del diccionario.
   print(word_index)

secuenciaEntrenamiento=tokenizador.texts_to_sequences(frasesEntrenamiento)        #Transformar texto en datos de entrenamiento
print("secuencia de entrenamiento 0: ",secuenciaEntrenamiento[0])
print("secuencia de entrenamiento 1: ",secuenciaEntrenamiento[1])
print("secuencia de entrenamiento 2: ",secuenciaEntrenamiento[2])
print("secuencia de entrenamiento 6: ",secuenciaEntrenamiento[6])



rellenadoEntrenamiento=pad_sequences(secuenciaEntrenamiento, maxlen=longitudMaxima, padding=tipoRellenado, truncating=tipoTruncado)
print("entrenamiento Rellenado: ", rellenadoEntrenamiento)
#se define el modelo de la red.
 #la capa retiene el valor maximo de matrix de caracteristicas # esta capa tienen 3 argumentos. Número de filtros para la operación convolución, segundo tamaño del kernel define la ventana del kernel tercer define funcion activación de la capa
## es la red lstm es para la memoria a corto paso, para no perder la información, tiene 3 partes, la primera puerta de entrada eligue si va a recordar la información precedente de la entrada anterior, segunda es una puerta de salida, aprende nueva información de la nueva entrada, puerta de olvido pasa la información actualizada a la siguente capa le llaman también celda 


import numpy as np

rellenadoEntrenamiento=np.array(rellenadoEntrenamiento) 
etiquetasEntrenamiento=np.array(etiquetasEntrenamiento)     # se convierte las secuencias rellenadas y etiquetas en arreglos numpy


from tensorflow.keras.models import Sequential #Construcción de capas para la RNC
from tensorflow.keras.layers import Embedding,LSTM,Dense #primer capa oculta
from tensorflow.keras.layers import Conv1D,Dropout,MaxPooling1D 


#Construcción del modelo.
modelo=tf.keras.Sequential([
    Embedding(tamanoVocablo,dimensionesIncrustradas,input_length=longitudMaxima),
    Dropout(0.2), 

    Conv1D(filters=256,kernel_size=3,activation="relu"),
    MaxPooling1D(pool_size=3),

    Conv1D(filters=128,kernel_size=3,activation="relu"),
    MaxPooling1D(pool_size=3),

    LSTM(128),

    Dense(256,activation="relu"),
    Dropout(0.2),
    Dense(128,activation="relu"),
    Dense(64,activation="softmax")])

#etapa de compilación.
modelo.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
modelo.summary()

print("*********************")
print("ENTRENAMIENTO DEL MODELO")

#Etapa de entrenamiento del modelo.
numero_epocas=400
historial=modelo.fit(rellenadoEntrenamiento,etiquetasEntrenamiento,epochs=numero_epocas,verbose=2)
modelo.save("RNCSentimientos.h5")


#Etapa de prueba del modelo
respuestasFrasesPP=["Me pongo contento cuando el pago esta proximo porque puedo comprar utiles escolares y con lo que me queda compro cosas que me gustan","hago los pagos escolares y lo que me queda lo ahorro"]
secuenciaRespuestaPP=tokenizador.texts_to_sequences(respuestasFrasesPP) 
rellenadoRespuestaPP=pad_sequences(secuenciaRespuestaPP, maxlen=longitudMaxima, padding=tipoRellenado, truncating=tipoTruncado)

prediccionRespuestaPP=modelo.predict(rellenadoRespuestaPP) # Utilizamos metodo predict para hacer una preddicción de prueba del modelo
prediccionClase= np.argmax(prediccionRespuestaPP,axis=1)
# encontrando el resultado utilizando el metodo arrgmax


print("*********************")
print("PREDICCIÓN DEL MODELO")
print(prediccionClase)

import tensorflow.keras
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

respuestasFrasesPP=["Me pongo nervioso porque a los demas les llega y a mi no","No me preocupo porque se que me va a llegar"]
#Esta es la parte de tokenización
tokenizador=Tokenizer(num_words=10000,oov_token="<OOV>")
tokenizador.fit_on_texts(respuestasFrasesPP)

#Crear un diccionario word index
word_index=tokenizador.word_index
secuenciaRespuestaPP=tokenizador.texts_to_sequences(respuestasFrasesPP)
print("*****************")
print("Diccionario de respuestas PP")
print(secuenciaRespuestaPP[0:2])

#Rellenar la secuencia
rellenadoRespuestaPP=pad_sequences(secuenciaRespuestaPP, maxlen=100, padding="post", truncating="post")
print("*****************")
print("Secuencia rellenada")
print(rellenadoRespuestaPP[0:2])

#Definir el modelo usando el archivo(RNCSentimientos.h5 este es el modelo entrenado)
modelo=tensorflow.keras.models.load_model("RNCSentimientos.h5")

#Probar el modelo
prediccionRespuestaPP=modelo.predict(rellenadoRespuestaPP)
print("*****************")
print("Prueba de modelo")
print(prediccionRespuestaPP)

#Mostrar resultado.
print("******************")
print("Resultado Prediccion Clase")
prediccionClase= np.argmax(prediccionRespuestaPP,axis=1)
print(prediccionClase)
