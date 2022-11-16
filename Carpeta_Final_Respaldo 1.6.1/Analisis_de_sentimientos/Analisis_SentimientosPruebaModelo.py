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

#sigue el front end.