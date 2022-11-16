#Matriz de recompensas se utiliza aprendisaje por refuerzo (prueba y error) respuesta correcta se asigna una recompensa aprende de su experiencia
#Componentes: 

# -1 en caso de penalización (el agente no se mueve camino incorrecto)
# 0 en caso que el agente se pueda mover (se cambia el estado del agente)
# 100 en caso de llegar a la meta, deja me moverse el agente
#La meta sería que no abandone y los estados por donde debe transitar para no abandonar
# las filas van a representar los estados y las columnas representan las acciones a realizar por el agente. es decir la maquina.
# Agente(la maquina) 
# El entorno(con quien va a interactuar el agente es decir la situación)
# los estados(el camino que sirve para rastrear posición del agente)
# Acciones son los movimiento que realiza el agente a travez de los estados esta es la parte del refuerzo
# Recompensas (se establecen las recompensas en caso de acertar y las penalizaciones en caso de no acertar) aqui va la matriz.
#Matriz de recompensas con 34 estados, las filas representan las acciones que el agente tiene que hacer para llegar a la meta.
# Las filas son las acciones.
# Se tiene que construir la matriz conforme a la base de datos. 

import numpy as np
import random


#Matriz de recompensas.

'''recompensas = np.array([
[-1, 0, -1, -1, -1, 100],
[0,-1, -1, 0, -1, -1],
[-1, -1, -1, -1, -1],
[-1, 0, -1, -1, -1, -1],
[-1, -1, -1, -1, -1, 1],
[0, -1, -1, -1, -1, 100]
])'''

#El resultado final del arreglo es encontrar que el agente elija las acciones que le den mayor recompensa positiva
#  y evitar las negativas paso 1.

# para hacer la construcción de matrices se puede utilizar rpa (automatización robotizada)
recompensas = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas2 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas3 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1,],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas4 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensa5 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensa6 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas7 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas8 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,-1,0,0,-1],[-1,-1,0,0,-1],[-1,-1,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,-1,-1],[0,-1,0,-1,-1],[0,-1,-1,-1,-1],[0,-1,-1,-1,-1],[100,-1,-1,-1,-1],[-1,-1,0,-1,100]])
recompensa9 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,-1,0,0,-1],[-1,-1,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,-1,-1],[0,-1,0,-1,1],[0,-1,-1,0,-1],[0,-1,-1,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas10 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas11 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas12 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[0,-1,0,0,-1],[100,-1,0,-1,-1],[-1,-1,0,-1,-1]])
recompensas13 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,0,0,-1],[0,-1,0,-1,-1],[0,-1,0,-1,-1],[0,-1,0,-1,-1],[0,-1,-1,0,-1],[100,-1,0,-1,-1],[-1,0,0,-1,100]])
recompensas14 = np.array([[-1,0,-1,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,0,-1],[-1,0,0,100,-1],[0,-1,-1,0,-1],[0,-1,-1,0,-1],[0,-1,-1,-1,-1],[0,-1,-1,-1,-1],[0,-1,-1,-1,-1],[0,-1,-1,-1,-1],[100,-1,-1,-1,-1],[-1,-1,-1,-1,100]])

matriz_recompensas_agrupadas = [recompensas,recompensas2,recompensas3,recompensas4,recompensa5,recompensa6,recompensas7,recompensas8,recompensa9,recompensas10,recompensas11,recompensas12,recompensas13,recompensas14]
#for i in matriz_recompensas_agrupadas:
 #   print("matriz recompensas agrupadas:" ,matriz_recompensas_agrupadas)

def estado_de_inicio():
    return np.random.randint(0,6)

estado_de_inicio()

def optener_acciones(estado_actual, matriz_recompensas):
    acciones_disponibles =[]
    print("matriz de recompensas", matriz_recompensas)
    
    for accion in enumerate(matriz_recompensas[estado_actual]):
        if accion[1] != -1:
            acciones_disponibles.append(accion[0])

    escoger_accion = random.choice(acciones_disponibles)

    print(" Eleccion aleatoria de la acción ", escoger_accion)
    return escoger_accion


# es el ciclo para contar y determinar ruta más corta de cada una de las matrices.
estado_actual = 1
for contador_de_matrices in matriz_recompensas_agrupadas:
    accion = optener_acciones(estado_actual,contador_de_matrices)
    print("*********************")
    print("accion:",contador_de_matrices)
    print(accion)

# paso 2 el agente aprenda la acción realizada y decida la acción en consecuencia. Se va utilizar el metodo de q learning 
# Matrix de calidad permite determinan que tan util es la acción para generar una recompensa.
