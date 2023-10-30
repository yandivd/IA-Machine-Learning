import tensorflow
import numpy

celsius = numpy.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = numpy.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

##### Usando una sola capa ######
# capa = tensorflow.keras.layers.Dense(units=1, input_shape=[1])
# modelo = tensorflow.keras.Sequential([capa])

##### Usando capas ocultas #####
oculta1 = tensorflow.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tensorflow.keras.layers.Dense(units=3)
salida = tensorflow.keras.layers.Dense(units=1)
modelo = tensorflow.keras.Sequential([oculta1, oculta2, salida])

modelo.compile(
    optimizer =  tensorflow.keras.optimizers.Adam(0.1),
    loss = 'mean_squared_error'
)

print('Comenzando entrenamiento.....')
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=True)
print('Modelo Entrenado') 

import  matplotlib.pyplot as plt

plt.xlabel('#Epoca')
plt.ylabel('Magnitud de Perdida')
plt.plot(historial.history['loss'])
plt.savefig("mi_grafico.png")

print('Hagamos una prediccion:....')
resultado = modelo.predict([100.0])
print('El resultado es '+ str(resultado) + 'fahrenheit')

print('Valores internos del modelo')
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())