import tensorflow as tf
import numpy as np

# entradaMLC = input('Inserte la lista de valores en mlc divididos por comas: ')
# mlc = entradaMLC.split(',')
# mlc = np.array([int(valor) for valor in mlc], dtype=float)
# for i in mlc:
#     print(i)

# entradaMN = input('Inserte ahora los valores correspondientes en mn tambien divididos por comas: ')
# mn = entradaMN.split(',')
# mn = np.array([int(valor) for valor in mn], dtype=float)
# for i in mn:
#     print(i)

mlc = np.array([7, 5, 89, 2, 67, 45, 32, 25, 965, 1], dtype=float)
mn = np.array([1645, 1175, 20915, 470, 15745, 10575, 7520, 5875, 226775, 235], dtype=float)

# capa = tf.keras.layers.Dense(units=1, input_shape=[1])
entrada = tf.keras.layers.Dense(units=3, input_shape=[1])
oculta2 = tf.keras.layers.Dense(units=3)
oculta3 = tf.keras.layers.Dense(units=3)
oculta4 = tf.keras.layers.Dense(units=3)
# oculta5 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([entrada, oculta2, oculta3, oculta4, salida])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print('Entrenando')
historial = modelo.fit(mlc, mn, epochs=1000, verbose=True)
print('Modelo entrenado')

import  matplotlib.pyplot as plt

plt.xlabel('#Epoca')
plt.ylabel('Magnitud de Perdida')
plt.plot(historial.history['loss'])
plt.savefig("divisas2.png")

choice = 'y'
while (choice=='y'):
    entrada = input('Inserte un valor: ')
    resultado = modelo.predict([float(entrada)])
    print('El resultado es {}'.format(resultado))
    choice = input('Desea volver a probar? (y/n) ')