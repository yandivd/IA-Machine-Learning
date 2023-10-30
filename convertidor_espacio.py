import tensorflow as tf
import numpy as np

metros = np.array([5, 13, 15, 4, 8, 9], dtype=float)
kms = np.array([0.005, 0.013, 0.015, 0.004, 0.008, 0.009], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print('Entrenando')
modelo.fit(metros, kms, epochs=1000, verbose=False)
print('Modelo entrenado')

choice = 'y'
while (choice=='y'):
    entrada = input('Inserte un valor: ')
    resultado = modelo.predict([float(entrada)])
    print('El resultado es {}'.format(resultado))
    choice = input('Desea volver a probar? (y/n) ')