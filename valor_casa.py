import tensorflow as tf
import numpy as np

# Datos de entrenamiento

metros_cuadrados = np.random.randint(100, 500, 1000)
cuartos = np.random.randint(2, 6, 1000, dtype=int)
banos = np.random.randint(1, 4, 1000, dtype=int)
precios = np.random.randint(2000, 10000, 1000)


# Combina todas las características en un solo tensor
caracteristicas = np.column_stack((metros_cuadrados, cuartos, banos))

capa = tf.keras.layers.Dense(units=3, input_shape=[3])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
)

print('Iniciando entrenamiento...')
historial = modelo.fit(caracteristicas, precios, epochs=1000, verbose=True)
print('Entrenamiento Finalizado')

import  matplotlib.pyplot as plt

plt.xlabel('#Epoca')
plt.ylabel('Magnitud de Perdida')
plt.plot(historial.history['loss'])
plt.savefig("valorCasas.png")


print('Haciendo predicción')
nueva_casa = np.array([50, 1, 1], dtype=float).reshape(1, 3)  # Nuevas características
precio = modelo.predict(nueva_casa)
print('El precio de la nueva casa es de ${:.2f}'.format(precio[0][0]))
