import tensorflow as tf
import numpy as np

# Datos de entrenamiento
metros_cuadrados = np.array([100, 400, 200, 250, 300], dtype=float)
cuartos = np.array([2, 5, 2, 1, 3], dtype=int)
banos = np.array([1, 3, 1, 1, 2], dtype=int)
precios = np.array([2000, 10000, 3000, 2500, 4000], dtype=float)

# Combina todas las características en un solo tensor
caracteristicas = np.column_stack((metros_cuadrados, cuartos, banos))

capa = tf.keras.layers.Dense(units=3, input_shape=[3])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error',
)

print('Iniciando entrenamiento...')
modelo.fit(caracteristicas, precios, epochs=1000, verbose=False)
print('Entrenamiento Finalizado')

print('Haciendo predicción')
nueva_casa = np.array([100, 1, 1], dtype=float).reshape(1, 3)  # Nuevas características
precio = modelo.predict(nueva_casa)
print('El precio de la nueva casa es de ${:.2f}'.format(precio[0][0]))
