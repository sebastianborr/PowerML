import tensorflow as tf

# Verificar dispositivos disponibles
devices = tf.config.list_physical_devices()
print("Dispositivos disponibles:", devices)

# Verificar si está usando la GPU
gpu_devices = tf.config.list_physical_devices('GPU')
if len(gpu_devices) == 0:
    print("TensorFlow no está utilizando una GPU.")
else:
    print("TensorFlow está utilizando la GPU:", gpu_devices)
