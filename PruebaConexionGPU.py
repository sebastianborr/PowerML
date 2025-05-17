import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for i, gpu in enumerate(gpus):
    print(f"GPU:{i} -", gpu.name)
