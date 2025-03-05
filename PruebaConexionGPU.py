import tensorflow as tf
import torch


print("CUDA Available:", tf.config.list_physical_devices('GPU'))

print("CUDA Available:", torch.cuda.is_available())