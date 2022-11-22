import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('dino_dragon_10_0.899.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with tf.io.gfile.GFile('dino-dragon-model.tflite', 'wb') as f:
    f.write(tflite_model)