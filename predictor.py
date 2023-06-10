import tensorflow as tf
import numpy as np
import os
def predictWithModel(model,imgPath):
    image = tf.io.read_file(imgPath)  
    image = tf.image.decode_png(image,channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image,[60,60])
    image = tf.expand_dims(image,axis=0)
    
    predictions = model.predict(image)
    predictions = np.argmax(predictions)
    return predictions

if __name__=='__main__':
    imgPath = os.path.join(os.getcwd(),'Data/Test/30/00041.png')
    model = tf.keras.models.load_model('./Model')
    prediction = predictWithModel(model,imgPath)
    print(f'Prediction = {prediction}')