import numpy as np
from tensorflow import keras 
from keras import backend as K 
from keras.utils import load_img, img_to_array
from keras.models import load_model

labels = ["Bowen's disease","basal cell carcinoma","benign keratosis-like lesions","dermatofibroma","melanoma","melanocytic nevi","vascular lesions"]

def prepare_image(file):

    img_path = 'C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/Test/'
    img = load_img(img_path + file, target_size=(224,224))
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

model = load_model('C:/Users/User/Documents/Corsi Uni/ML4CV/Exam/models/skin_model.h5')
preprocessed_image = prepare_image('ISIC_0025069.jpg')
predictions = model.predict(preprocessed_image)
classes = predictions.argmax(axis=1)
print(labels[classes[0]], "  ", predictions[0][classes[0]])