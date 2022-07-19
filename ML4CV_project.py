import time
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K 
from keras.layers.core import Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense
from keras.applications.mobilenet import preprocess_input
from sklearn.model_selection import train_test_split
from myUtils import find_next_file_history, save_history, show_history, save_elapsedTime
from sklearn.metrics import confusion_matrix

histFileName = 'historyMobilenet.csv'
dirHistFileName = './history'
start = time.time()

#df = pd.read_csv("C:/Users/User/Documents/Corsi Uni/ML4CV/ProjectML4CV/ResempledDataset.csv", dtype=str)
df = pd.read_csv("C:/Users/User/Documents/Corsi Uni/ML4CV/ProjectML4CV/HAM10000_metadata.csv", dtype=str)
new_df, testdf = train_test_split(df,test_size=0.20,random_state=42 )
traindf, valid = train_test_split(new_df,test_size=0.10,random_state=42 )
inputShape = (224,224,3)
imgSize = (224,223)

starting_model =  tf.keras.applications.MobileNetV3Small(input_shape=inputShape,
    alpha=1.0,
    minimalistic=False,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling="avg", 
    dropout_rate=0.5,
    include_preprocessing=False)

#starting_model.trainable = False

x = starting_model.output
x = Dense(1024,activation='relu')(x)
#x = Dropout(0.5)(x)
#x = Dense(512,activation='relu')(x)
x = Dropout(0.8)(x)
preds = Dense(7,activation='softmax')(x)
model = Model(inputs=starting_model.input,outputs=preds)

for layer in model.layers[:200]:
    layer.trainable = False

for  layer in model.layers[200:]:
    layer.trainable = True 

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator= train_datagen.flow_from_dataframe(
dataframe = traindf,
directory = "C:/Users/User/Documents/Corsi Uni/ML4CV/ProjectML4CV/Images",
x_col = "image_id",
y_col = "dx",
batch_size = 32,
seed = 42,
shuffle = True,
class_mode = "categorical",
target_size = imgSize)

valid_generator= train_datagen.flow_from_dataframe(
dataframe = valid,
directory = "C:/Users/User/Documents/Corsi Uni/ML4CV/ProjectML4CV/Images",
x_col = "image_id",
y_col = "dx",
batch_size = 32,
seed = 42,
shuffle = True,
class_mode = "categorical",
target_size = imgSize)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['acc'])
step_size_train = train_generator.n//train_generator.batch_size
step_size_val = valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,epochs=10,
                    validation_data=valid_generator,validation_steps=step_size_val)

model.save('C:/Users/User/Documents/Corsi Uni/ML4CV/ProjectML4CV/models/skin_modelSmall10eFT_SH.h5')
end = time.time()
elapsedTime= (end - start)
print("Elapsed Time:")
print("\t\t{:.3f}s".format(elapsedTime))

show_history(history)
finalHistoryFile=find_next_file_history(dirHistFileName,histFileName)
save_history(history.history,finalHistoryFile)

save_elapsedTime(elapsedTime,finalHistoryFile)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator= test_datagen.flow_from_dataframe(
dataframe = testdf,
directory = "C:/Users/User/Documents/Corsi Uni/ML4CV/ProjectML4CV/Images",
x_col = "image_id",
y_col = None,
batch_size = 1,
seed = 42,
shuffle = False,
class_mode = None,
target_size = imgSize)

test_generator.reset()
pred=model.predict_generator(test_generator, steps= test_generator.n//test_generator.batch_size, verbose=1)

predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)

conf_mat =confusion_matrix(testdf["dx"],results["Predictions"])
print(conf_mat)