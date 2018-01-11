import os
import imageConverter
import keras
from keras.models import model_from_json
import numpy as np
import random
from keras.optimizers import Adam

num_classes = 17
target_label = "plus"
num_target_images = 10

# prediction 
tagetLabelPath = os.path.join(os.getcwd(), 'jpgImages', target_label)
tergetImages = os.listdir(tagetLabelPath)
random.shuffle(tergetImages)
randomSelectedFileNames = [image for idx, image in enumerate(tergetImages) if idx < num_target_images ]
randomSelectedFilePathes = [os.path.join(tagetLabelPath ,image) for image in randomSelectedFileNames ]

(x_predict, y_predict) = imageConverter.getFeedDataforPrediction(randomSelectedFilePathes)
img_rows = x_predict.shape[1]
img_cols = x_predict.shape[2]
channels = x_predict.shape[3]
x_predict = x_predict.reshape(x_predict.shape[0], img_rows, img_cols, channels)

# Normalize data.
x_predict = x_predict.astype('float32') / 255

model = model_from_json(open(os.path.join(os.getcwd(),'saved_models','icon_resnet_model.json')).read())
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])
model.load_weights(os.path.join(os.getcwd(),'saved_models','icon_resnet_model.h5'))
model.summary()

for i in range(num_target_images) :  # len(x_predict)):
#     predicted_vector = model.predict(X_test[i:i+1,0:1], batch_size=1, verbose=0)
    y_prob = model.predict(np.array([x_predict[i]]), batch_size=1, verbose=0)
    y_classes = y_prob.argmax(axis=-1)
#     print('predicted vector', predicted_vector)
    print('predicted class', y_classes, 'when y', y_predict[i])
    if y_classes[0] == y_predict[i]: print("trial {0} matched : {1}".format(i,randomSelectedFileNames[i]))
    else : print("trial {0} failed : {1}".format(i,randomSelectedFileNames[i]))

# if __name__ == "__main__":
#     while(label == exit):
#         print("Choose label name in")
#         print(os.listdir())
