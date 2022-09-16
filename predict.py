import os
import pandas as pd
import numpy as np
import keras

from preprocess_image import images_to_keras_dataset
from deepfake_model import DeepfakeModel
from tensorflow.keras.preprocessing.image import ImageDataGenerator


model = keras.models.load_model('model_0.76')


batch_size=32
target_size=(150, 150)

test_df = pd.read_csv("../hfactory_magic_folders/tooling_for_the_data_scientist/deepfakes_detection/test.csv", dtype=str, sep=";")

test_df.loc[:, "image_id"] = test_df.loc[:, "image_id"] + ".jpg"


datagen = ImageDataGenerator(rescale=1./255)
data_dir = "../hfactory_magic_folders/tooling_for_the_data_scientist/deepfakes_detection/images/"

pred_dataset = datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=data_dir,
    x_col="image_id",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    label=None,
    target_size=target_size)

predictions = model.predict(pred_dataset)
score = predictions.argmax(axis=1)
test_df['label'] = score
test_df["image_id"] = test_df["image_id"].str.rstrip('.jpg')
test_df.to_csv('predictions.csv', sep=";", index=False)